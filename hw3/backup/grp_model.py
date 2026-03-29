import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_patches_fast(images, cfg):
    from einops import rearrange
    batch_size, height, width, channels = images.shape
    patch_size = cfg.patch_size ## n_patches = 8

    patches = rearrange(images[:,:,:,:3], 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
    if channels > 3:
        ## History stacking in the channel dimension for observations only, not goal images.
        patches = rearrange(images, 'b (h p1) (w p2) (c hs) -> b (h w hs) (p1 p2 c)', p1 = patch_size, p2 = patch_size, hs=cfg.policy.obs_stacking) ## Stack the history in the channel dimension
    return patches


def calc_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        with torch.profiler.record_function("Self-Attention"):
            out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, dropout=dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class GRP(nn.Module):
    def __init__(self, cfg, mlp_ratio=4):
        super(GRP, self).__init__()
        self._cfg = cfg
        chars = cfg.dataset.chars_list
        cfg.vocab_size = len(chars)
        # 1) Patch embedding layer
        self.patch_embedding = nn.Linear(cfg.patch_size * cfg.patch_size * 3, cfg.n_embd)
        # 2) Learnable CLS token — initialised with small noise so it has a distinct
        #    identity from the first step rather than being indistinguishable from a
        #    zero-padded position.
        self.class_token = nn.Parameter(torch.randn(1, 1, cfg.n_embd) * 0.02)
        # 3) Token embedding table for text goals (if not using T5)
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        
        # 4) Positional embedding - compute sequence length from all token types
        text_block_size = self._cfg.max_block_size
        num_obs_tokens = (self._cfg.n_patches ** 2) * self._cfg.policy.obs_stacking
        num_goal_img_tokens = self._cfg.n_patches ** 2
        seq_len = 1 + num_obs_tokens + num_goal_img_tokens + text_block_size
        self.register_buffer('pos_emb', calc_positional_embeddings(seq_len, cfg.n_embd), persistent=False)

        # 6) Transformer encoder blocks
        self.blocks = nn.ModuleList([Block(cfg.n_embd, cfg.n_head, cfg.dropout) for _ in range(cfg.n_blocks)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # 7) Action head
        if cfg.action_space == "continuous":
            self.action_head = nn.Sequential(
                nn.Linear(cfg.n_embd, cfg.n_embd * mlp_ratio),
                nn.ReLU(),
                nn.Linear(cfg.n_embd * mlp_ratio, cfg.action_dim * cfg.policy.action_stacking)
            )
        elif cfg.action_space == "discrete":
            self.action_head = nn.Sequential(
                nn.Linear(cfg.n_embd, cfg.n_embd * mlp_ratio),
                nn.ReLU(),
                nn.Linear(cfg.n_embd * mlp_ratio, cfg.action_dim * cfg.policy.action_stacking * 14)
            )
        
        # Weight initialization
        self.apply(self._init_weights)

        # Cache pose normalisation stats as buffers so they live on the right device
        self.action_mean = torch.tensor(cfg.dataset.action_mean, dtype=torch.float32)
        self.action_std = torch.tensor(cfg.dataset.action_std, dtype=torch.float32)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, images, goals_txt, goal_imgs, targets=None, pose=None, mask_=None, last_action=None):
        B = images.shape[0]
        obs_patches = get_patches_fast(images, self._cfg)
        patches_g = get_patches_fast(goal_imgs, self._cfg)
        
        obs_tokens = self.patch_embedding(obs_patches)        # (B, n_obs_patches, n_embd)
        goal_img_tokens = self.patch_embedding(patches_g)     # (B, n_goal_patches, n_embd)
        
        if self._cfg.dataset.encode_with_t5:
            goals_e = goals_txt   # already (B, T, n_embd) from T5 encoder
            T = goals_txt.shape[1]
        else:
            # goals_txt is (B, T) integer token ids from the buffer
            goals_e = self.token_embedding_table(goals_txt)   # (B, T, n_embd)
            T = self._cfg.max_block_size

        # CLS and goal separator tokens
        cls_token = self.class_token.expand(B, -1, -1)        # (B, 1, n_embd)

        # Build token sequence: [CLS | obs_patches | GOAL_SEP | goal_img_patches | text_tokens]
        x = torch.cat([cls_token, obs_tokens, goal_img_tokens, goals_e], dim=1)

        # Positional embeddings
        x = x + self.pos_emb[:x.size(1)].unsqueeze(0)
        # x = self.dropout(x)

        # Compute blocked masks
        mask_ = self._cfg.policy.random_masking_enabled
        att_mask = torch.ones((B, x.shape[1]), device=x.device)
        if mask_:
            total_patches = (self._cfg.image_shape[0] // self._cfg.patch_size) * (self._cfg.image_shape[1] // self._cfg.patch_size)
            obs_start = 1
            obs_end = obs_start + total_patches * self._cfg.policy.obs_stacking
            goal_img_start = obs_end
            goal_img_end = goal_img_start + total_patches
            goal_text_start = goal_img_end
            goal_text_end = goal_text_start + T

            assert x.shape[1] == goal_text_end

            # Create attention mask
            # Randomly mask Text or Image goal with equal probability (~33% each, ~33% neither)
            rand_val = torch.rand(B, device=x.device)
            mask_text = (rand_val < 0.33).unsqueeze(1)  # (B, 1)
            mask_image = (rand_val > 0.66).unsqueeze(1)  # (B, 1)

            # Apply masking: set 0 for tokens to ignore, 1 for tokens to attend.
            # Use index assignment (not masked_fill_ on a slice) to avoid silent no-ops
            # on non-contiguous views.
            att_mask[:, goal_text_start:goal_text_end] *= (~mask_text).float()
            att_mask[:, goal_img_start:goal_img_end]   *= (~mask_image).float()

        block_mask = att_mask.unsqueeze(1)  # (B, 1, T)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=block_mask)
        x = self.ln_f(x)

        # Predict action from CLS token
        if targets is not None:
            if self._cfg.action_space == "continuous":
                out = self.action_head(x[:, 0, :])   # (B, action_dim * action_stacking)
                targets_norm = self.encode_action(targets)  # normalise targets to match model output scale
                loss = F.mse_loss(out, targets_norm)
            elif self._cfg.action_space == "discrete":
                logits = self.action_head(x[:, 0, :]).view(B, -1, 14)
                targets_clamped = torch.clamp(targets, -1, 1)
                targets_bins = ((targets_clamped + 1) / 2 * 13).long()
                loss = F.cross_entropy(logits.permute(0, 2, 1), targets_bins)
                out = logits.argmax(dim=-1)
        else:
            if self._cfg.action_space == "continuous":
                out = self.action_head(x[:, 0, :])
                loss = torch.tensor(0.0, device=out.device)
            elif self._cfg.action_space == "discrete":
                logits = self.action_head(x[:, 0, :]).view(B, -1, 14)
                bin_idxs = logits.argmax(dim=-1).float()
                out = (bin_idxs / 13.0) * 2.0 - 1.0
                loss = torch.tensor(0.0, device=out.device)

        return (out, loss)
    
    def resize_image(self, image):
        """
        Docstring for resize_image
        
        :param self: Description
        :param image: Description
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        import cv2
        import numpy as _np
        img = _np.array(image, dtype=_np.float32)
        img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        return img

    def normalize_state(self, image):
        """
        Docstring for preprocess_state
        
        :param self: Description
        :param image: Description
        self._encode_state = lambda af:   ((af/(255.0)*2.0)-1.0) # encoder: take a float, output an integer
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        # img = _np.array(image, dtype=_np.float32)
        # img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        enc = ((image / 255.0) * 2.0) - 1.0
        # t = _torch.tensor(enc, dtype=_torch.float32, device=self._cfg.device)
        return enc
    
    def preprocess_state(self, image):
        img = self.resize_image(image)
        img = self.normalize_state(img)
        return img

    def preprocess_goal_image(self, image):
        return self.preprocess_state(image)
    
    def reset(self):
        """
        Reset the model's internal state if needed.
        """
        return None

    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        import numpy as _np
        import torch as _torch
        if self._cfg.dataset.encode_with_t5:
            if tokenizer is None or text_model is None:
                raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
            goal_ = _np.zeros((self._cfg.max_block_size, self._cfg.n_embd), dtype=_np.float32)
            with _torch.no_grad():
                input_ids = tokenizer(goal, return_tensors="pt").input_ids.to(text_model.device)
                goal_t = text_model.encoder(input_ids).last_hidden_state.detach().cpu().numpy()
            goal_[: len(goal_t[0]), :] = goal_t[0][: self._cfg.max_block_size]
            return _torch.tensor(goal_, dtype=_torch.float32, device=self._cfg.device)

        else:
            pad = " " * self._cfg.max_block_size
            goal_ = goal[:self._cfg.max_block_size] + pad[len(goal):self._cfg.max_block_size]
            try:
                stoi = {c: i for i, c in enumerate(self._cfg.dataset.chars_list)}
                ids = [stoi.get(c, 0) for c in goal_]
            except Exception:
                ids = [0] * self._cfg.max_block_size
            return _torch.tensor(_np.expand_dims(_np.array(ids, dtype=_np.int64), axis=0), dtype=_torch.long, device=self._cfg.device)

    def process_text_embedding_for_buffer(self, goal, tokenizer=None, text_model=None):
        """
        Process text goal embedding for storing in the circular buffer.
        Returns a numpy array of shape (max_block_size, n_embd) without batch dimension.
        """
        import numpy as _np
        if tokenizer is None or text_model is None:
            raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
        
        goal_ = _np.zeros((self._cfg.max_block_size, self._cfg.n_embd), dtype=_np.float32)
        input_ids = tokenizer(goal, return_tensors="pt").input_ids
        goal_t = text_model.encoder(input_ids).last_hidden_state.detach().cpu().numpy()
        goal_[:len(goal_t[0]), :] = goal_t[0][:self._cfg.max_block_size]
        return goal_

    def decode_action(self, action_tensor):
        """Decode normalized actions to original action space"""
        # import torch as _torch
        # action_mean = _torch.tensor(np.repeat([self._cfg.dataset.action_mean], self._cfg.policy.action_stacking, axis=0).flatten(), 
        #                            dtype=action_tensor.dtype, device=action_tensor.device)
        # action_std = _torch.tensor(np.repeat([self._cfg.dataset.action_std], self._cfg.policy.action_stacking, axis=0).flatten(), 
        #                           dtype=action_tensor.dtype, device=action_tensor.device)
        # return (action_tensor * (action_std)) + action_mean
        return action_tensor

    def encode_action(self, action_float):
        """Encode actions to normalized space [-1, 1]"""
        # import torch as _torch
        # ## If the action_float has length greater than action_dim then use stacking otherwise just use normal standardiaztion vectors
        # if action_float.shape[1] == len(self._cfg.dataset.action_mean):
        #     action_mean = _torch.tensor(self._cfg.dataset.action_mean, dtype=action_float.dtype, device=action_float.device)
        #     action_std = _torch.tensor(self._cfg.dataset.action_std, dtype=action_float.dtype, device=action_float.device)
        #     return (action_float - action_mean) / (action_std)  

        # action_mean = _torch.tensor(np.repeat([self._cfg.dataset.action_mean], self._cfg.policy.action_stacking, axis=0).flatten(), 
        #                            dtype=action_float.dtype, device=action_float.device)
        # action_std = _torch.tensor(np.repeat([self._cfg.dataset.action_std], self._cfg.policy.action_stacking, axis=0).flatten(), 
        #                           dtype=action_float.dtype, device=action_float.device)
        # return (action_float - action_mean) / (action_std)
        return torch.tensor(action_float, dtype=torch.float32, device=self._cfg.device)

    def decode_pose(self, pose_tensor):
        """Decode normalized pose to original pose space."""
        import torch as _torch
        pose_mean = _torch.tensor(self._cfg.dataset.pose_mean, dtype=pose_tensor.dtype, device=pose_tensor.device)
        pose_std = _torch.tensor(self._cfg.dataset.pose_std, dtype=pose_tensor.dtype, device=pose_tensor.device)
        return (pose_tensor * pose_std) + pose_mean

    def encode_pose(self, pose_float):
        """Encode pose to normalized space."""
        import torch as _torch
        pose_mean = _torch.tensor(self._cfg.dataset.pose_mean, dtype=pose_float.dtype, device=pose_float.device)
        pose_std = _torch.tensor(self._cfg.dataset.pose_std, dtype=pose_float.dtype, device=pose_float.device)
        return (pose_float - pose_mean) / pose_std
    
    def decode_state(self, state_tensor):
        """
        Docstring for decode_state
        
        :param self: Description
        :param state_tensor: Description
        self._decode_state = lambda sinN: (sinN * state_std) + state_mean  # Undo mapping to [-1, 1]
        """
        import torch as _torch
        state_mean = _torch.tensor(self._cfg.dataset.state_mean, dtype=state_tensor.dtype, device=state_tensor.device)
        state_std = _torch.tensor(self._cfg.dataset.state_std, dtype=state_tensor.dtype, device=state_tensor.device)
        return (state_tensor * (state_std)) + state_mean
    
    def encode_state(self, state_float):
        """
        Docstring for encode_state
        
        :param self: Description
        :param state_float: Description
        self._encode_state = lambda sf:   (sf - state_mean)/(state_std) # encoder: take a float, output an integer
        """
        import torch as _torch
        state_mean = _torch.tensor(self._cfg.dataset.state_mean, dtype=state_float.dtype, device=state_float.device)
        state_std = _torch.tensor(self._cfg.dataset.state_std, dtype=state_float.dtype, device=state_float.device)
        return (state_float - state_mean) / (state_std)


@torch.no_grad()
def estimate_loss(model, dataset):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(model._cfg.eval_iters)
        for k in range(model._cfg.eval_iters):
            X, x_pose, x_goal, x_goal_img, Y, last_action = dataset.get_batch_grp(split, model._cfg, model._cfg.batch_size)
            logits, loss = model(X, x_goal, x_goal_img, Y, pose=x_pose, last_action=last_action)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
