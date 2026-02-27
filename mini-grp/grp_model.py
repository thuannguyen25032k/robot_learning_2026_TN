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
        # TODO: 
        ## Provide the block masking logic for the attention head
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
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
        # TODO: 
        ## Provide the logic for the GRP network
        # 1) Patch embedding layer
        self.patch_embedding = nn.Linear(cfg.patch_size * cfg.patch_size * 3, cfg.n_embd)
        # 2) Learnable token embeddings for classification and goal image tokens
        self.class_token = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        self.goal_token = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        # 3) Token embedding table for text goals (if not using T5)
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)   
        self.dropout = nn.Dropout(cfg.dropout)

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([Block(cfg.n_embd, cfg.n_head, cfg.dropout) for _ in range(cfg.n_blocks)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        # 5) Classification MLP head
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
        # Initialize learnable tokens with small random values instead of zeros

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, images, goals_txt, goal_imgs, targets=None, pose=None, mask_=True):
        n, c, h, w = images.shape
        obs_patches = get_patches_fast(images, self._cfg)
        patches_g = get_patches_fast(goal_imgs, self._cfg)
        if self._cfg.dataset.encode_with_t5:
            goals_e = goals_txt
            B, T, E = goals_txt.shape
        else:
            goals_e = self.token_embedding_table(goals_txt)
            B, E = goals_txt.shape
            T = self._cfg.max_block_size

        # TODO: 
        ## Provide the logic to produce the output and loss for the GRP
        
        # Map the vector corresponding to each patch to the hidden size dimension
        obs_tokens = self.patch_embedding(obs_patches)  # (n, n_patches, n_embd)
        goal_img_tokens = self.patch_embedding(patches_g)  # (n, n_patches, n_embd)

        # Adding classification and goal_img tokens to the tokens
        cls_token = self.class_token.expand(B, -1, -1)  # (batch, 1, n_embd)
        goal_token = self.goal_token.expand(B, -1, -1)  # (batch, 1, n_embd)
        x = torch.cat([cls_token, obs_tokens, goal_token, goal_img_tokens, goals_e], dim=1)  # (batch, total_tokens, n_embd)
        # Adding positional embedding
        pos_emb = calc_positional_embeddings(x.shape[1], self._cfg.n_embd).to(x.device)
        x = x + pos_emb.unsqueeze(0)[:, :x.shape[1], :]
        x = self.dropout(x)

        # Compute blocked masks
        att_mask = torch.ones((B, x.shape[1]), device=x.device)
        if mask_:
            total_patches = (self._cfg.image_shape[0] // self._cfg.patch_size) * (self._cfg.image_shape[1] // self._cfg.patch_size)
            obs_start = 1
            obs_end = obs_start + total_patches * self._cfg.policy.obs_stacking
            goal_img_start = obs_end + 1
            goal_img_end = goal_img_start + total_patches
            goal_text_start = goal_img_end
            goal_text_end = goal_text_start + T

            assert x.shape[1] == goal_text_end

            # Create attention mask
            # Randomly mask Text or Image goal
            rand_val = torch.rand(n, device=x.device)
            mask_text = (rand_val < 0.33).unsqueeze(1)  # (B, 1)
            mask_image = (rand_val > 0.66).unsqueeze(1)  # (B, 1)

            # Apply masking (0 = ignore, 1 = attend)
            att_mask[:, goal_text_start:goal_text_end].masked_fill_(mask_text, 0)
            att_mask[:, goal_img_start:goal_img_end].masked_fill_(mask_image, 0)

        block_mask = att_mask.unsqueeze(1)  # (B, 1, T)

        # New masking logic
        # att_mask = torch.ones((B, x.shape[1], self._cfg.n_embd), device=x.device)
        # if mask_:
        #     total_patches = (self._cfg.image_shape[0] // self._cfg.patch_size) * (self._cfg.image_shape[1] // self._cfg.patch_size)
        #     obs_start = 1
        #     obs_end = obs_start + total_patches * self._cfg.policy.obs_stacking
        #     goal_img_start = obs_end + 1
        #     goal_img_end = goal_img_start + total_patches
        #     goal_text_start = goal_img_end
        #     goal_text_end = goal_text_start + T

        #     assert x.shape[1] == goal_text_end

        #     # Create attention mask
        #     # Randomly mask Text or Image goal
        #     rand_val = torch.rand(n, device=x.device)
        #     mask_text = (rand_val < 0.33).unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        #     mask_image = (rand_val > 0.66).unsqueeze(1).unsqueeze(2)  # (B, 1, 1)

        #     # Apply masking (0 = ignore, 1 = attend)
        #     att_mask[:, goal_text_start:goal_text_end, :].masked_fill_(mask_text, 0)
        #     att_mask[:, goal_img_start:goal_img_end, :].masked_fill_(mask_image, 0)
            
        # # Apply mask to the embedding dimension as well
        # x = x * att_mask
        # Pass the mask to transformer blocks
        for block in self.blocks:
            x = block(x, mask=block_mask) 

        x = self.ln_f(x)
        
        if targets is not None:
            if self._cfg.action_space == "continuous":
                out = self.action_head(x[:, 0, :])  # (batch, action_dim * action_stacking)
                loss = F.mse_loss(out, targets)
            elif self._cfg.action_space == "discrete":
                logits = self.action_head(x[:, 0, :]).view(B, -1, 14)  # (B, action_dim * action_stacking, 14)
                targets_clamped = torch.clamp(targets, -1, 1)  # Ensure targets are within valid range
                targets_bins = ((targets_clamped + 1) / 2 * 13).long()  # Map targets from [-1, 1] to [0, 13]
                loss = F.cross_entropy(logits.permute(0, 2, 1), targets_bins)
                out = logits.argmax(dim=-1)

        else:
            if self._cfg.action_space == "continuous":
                out = self.action_head(x[:, 0, :])  # (batch, action_dim * action_stacking)
                loss = torch.tensor(0.0, device=out.device)
            elif self._cfg.action_space == "discrete":
                logits = self.action_head(x[:, 0, :]).view(B, -1, 14)  # (B, action_dim * action_stacking, 14)
                bin_idxs = logits.argmax(dim=-1).float()
                # Convert bin indices back to continuous values in [-1, 1]
                out = (bin_idxs / 13.0) * 2.0 - 1.0
                loss = torch.tensor(0.0, device=out.device)
            # print("No targets provided, loss set to 0.0")

        # Compute output and loss
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

    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        import numpy as _np
        import torch as _torch
        if self._cfg.dataset.encode_with_t5:
            if tokenizer is None or text_model is None:
                raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
            # TODO:    
            ## Provide the logic converting text goal to T5 embedding tensor
            device = text_model.device
            tokens = tokenizer(goal, return_tensors="pt").input_ids.to(device)

            with _torch.no_grad():
                embedding = text_model.encoder(tokens).last_hidden_state
            
            return embedding.to(self._cfg.device)
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
        
        """
        Docstring for decode_action
        
        :param self: Description
        :param action_tensor: Description
        self._decode_action = lambda binN: (binN * action_std) + action_mean  # Undo mapping to [-1, 1]
        """
        import torch as _torch
        ## The action tensor is of shape (batch_size, action_dim * action_stacking) so we need to repeat the mean and std per action stacking
        action_mean = _torch.tensor(np.repeat(self._cfg.dataset.action_mean, self._cfg.policy.action_stacking), dtype=action_tensor.dtype, device=action_tensor.device)
        action_std = _torch.tensor(np.repeat(self._cfg.dataset.action_std, self._cfg.policy.action_stacking), dtype=action_tensor.dtype, device=action_tensor.device)
        return (action_tensor * action_std) + action_mean
    
    def encode_action(self, action_float):
        """
        Docstring for encode_action
        
        :param self: Description
        :param action_float: Description
        self._encode_action = lambda af:   (af - action_mean)/(action_std) # encoder: take a float, output an integer
        """
        import torch as _torch
        action_mean = _torch.tensor(self._cfg.dataset.action_mean, dtype=action_float.dtype, device=action_float.device)
        action_std = _torch.tensor(self._cfg.dataset.action_std, dtype=action_float.dtype, device=action_float.device)
        return (action_float - action_mean) / action_std


@torch.no_grad()
def estimate_loss(model, dataset):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(model._cfg.eval_iters)
        for k in range(model._cfg.eval_iters):
            X, x_pose, x_goal, x_goal_img, Y = dataset.get_batch_grp(split, model._cfg, model._cfg.batch_size)
            logits, loss = model(X, x_goal, x_goal_img, Y, pose=x_pose)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
