import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal, Bernoulli, Independent, OneHotCategoricalStraightThrough
from torch.distributions.utils import probs_to_logits
import numpy as np
try:
    # When imported as a package module: `from hw2.dreamerV3 import DreamerV3`
    from .networks import (
        EncoderConv,
        DecoderConv,
        RecurrentModel,
        PriorNet,
        PosteriorNet,
        RewardPredictor,
        ContinuePredictor,
        ActorNet,
        CriticNet,
    )
except ImportError:
    # When executed or imported with cwd=hw2/: `from dreamerV3 import DreamerV3`
    from networks import (
        EncoderConv,
        DecoderConv,
        RecurrentModel,
        PriorNet,
        PosteriorNet,
        RewardPredictor,
        ContinuePredictor,
        ActorNet,
        CriticNet,
    )

def symlog(x):
    """
    Symmetric log transformation.
    Squashes large values while preserving sign and small values.
    y = sign(x) * ln(|x| + 1)
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

class GRPBase(nn.Module):
    """Base class for GRP models"""
    def __init__(self, cfg):
        super(GRPBase, self).__init__()
        self._cfg = cfg
        self._action_mean = torch.tensor(self._cfg.dataset.action_mean, dtype=torch.float32, device=self._cfg.device)
        self._action_std = torch.tensor(self._cfg.dataset.action_std, dtype=torch.float32, device=self._cfg.device)
        self._stacking_action_mean = torch.tensor(np.repeat([self._cfg.dataset.action_mean], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                                 dtype=torch.float32, device=self._cfg.device)
        self._stacking_action_std = torch.tensor(np.repeat([self._cfg.dataset.action_std], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                                dtype=torch.float32, device=self._cfg.device)
        self._pose_mean = torch.tensor(self._cfg.dataset.pose_mean, dtype=torch.float32, device=self._cfg.device)
        self._pose_std = torch.tensor(self._cfg.dataset.pose_std, dtype=torch.float32, device=self._cfg.device)

    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        import numpy as _np
        import torch as _torch
        if self._cfg.dataset.encode_with_t5:
            if tokenizer is None or text_model is None:
                raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
            # TODO:    
            ## Provide the logic converting text goal to T5 embedding tensor
            pass
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


    def resize_image(self, image):
        """Resize image to match model input size"""
        import cv2
        import numpy as _np
        img = _np.array(image, dtype=_np.float32)
        img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        return img

    def normalize_state(self, image):
        """Normalize image to [-1, 1] range"""
        enc = ((image / 255.0) * 2.0) - 1.0
        return enc
    
    def preprocess_state(self, image):
        """Preprocess observation image"""
        img = self.resize_image(image)
        img = self.normalize_state(img)
        return img

    def preprocess_goal_image(self, image):
        """Preprocess goal image"""
        return self.preprocess_state(image)

    def decode_action(self, action_tensor):
        """Decode normalized actions to original action space"""
        import torch as _torch
        action_mean = self._stacking_action_mean
        action_std = self._stacking_action_std
        return (action_tensor * (action_std)) + action_mean

    def encode_action(self, action_float):
        """Encode actions to normalized space [-1, 1]"""
        import torch as _torch
        ## If the action_float has length greater than action_dim then use stacking otherwise just use normal standardiaztion vectors
        if action_float.shape[1] == len(self._cfg.dataset.action_mean):
            action_mean = self._action_mean
            action_std = self._action_std
            return (action_float - action_mean) / (action_std)  

        action_mean = self._stacking_action_mean
        action_std = self._stacking_action_std
        return (action_float - action_mean) / (action_std)
    
    def decode_pose(self, pose_tensor):
        """
        Docstring for decode_pose
        
        :param self: Description
        :param pose_tensor: Description
        self._decode_state = lambda sinN: (sinN * state_std) + state_mean  # Undo mapping to [-1, 1]
        """
        import torch as _torch
        pose_mean = self._pose_mean
        pose_std = self._pose_std
        return (pose_tensor * (pose_std)) + pose_mean
    
    def encode_pose(self, pose_float):
        """
        Docstring for encode_pose
        
        :param self: Description
        :param pose_float: Description
        self._encode_pose = lambda pf:   (pf - pose_mean)/(pose_std) # encoder: take a float, output an integer
        """
        import torch as _torch
        pose_mean = self._pose_mean
        pose_std = self._pose_std
        return (pose_float - pose_mean) / (pose_std)

class DreamerV3(GRPBase):
    def __init__(self, 
                 obs_shape=(3, 128, 128),  # Updated default to match your error
                 action_dim=6, 
                 stoch_dim=32, 
                 discrete_dim=32, 
                 deter_dim=512, 
                 hidden_dim=512, cfg=None):
        """DreamerV3 model implementation.
        
        Args:
            obs_shape: Shape of input observations (C, H, W)
            action_dim: Dimension of action space
            stoch_dim: Dimension of stochastic latent state
            discrete_dim: Number of discrete categories for stochastic state
            deter_dim: Dimension of deterministic state
            hidden_dim: Dimension of hidden layers in encoder/decoder
            cfg: Configuration object for model hyperparameters and settings
        """
        # TODO: Part 3.1 - Initialize DreamerV3 architecture
        ## Define encoder, RSSM components (GRU, prior/posterior nets), and decoder heads
        super(DreamerV3, self).__init__(cfg)
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim  # Latent dimension for stochastic state
        self.discrete_dim = discrete_dim    # Latent dimension for discrete representation predicted by a sequence model
        self.deter_dim = deter_dim  # Recurrent dimension for deterministic state
        self.hidden_dim = hidden_dim    # Hidden dimension for encoder/decoder networks
        self.encodedObsSize = hidden_dim + deter_dim  # Output dimension of the encoder
        # Define encoder for images
        self.encoder = EncoderConv(input_dim=obs_shape, output_dim=hidden_dim)
        
        # GRU for deterministic state update
        self.recurrent_net = RecurrentModel(recurrent_dim=deter_dim, latent_dim=stoch_dim*discrete_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        # Prior and posterior networks for stochastic state
        self.prior_net = PriorNet(input_dim=deter_dim, latent_dim=stoch_dim, latent_classes=discrete_dim, hidden_dim=hidden_dim)
        self.post_net = PosteriorNet(input_dim=self.encodedObsSize, latent_dim=stoch_dim, latent_classes=discrete_dim, hidden_dim=hidden_dim)
        # Decoder heads for reconstruction, reward, and continue prediction
        self.decoder = DecoderConv(input_dim=deter_dim + stoch_dim * discrete_dim, output_dim=obs_shape) # Reconstruct image from combined deterministic and stochastic state
        self.reward_head = RewardPredictor(input_dim=deter_dim + stoch_dim * discrete_dim, hidden_dim=hidden_dim)  # Predict reward from combined state
        self.continue_head = ContinuePredictor(input_dim=deter_dim + stoch_dim * discrete_dim, hidden_dim=hidden_dim)  # Predict continue flag from combined state

        # Dreamer-style actor/critic (not used by the current trainer yet).
        # These operate on RSSM features feat = concat([h,z]).
        feat_dim = deter_dim + stoch_dim * discrete_dim
        self.actor = ActorNet(input_dim=feat_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.critic = CriticNet(input_dim=feat_dim, hidden_dim=hidden_dim)

        self.type = 'dreamerV3'
        self.device = self._cfg.device
        self.to(self.device)

    # ... [Helper methods same as before] ...

    def get_initial_state(self, batch_size, device):
        return {
            'h': torch.zeros(batch_size, self.deter_dim, device=device),
            'z': torch.zeros(batch_size, self.stoch_dim * self.discrete_dim, device=device),
            'z_probs': torch.zeros(batch_size, self.stoch_dim, self.discrete_dim, device=device)
        }

    def sample_stochastic(self, logits, training=True):
        # TODO: Part 3.1 - Implement stochastic sampling
        ## Sample from discrete categorical distribution using logits
        # Expected logits shape: (B, stoch_dim, discrete_dim) or (B, stoch_dim * discrete_dim)
        if logits.dim() == 2:
            B = logits.shape[0]
            logits = logits.view(B, self.stoch_dim, self.discrete_dim)
        elif logits.dim() != 3:
            raise ValueError(
                f"logits must have shape (B, stoch_dim, discrete_dim) or (B, stoch_dim*discrete_dim); got {tuple(logits.shape)}"
            )

        # Categorical probabilities (useful for logging / KL computation).
        probs = torch.softmax(logits, dim=-1)

        # Optional unimix smoothing for stability.
        unimix = 0.01
        if unimix > 0.0:
            uniform = torch.ones_like(probs) / self.discrete_dim
            probs = (1.0 - unimix) * probs + unimix * uniform

        # For sampling we want logits corresponding to the (possibly unimixed) probs.
        logits_for_sampling = probs_to_logits(probs)

        if training:
            # Straight-through one-hot categorical sampling so gradients can flow.
            z = Independent(OneHotCategoricalStraightThrough(logits=logits_for_sampling), 1).rsample()  # (B, stoch_dim, discrete_dim)
        else:
            # Deterministic mode for evaluation/planning.
            idx = torch.argmax(logits_for_sampling, dim=-1)  # (B, stoch_dim)
            z = torch.nn.functional.one_hot(idx, num_classes=self.discrete_dim).to(logits.dtype)

        # Flatten to match the rest of the model which expects (B, stoch_dim * discrete_dim)
        z_flat = z.view(z.shape[0], self.stoch_dim * self.discrete_dim)
        return z_flat, probs

    def rssm_step(self, prev_state, action, embed=None):
        # TODO: Part 3.1 - Implement RSSM step
        ## Update deterministic state (h) with GRU, compute prior and posterior distributions
        # prev_state: dict with keys {'h', 'z', 'z_probs'}
        # action: (B, action_dim)
        # embed: encoded observation features (B, hidden_dim) or None
        if action.dim() > 2:
            # If a (B, T, A) slipped through, we only support one step here.
            action = action[:, 0]

        h_prev = prev_state['h']  # (B, deter_dim)
        z_prev = prev_state['z']  # (B, stoch_dim * discrete_dim)

        # 1) Deterministic state update via GRU
        h = self.recurrent_net(h_prev, z_prev, action)  # (B, deter_dim): the new deterministic state after observing the previous state and action

        # 2) Prior over stochastic state from new deterministic state
        prior_logits = self.prior_net(h)  # (B, stoch_dim, discrete_dim): the prior distribution over the stochastic state based on the new deterministic state

        # 3) If we have an observation embedding, compute posterior and sample from it.
        if embed is not None:
            # Posterior conditions on [h, embed]
            if embed.dim() > 2:
                embed = embed[:, 0]
            post_in = torch.cat([h, embed], dim=-1) # (B, deter_dim + hidden_dim): combine deterministic state and observation embedding for posterior computation
            post_logits = self.post_net(post_in)  # (B, stoch_dim, discrete_dim): the posterior distribution over the stochastic state based on the new deterministic state and the current observation embedding
            z, z_probs = self.sample_stochastic(post_logits, training=self.training)    # Sample from the posterior during training, and use argmax during evaluation
            return {
                'h': h,
                'z': z,
                'z_probs': z_probs,
                'prior_logits': prior_logits,
                'post_logits': post_logits,
            }

        # 4) Otherwise, sample from the prior (imagination / rollout)
        z, z_probs = self.sample_stochastic(prior_logits, training=self.training)
        return {
            'h': h,
            'z': z,
            'z_probs': z_probs,
            'prior_logits': prior_logits,
            'post_logits': None,
        }

    def forward(self, observations, prev_actions=None, prev_state=None,
                mask_=True, pose=None, last_action=None,
                text_goal=None, goal_image=None):
        # TODO: Part 3.2 - Implement DreamerV3 forward pass
        ## Encode images, unroll RSSM, and compute reconstructions and heads
        # observations: (B, T, C, H, W)
        # prev_actions: (B, T, action_dim)
        if observations is None:
            raise ValueError("DreamerV3.forward requires `observations` (B, T, C, H, W)")
        if prev_actions is None:
            raise ValueError("DreamerV3.forward requires `prev_actions` (B, T, action_dim)")

        if observations.dim() != 5:
            raise ValueError(f"observations must have shape (B, T, C, H, W); got {tuple(observations.shape)}")
        if prev_actions.dim() != 3:
            raise ValueError(f"prev_actions must have shape (B, T, A); got {tuple(prev_actions.shape)}")

        B, T, C, H, W = observations.shape
        device = observations.device

        # Initialize RSSM state
        state = prev_state if prev_state is not None else self.get_initial_state(B, device=device)

        # Encode observations frame-wise
        obs_flat = observations.reshape(B * T, C, H, W) # (B*T, C, H, W)
        embed_flat = self.encoder(obs_flat)  # (B*T, hidden_dim)
        embed = embed_flat.view(B, T, -1)    # (B, T, hidden_dim)

        hs, zs, z_probs_list = [], [], []   # hs: hidden states, zs: sampled stochastic states, z_probs_list: categorical probabilities for KL computation
        priors_logits, posts_logits = [], []    # priors_logits/posts_logits: for KL divergence losses in compute_loss
        rewards, continues = [], []     # rewards: predicted rewards, continues: predicted continue logits
        recons = []     # recons: reconstructed images

        for t in range(T):
            a_t = prev_actions[:, t, :]     # (B, action_dim): the action taken at time t
            e_t = embed[:, t, :]    # (B, hidden_dim): the encoded observation at time t

            step_out = self.rssm_step(state, a_t, embed=e_t)    # (B, deter_dim), (B, stoch_dim*discrete_dim), (B, stoch_dim, discrete_dim), (B, stoch_dim, discrete_dim)
            state = {
                'h': step_out['h'],  # (B, deter_dim): the new deterministic state
                'z': step_out['z'],  # (B, stoch_dim*discrete_dim): the sampled stochastic state (flattened)
                'z_probs': step_out['z_probs'], # (B, stoch_dim, discrete_dim): the probabilities of the categorical distribution for KL divergence computation
            }

            h_t = step_out['h']     # (B, deter_dim): the new deterministic state
            z_t = step_out['z']     # (B, stoch_dim*discrete_dim): the sampled stochastic state (flattened)
            feat_t = torch.cat([h_t, z_t], dim=-1)  # (B, deter_dim + stoch_dim*discrete_dim): the combined feature vector for decoding and heads

            hs.append(h_t) 
            zs.append(z_t)
            z_probs_list.append(step_out['z_probs'])
            priors_logits.append(step_out['prior_logits'])
            posts_logits.append(step_out['post_logits'])

            # Heads
            recons.append(self.decoder(feat_t))
            # RewardPredictor returns a Normal distribution per networks.py
            rewards.append(self.reward_head(feat_t).mean)
            # ContinuePredictor (as written) returns a distribution; keep logits/mean-like value
            cont_dist = self.continue_head(feat_t)
            if hasattr(cont_dist, 'logits'):
                continues.append(cont_dist.logits)
            else:
                continues.append(cont_dist)

        # Stack time dimension
        out = {
            'reconstructions': torch.stack(recons, dim=1),          # (B, T, C, H, W)
            'rewards': torch.stack(rewards, dim=1),                 # (B, T)
            'continues': torch.stack(continues, dim=1),             # (B, T)
            'priors_logits': torch.stack(priors_logits, dim=1),     # (B, T, stoch_dim, discrete_dim)
            'posts_logits': torch.stack(posts_logits, dim=1),       # (B, T, stoch_dim, discrete_dim) (may contain None)
            'h': torch.stack(hs, dim=1),
            'z': torch.stack(zs, dim=1),
            'z_probs': torch.stack(z_probs_list, dim=1),
            'final_state': state,
        }
        return out

    def get_feat(self, state_or_output):
        """Return Dreamer feature = concat([h, z])

        Accepts either:
        - a state dict with keys {'h','z'}
        - a forward() output dict with keys {'h','z'} shaped (B,T,*)
        """
        if isinstance(state_or_output, dict) and 'h' in state_or_output and 'z' in state_or_output:
            h = state_or_output['h']
            z = state_or_output['z']
            return torch.cat([h, z], dim=-1)
        raise ValueError("get_feat expected a dict with keys 'h' and 'z'")

    @torch.no_grad()
    def _encode_obs_seq(self, observations):
        """Encode (B,T,C,H,W) -> (B,T,E)."""
        if observations.dim() != 5:
            raise ValueError(f"observations must have shape (B,T,C,H,W); got {tuple(observations.shape)}")
        B, T, C, H, W = observations.shape
        obs_flat = observations.reshape(B * T, C, H, W)
        embed_flat = self.encoder(obs_flat)
        return embed_flat.view(B, T, -1)

    def imagine(self, start_state, horizon, deterministic=False):
        """Imagine a rollout in latent space starting from `start_state`.

        Args:
            start_state: dict containing {'h','z','z_probs'} for time t=0
            horizon: number of imagined transitions
            deterministic: if True, use actor mean action (via eval-mode sampling)

        Returns:
            dict with keys:
              - feat: (B, H, F)
              - actions: (B, H, A)
              - rewards: (B, H)
              - continues: (B, H) in [0,1]
              - values: (B, H) critic mean
        """
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")

        B = start_state['h'].shape[0]
        device = start_state['h'].device

        # We'll roll forward using the prior (embed=None).
        state = {
            'h': start_state['h'],
            'z': start_state['z'],
            'z_probs': start_state.get('z_probs', torch.zeros(B, self.stoch_dim, self.discrete_dim, device=device)),
        }

        feats, actions, rewards, continues, values = [], [], [], [], []

        for _ in range(horizon):
            feat = self.get_feat(state)  # (B, F)
            # ActorNet.forward returns an action; it supports training flag but we want differentiable path
            # through world model, not through environment.
            if deterministic:
                # crude deterministic by temporarily switching training=False
                a = self.actor(feat, training=False)
            else:
                a = self.actor(feat, training=False)

            step = self.rssm_step(state, a, embed=None)
            state = {
                'h': step['h'],
                'z': step['z'],
                'z_probs': step['z_probs'],
            }

            feat_next = self.get_feat(state)
            r = self.reward_head(feat_next).mean  # (B,)
            c_dist = self.continue_head(feat_next)
            c_logits = c_dist.logits if hasattr(c_dist, 'logits') else c_dist
            c = torch.sigmoid(c_logits)  # (B,)
            v = self.critic(feat_next).mean  # (B,)

            feats.append(feat_next)
            actions.append(a)
            rewards.append(r)
            continues.append(c)
            values.append(v)

        return {
            'feat': torch.stack(feats, dim=1),
            'actions': torch.stack(actions, dim=1),
            'rewards': torch.stack(rewards, dim=1),
            'continues': torch.stack(continues, dim=1),
            'values': torch.stack(values, dim=1),
        }

    def lambda_returns(self, rewards, values, continues, gamma=0.99, lam=0.95):
        """Compute Dreamer-style lambda returns.

        Args:
            rewards: (B,H)
            values: (B,H) bootstrap values for each step
            continues: (B,H) continuation probabilities in [0,1]

        Returns:
            returns: (B,H)
        """
        if rewards.shape != values.shape or rewards.shape != continues.shape:
            raise ValueError("rewards/values/continues must have the same shape (B,H)")

        B, H = rewards.shape
        returns = torch.zeros_like(rewards)
        next_return = values[:, -1]

        for t in reversed(range(H)):
            discount = gamma * continues[:, t]
            # One-step target is r + discount * V_{t}
            # Lambda return mixes bootstrapped target and the accumulated return.
            bootstrap = values[:, t]
            next_return = rewards[:, t] + discount * ((1 - lam) * bootstrap + lam * next_return)
            returns[:, t] = next_return

        return returns

    def compute_actor_critic_loss(self, output, horizon=15, gamma=0.99, lam=0.95, entropy_coef=0.0):
        """Compute actor/critic losses using imagination rollouts.

        This assumes the world model is already trained / being trained by compute_loss.
        We start imagination from the posterior state at the last observed time step.

        Args:
            output: dict from forward()
            horizon: imagination horizon
            gamma: discount
            lam: lambda for returns
            entropy_coef: optional entropy regularization (0 to disable)

        Returns:
            dict: {'actor_loss','critic_loss','imagine_returns_mean'}
        """

        # Start from last posterior state (B,)
        h_last = output['h'][:, -1]
        z_last = output['z'][:, -1]
        z_probs_last = output.get('z_probs', None)
        if z_probs_last is not None:
            z_probs_last = z_probs_last[:, -1]

        start_state = {
            'h': h_last,
            'z': z_last,
            'z_probs': z_probs_last,
        }

        imag = self.imagine(start_state, horizon=horizon)

        # Lambda returns for critic training
        with torch.no_grad():
            returns = self.lambda_returns(imag['rewards'], imag['values'], imag['continues'], gamma=gamma, lam=lam)

        # Critic predicts value for imagined features
        feat = imag['feat']  # (B,H,F)
        B, H, feat_dim = feat.shape
        value_pred = self.critic(feat.reshape(B * H, feat_dim)).mean.view(B, H)
        critic_loss = F.mse_loss(value_pred, returns)

        # Actor loss: maximize returns / value via features (simple Dreamer-style objective)
        # We use predicted value as a surrogate; this is common and keeps gradients local.
        actor_obj = value_pred

        # Optional entropy regularization for exploration (not required for correctness)
        if entropy_coef != 0.0:
            # ActorNet doesn't expose distribution; keep entropy term disabled by default.
            pass

        actor_loss = -actor_obj.mean()

        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'imagine_returns_mean': returns.mean().detach(),
        }

    # [Imagine method remains mostly the same, ensuring valid input shapes for heads]
    def preprocess_state(self, image):
        """Preprocess observation image"""
        img = self.resize_image(image)
        img = self.normalize_state(img)
        ## Change numpy array from channel-last to channel-first
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        # img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return img
    
    def compute_loss(self, output, images, rewards, dones, device):
        """
        Compute the total loss for DreamerV3 model training.
        
        Args:
            output: Dictionary containing model outputs (reconstructions, rewards, continues, priors_logits, posts_logits)
            images: Ground truth images tensor
            rewards: Ground truth rewards tensor
            dones: Ground truth done flags tensor
            device: Device to perform computations on
            pred_coeff: Coefficient for prediction losses (reconstruction + reward + continue)
            dyn_coeff: Coefficient for dynamics loss
            rep_coeff: Coefficient for representation loss
        
        Returns:
            Dictionary containing:
                - total_loss: Combined weighted loss
                - recon_loss: Reconstruction loss
                - reward_loss: Reward prediction loss
                - continue_loss: Continue prediction loss
                - dyn_loss: Dynamics loss (KL divergence)
                - rep_loss: Representation loss (KL divergence)
        """
        # TODO: Part 3.2 - Implement DreamerV3 loss computation
        ## Compute reconstruction, reward, KL divergence losses and combine them
        # Shapes we expect:
        # - images: (B, T, C, H, W)
        # - rewards: (B, T) or (B, T, 1)
        # - dones: (B, T) or (B, T, 1)
        # - output['reconstructions']: (B, T, C, H, W)
        # - output['rewards']: (B, T)
        # - output['continues']: (B, T) (logits)
        # - output['priors_logits']: (B, T, Z, C)
        # - output['posts_logits']: (B, T, Z, C)

        pred_coeff = float(getattr(getattr(self._cfg, 'loss_coeffs', {}), 'pred_coeff', 1.0))
        dyn_coeff = float(getattr(getattr(self._cfg, 'loss_coeffs', {}), 'dyn_coeff', 1.0))
        rep_coeff = float(getattr(getattr(self._cfg, 'loss_coeffs', {}), 'rep_coeff', 0.1))

        recons = output['reconstructions']
        pred_rewards = output['rewards']
        cont_logits = output['continues']
        priors_logits = output['priors_logits']
        posts_logits = output['posts_logits']

        if rewards.dim() == 3 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
        if dones.dim() == 3 and dones.shape[-1] == 1:
            dones = dones.squeeze(-1)

        # --- Reconstruction loss (pixel MSE in normalized space) ---
        recon_loss = F.mse_loss(recons, images)

        # --- Reward prediction loss ---
        # A simple, robust choice: MSE on raw rewards.
        reward_loss = F.mse_loss(pred_rewards, rewards)

        # --- Continue prediction loss ---
        # Continue = 1 - done
        cont_target = (1.0 - dones.float()).to(cont_logits.device)
        continue_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

        # --- KL losses between posterior and prior categorical latents ---
        # posts_logits/priors_logits: (B, T, Z, C)
        # Compute KL per (B,T,Z) and average.
        # Use log-softmax for stability.
        log_q = F.log_softmax(posts_logits, dim=-1)
        log_p = F.log_softmax(priors_logits, dim=-1)
        q = log_q.exp()
        kl = (q * (log_q - log_p)).sum(dim=-1)  # (B, T, Z)

        # Dreamer typically splits into:
        # - dynamics loss: KL(post || prior) gradient flows into prior
        # - representation loss: KL(post || prior) gradient flows into posterior
        # We approximate this with stop-grad splits.
        log_q_sg = log_q.detach()
        q_sg = q.detach()
        log_p_sg = log_p.detach()

        dyn_kl = (q_sg * (log_q_sg - log_p)).sum(dim=-1)  # (B, T, Z)
        rep_kl = (q * (log_q - log_p_sg)).sum(dim=-1)     # (B, T, Z)
        dyn_loss = dyn_kl.mean()
        rep_loss = rep_kl.mean()

        pred_loss = recon_loss + reward_loss + continue_loss
        total_loss = pred_coeff * pred_loss + dyn_coeff * dyn_loss + rep_coeff * rep_loss

        losses = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'reward_loss': reward_loss,
            'continue_loss': continue_loss,
            'dyn_loss': dyn_loss,
            'rep_loss': rep_loss,
        }

        # Optionally include actor/critic losses if enabled in cfg.
        ac_cfg = getattr(self._cfg, 'actor_critic', None)
        use_ac = bool(getattr(ac_cfg, 'enabled', False)) if ac_cfg is not None else False
        if use_ac:
            horizon = int(getattr(ac_cfg, 'horizon', 15))
            gamma = float(getattr(ac_cfg, 'gamma', 0.99))
            lam = float(getattr(ac_cfg, 'lambda_', 0.95))
            actor_coef = float(getattr(ac_cfg, 'actor_coef', 1.0))
            critic_coef = float(getattr(ac_cfg, 'critic_coef', 1.0))
            ac_losses = self.compute_actor_critic_loss(output, horizon=horizon, gamma=gamma, lam=lam)

            losses['actor_loss'] = ac_losses['actor_loss']
            losses['critic_loss'] = ac_losses['critic_loss']
            losses['imagine_returns_mean'] = ac_losses['imagine_returns_mean']
            losses['total_loss'] = losses['total_loss'] + actor_coef * losses['actor_loss'] + critic_coef * losses['critic_loss']

        return losses

