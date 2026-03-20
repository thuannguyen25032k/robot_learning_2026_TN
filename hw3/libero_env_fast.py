"""
Fast LIBERO environment - bypasses Robosuite wrapper for ~6000 SPS
"""
import hydra
import numpy as np
import gymnasium as gym
import os
import sys
from typing import Any, Mapping, cast

# Add LIBERO path
# sys.path.insert(0, '/teamspace/studios/this_studio/LIBERO')

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import DenseRewardEnv, OffScreenRenderEnv


class FastLIBEROEnv:
    """
    Fast LIBERO using raw MuJoCo for stepping.
    Observation: robot proprio + simple object positions.
    Reward: dense distance-based reward.
    """
    
    def __init__(self, benchmark_name='libero_spatial', task_id=0, max_episode_steps=300,
                 num_sim_steps=1, action_repeat=1, render_mode=None, cfg=None,
                 output_image_obs=False, image_size=64, image_camera='agentview'):
        self.benchmark_name = benchmark_name
        self.task_id = task_id
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.prev_bowl_pos = None
        self.prev_bowl_rel = None
        # Persistent "holding" estimate used by shaped reward to avoid gating flicker.
        self._hold_conf = 0.0
        self._prev_bowl_pos_world = None
        self.render_mode = render_mode
        self._num_settle_steps = 10

        sim_cfg = getattr(cfg, "sim", None)
        cfg_output_image = bool(getattr(sim_cfg, "fast_env_output_image", False)) if sim_cfg is not None else False
        cfg_image_size = int(getattr(sim_cfg, "fast_env_image_size", image_size)) if sim_cfg is not None else image_size
        cfg_image_camera = str(getattr(sim_cfg, "fast_env_image_camera", image_camera)) if sim_cfg is not None else image_camera
        self.output_image_obs = bool(output_image_obs or cfg_output_image)
        self.image_size = int(cfg_image_size)
        self.image_camera = cfg_image_camera
        
        # Get task
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_name = cfg.sim.task_set if cfg is not None else benchmark_name
        self.benchmark_name = task_suite_name
        task_suite = benchmark_dict[task_suite_name]()
        task = task_suite.get_task(task_id)
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        
        # Create base env
        enable_offscreen = (render_mode == 'rgb_array') or self.output_image_obs
        camera_size = self.image_size if self.output_image_obs else 256
        self.env = DenseRewardEnv(
            bddl_file_name=task_bddl_file,
            use_camera_obs=enable_offscreen,
            has_offscreen_renderer=enable_offscreen,
            has_renderer=False,
            camera_heights=camera_size,
            camera_widths=camera_size
            # control_freq=5,  # Control frequency (Hz)
            # controller="OSC_POSE",
        )
        
        # Action space: 7DOF arm
        self._action_dim = 7
        self._num_sim_steps = num_sim_steps
        self._action_repeat = action_repeat
        
        # Get robot and object info
        self._setup()
        
        # Observation space: either proprio/object state vector or RGB image.
        self.obs_dim = 7 + 3 * len(self.target_objects)  # 7 + 3*2 = 13
        if self.output_image_obs:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(self.image_size, self.image_size, 3), dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-10, high=10, shape=(self.obs_dim,), dtype=np.float32
            )
        
        # Action space: 7DOF arm + 1 gripper (use 7 for now)
        self._action_dim = 7
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self._action_dim,), dtype=np.float32
        )
        
    def set_init_state(self, num_states):
        self.env.set_init_state(num_states)
        
    def _setup(self):
        """Extract object names and control range"""
        # Get inner env
        inner = self.env.env if hasattr(self.env, 'env') else self.env
        
        # Get robot
        self.robot = inner.robots[0]
        
        # Get target objects (first 2)
        self.target_objects = []
        # for name in inner.objects_dict.keys():
        #     if len(self.target_objects) >= 2:
        #         break
        self.target_objects.append("akita_black_bowl_1_main")
        self.target_objects.append("plate_1_main")
        
        # Get goal position (from task)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.benchmark_name]()
        task = task_suite.get_task(self.task_id)
        self.goal_description = task.language
        
        # Control range
        self.ctrlrange = self.env.sim.model.actuator_ctrlrange[:self._action_dim]
        
    def _get_state_obs(self, obs_dict=None):
        """State observation: [eef_pos(3), eef_quat_xyz(3), gripper_qpos(1), rel_obj_offsets(6)]."""
        if obs_dict is None:
            obs_getter = getattr(self.env, "_get_observations", None)
            if callable(obs_getter):
                obs_dict = obs_getter()
            else:
                raise RuntimeError("FastLIBEROEnv requires an observation dict from env.reset/env.step")

        obs_dict = cast(Mapping[str, Any], obs_dict)

        eef_pos = np.asarray(obs_dict["robot0_eef_pos"], dtype=np.float32)
        eef_quat_xyz = np.asarray(obs_dict["robot0_eef_quat"], dtype=np.float32)[:3]
        gripper_qpos = np.asarray([obs_dict["robot0_gripper_qpos"][0]], dtype=np.float32)

        sim = self.env.sim
        rel_obj_pos = []
        for name in self.target_objects:
            body_id = sim.model.body_name2id(name)
            obj_pos = sim.data.body_xpos[body_id]
            rel_obj_pos.append((obj_pos - eef_pos).astype(np.float32))

        state = np.concatenate([eef_pos, eef_quat_xyz, gripper_qpos] + rel_obj_pos).astype(np.float32)
        return state

    def _get_image_obs(self):
        """RGB image observation from simulator camera."""
        frame = self.render(camera_name=self.image_camera, width=self.image_size, height=self.image_size)
        if frame is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.shape[-1] > 3:
            frame = frame[:, :, :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def _get_obs(self, obs_dict=None):
        if self.output_image_obs:
            return self._get_image_obs()
        return self._get_state_obs(obs_dict)
    
    def _reward(self, state, action):
        """Reusable reward function from action + observation vector.

        Expects the same layout as _get_obs:
            [qpos(7), bowl_rel_to_gripper(3), plate_rel_to_gripper(3), ...padding]
        """
        bowl_rel = state[7:10]

        # Distance: gripper to bowl from relative position
        reward_gripper_bowl = -np.linalg.norm(bowl_rel)
        # reward_gripper_bowl = np.exp(-dist_gripper_bowl * dist_gripper_bowl * 10.0)

        # Distance: gripper to plate from relative position
        plate_rel = state[10:13]
        reward_gripper_plate = - np.linalg.norm(plate_rel)
        # reward_gripper_plate = np.exp(-dist_gripper_plate * dist_gripper_plate * 10.0)

        # # Height proxy from observation-only state (relative z)
        # height_reward = max(0, min(1.0, (bowl_rel[2] + 0.05) * 10))

        # # Bowl movement penalty from consecutive observation states
        # bowl_movement_penalty = 0.0
        # if self.prev_bowl_rel is not None:
        #     bowl_movement = np.linalg.norm(bowl_rel - self.prev_bowl_rel)
        #     bowl_movement_penalty = -0.1 * bowl_movement

        # Scale: 85% gripper->bowl, 10% height, -5% bowl movement penalty
        # reward = 0.85 * reward_gripper_bowl # + 0.15 * reward_gripper_plate  # + 0.1 * height_reward + bowl_movement_penalty
        reward = reward_gripper_plate  # + 0.1 * height_reward + bowl_movement_penalty

        reward_info = {
            'reward_gripper_bowl': float(reward_gripper_bowl),
            # 'reward_height': height_reward,
            # 'bowl_movement_penalty': bowl_movement_penalty,
        }

        return reward, reward_info
    
    def _new_reward(self, state, action):
        """Reusable reward function from action + observation vector.

        Expects the same layout as _get_obs:
            [qpos(7), bowl_rel_to_gripper(3), plate_rel_to_gripper(3), ...padding]
        """
        # -----------------------------
        # Dense staged reward design
        # -----------------------------
        # Intuition:
        #   1) Always encourage reaching the bowl (eef -> bowl).
        #   2) Once the gripper is close enough to plausibly control the bowl,
        #      shift emphasis to placing (bowl -> plate) with XY alignment + height.
        #   3) Add a small action penalty for stability.
        #
        # This avoids the common local optimum "hover near bowl forever" and makes
        # the task decomposition (reach -> transport -> place) explicit.

        # From observation vector: bowl position is encoded as (bowl_pos - eef_pos)
        eef_pos = state[:3].astype(np.float32)
        gripper_qpos = float(state[6])
        bowl_rel = state[7:10].astype(np.float32)
        d_gb = float(np.linalg.norm(bowl_rel))  # distance gripper->bowl

        # Get world positions for bowl/plate for a proper place reward.
        sim = self.env.sim
        try:
            bowl_body = sim.model.body_name2id(self.target_objects[0])
            plate_body = sim.model.body_name2id(self.target_objects[1])
            bowl_pos = np.asarray(sim.data.body_xpos[bowl_body], dtype=np.float32)
            plate_pos = np.asarray(sim.data.body_xpos[plate_body], dtype=np.float32)
        except Exception:
            # Fallback to observation-only; place shaping will degrade gracefully.
            bowl_pos = None
            plate_pos = None

        # --- Reach term (smooth, dense) ---
        # exp(-k * d) gives strong gradient when close and still informative far away.
        k_reach = 10.0
        r_reach = float(np.exp(-k_reach * d_gb))

        # --- Place terms (XY + height) ---
        # Use XY alignment and a target height slightly above plate surface.
        if bowl_pos is not None and plate_pos is not None:
            bowl_minus_plate = bowl_pos - plate_pos
            d_bp_xy = float(np.linalg.norm(bowl_minus_plate[:2]))

            # Target: bowl should end up just above plate height; 2cm is a forgiving shim.
            z_target = float(plate_pos[2] + 0.02)
            d_bp_z = float(abs(bowl_pos[2] - z_target))
        else:
            # If we can't access MuJoCo bodies, fall back to gripper->plate (weaker proxy).
            plate_rel = state[10:13].astype(np.float32)
            d_bp_xy = float(np.linalg.norm(plate_rel[:2]))
            d_bp_z = float(abs(plate_rel[2]))

        k_xy = 15.0
        k_z = 20.0
        r_place_xy = float(np.exp(-k_xy * d_bp_xy))
        r_place_z = float(np.exp(-k_z * d_bp_z))

        # --- Grip / hold shaping to prevent "flicker" ---
        # If we gate placing purely on distance-to-bowl, the agent can oscillate:
        # reach bowl -> gate on -> move away without grasp -> gate off -> repeat.
        #
        # We instead build a holding confidence that increases when (a) close to bowl,
        # (b) gripper is closed (proxy), and (c) bowl is moving (proxy for contact).
        reach_eps = 0.06  # 6cm
        g_reach = float(np.clip((reach_eps - d_gb) / max(reach_eps, 1e-6), 0.0, 1.0))
        
        # Gripper closure proxy.
        # Convention is env-dependent; in many robosuite variants smaller qpos ~= more closed.
        close_lo, close_hi = 0.01, 0.04
        g_close = float(np.clip((close_hi - gripper_qpos) / max(close_hi - close_lo, 1e-6), 0.0, 1.0))

        # Geometry check: Is the bowl center physically between the fingers?
        # d_gb is the distance from the effector to the bowl.
        g_in_palm = float(d_gb < 0.03)
        
        # Bowl-follow proxy (movement indicates interaction).
        bowl_follow = 0.0
        if bowl_pos is not None:
            if self._prev_bowl_pos_world is None:
                self._prev_bowl_pos_world = bowl_pos.copy()
            bowl_delta = bowl_pos - self._prev_bowl_pos_world
            bowl_move = float(np.linalg.norm(bowl_delta))
            # Map 0..2cm to 0..1
            bowl_follow = float(np.clip(bowl_move / 0.02, 0.0, 1.0))
            self._prev_bowl_pos_world = bowl_pos.copy()

        # --- Combined Grasp Score ---
        # We want this to be 1.0 when:
        # We are at the bowl AND (We are squeezing it OR We are moving it)
        grasp_score = g_reach * np.max([g_close * g_in_palm, bowl_follow])

        # Persistent hold confidence (EMA with slower decay than growth).
        alpha_up, alpha_down = 0.2, 0.05
        if grasp_score > self._hold_conf:
            self._hold_conf = (1.0 - alpha_up) * self._hold_conf + alpha_up * grasp_score
        else:
            self._hold_conf = (1.0 - alpha_down) * self._hold_conf + alpha_down * grasp_score

        # Hysteresis: treat hold_gate as the "mode" for placing.
        hold_on, hold_off = 0.55, 0.35
        if self._hold_conf >= hold_on:
            hold_gate = 1.0
        elif self._hold_conf <= hold_off:
            hold_gate = 0.0
        else:
            hold_gate = float(np.clip((self._hold_conf - hold_off) / max(hold_on - hold_off, 1e-6), 0.0, 1.0))

        # Gate placing on "likely holding" instead of purely distance.
        g = float(hold_gate)

        # --- Control penalty (small) ---
        action = np.asarray(action, dtype=np.float32)
        act_pen = float(np.sum(action * action))
        r_act = -0.01 * act_pen

        # --- Explicit grasp/hold shaping ---
        r_grasp = float(grasp_score)
        r_hold = float(self._hold_conf)

        # --- Weighted sum ---
        w_reach = 1.0
        w_place_xy = 1.0
        w_place_z = 0.5
        w_grasp = 0.3
        w_hold = 0.2
        reward = (
            (w_reach * (1.0 - g) * r_reach)
            + (w_grasp * r_grasp)
            + (w_hold * r_hold)
            + (g * (w_place_xy * r_place_xy + w_place_z * r_place_z))
            + r_act
        )

        reward_info = {
            # distances / gate
            "d_gripper_bowl": d_gb,
            "d_bowl_plate_xy": d_bp_xy,
            "d_bowl_plate_z": d_bp_z,
            "gate_place": g,
            "gate_reach": float(g_reach),
            "gate_close": float(g_close),
            "bowl_follow": float(bowl_follow),
            "hold_conf": float(self._hold_conf),
            # components
            "reward_reach": r_reach,
            "reward_place_xy": r_place_xy,
            "reward_place_z": r_place_z,
            "reward_grasp": float(r_grasp),
            "reward_hold": float(r_hold),
            "reward_action_pen": r_act,
            # total
            "reward_total": float(reward),
        }

        return float(reward), reward_info

    def _compute_reward(self, action, state=None):
        """Dense reward computed from the same vector returned by _get_obs."""
        if state is None:
            raise RuntimeError("_compute_reward requires an explicit state when using FastLIBEROEnv")
        reward, reward_info = self._reward(state, action)

        return reward, reward_info

    def _compute_init_distance(self):
        """Compute initial distance between bowl and plate"""
        sim = self.env.sim
        try:
            init_bowl = sim.data.body_xpos[sim.model.body_name2id(self.target_objects[0])]
            init_plate = sim.data.body_xpos[sim.model.body_name2id(self.target_objects[1])]
            self.init_bowl_plate_dist = np.linalg.norm(init_bowl - init_plate)
        except:
            self.init_bowl_plate_dist = 0.1  # default
    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        obs_dict = self.env.reset()

        init_state = None
        if options is not None:
            init_state = options.get("init_state", None)
        if init_state is not None:
            obs_dict = self.env.set_init_state(init_state)

        # Compute initial distance
        self._compute_init_distance()

        # Warmup: apply no-op actions to let objects settle
        for _ in range(self._num_settle_steps):
            obs_dict, _, done, _ = self.env.step([0, 0, 0, 0, 0, 0, -1])
            if done:
                break

        state = self._get_state_obs(obs_dict)
        # Update previous bowl-relative position for next step
        self.prev_bowl_rel = state[7:10].copy()

        # Reset hold tracking.
        self._hold_conf = 0.0
        try:
            sim = self.env.sim
            bowl_pos = sim.data.body_xpos[sim.model.body_name2id(self.target_objects[0])]
            self._prev_bowl_pos_world = np.asarray(bowl_pos, dtype=np.float32).copy()
        except Exception:
            self._prev_bowl_pos_world = None

        obs = self._get_image_obs() if self.output_image_obs else state
        info = {"state_obs": state.copy()}
        return obs, info
    
    def step(self, action):
        action = np.clip(action, -1, 1)
        obs_dict = None
        done = False
        info_env = {}
        for _ in range(max(1, self._num_sim_steps)):
            obs_dict, _, done, info_env = self.env.step(action.tolist())
            if done:
                break
        
        # Compute reward
        state = self._get_state_obs(obs_dict)
        reward, reward_info = self._reward(state, action)
        
        # Check success metrics (regardless of reward function)
        sim = self.env.sim
        gripper_pos = state[:3]
        
        try:
            bowl_pos = sim.data.body_xpos[sim.model.body_name2id(self.target_objects[0])]
        except:
            bowl_pos = np.zeros(3)
        
        try:
            plate_pos = sim.data.body_xpos[sim.model.body_name2id(self.target_objects[1])]
        except:
            plate_pos = np.zeros(3)
        
        # Success 1: Gripper close to bowl (grasping) - threshold 4cm
        dist_gripper_bowl = np.linalg.norm(gripper_pos - bowl_pos)
        success_grasping = 1.0 if dist_gripper_bowl < 0.04 else 0.0
        
        # Success 2: Bowl on plate - threshold 5cm horizontal, similar height
        dist_bowl_plate = np.linalg.norm(bowl_pos[:2] - plate_pos[:2])  # x,y only
        height_close = abs(bowl_pos[2] - plate_pos[2]) < 0.05
        success_placed = 1.0 if (dist_bowl_plate < 0.05 and height_close and bowl_pos[2] > plate_pos[2]) else 0.0
        
        # Add to info
        reward_info['success_grasping'] = float(success_grasping)
        reward_info['success_placed'] = float(success_placed)
        
        # Check termination
        self.current_step += 1
        
        done = False
        truncated = self.current_step >= self.max_episode_steps
        if truncated:
            done = True
            
        info = dict(info_env)
        info.update(reward_info)
        info['state_obs'] = state.copy()

        obs = self._get_image_obs() if self.output_image_obs else state
        return obs, reward, done, truncated, info
    
    def close(self):
        self.env.close()

    def render(self, camera_name='agentview', width=256, height=256):
        if not (self.render_mode == 'rgb_array' or self.output_image_obs):
            return None

        try:
            frame = self.env.sim.render(camera_name=camera_name, width=width, height=height)
            if frame is None:
                return None
            frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            return frame[::-1, :, :]
        except Exception:
            return None
    
    @property
    def unwrapped(self):
        return self


if __name__ == '__main__':
    import time
    
    env = FastLIBEROEnv()
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")
    
    # Test speed
    start = time.time()
    episodes = 0
    steps = 0
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        steps += 1
        if done:
            env.reset()
            episodes += 1
    elapsed = time.time() - start
    
    print(f"FastLIBERO: {steps/elapsed:.0f} SPS, {episodes} episodes")
    env.close()
