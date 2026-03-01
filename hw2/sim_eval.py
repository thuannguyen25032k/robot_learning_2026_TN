
import dill
import h5py
import numpy as np
import torch

def get_text_tokens(cfg, tokenizer, text_model, goal, model=None):
    """
    Get the text tokens/embeddings for the goal.
    If a `model` with `encode_text_goal` is provided, use it so callers don't need a buffer.
    """
    if model is not None:
        return model.encode_text_goal(goal, tokenizer=tokenizer, text_model=text_model)
    # fallback to legacy behaviour
    if cfg.dataset.encode_with_t5:
        goal_ = np.zeros((cfg.max_block_size, cfg.n_embd), dtype=np.float32)
        input_ids = tokenizer(goal, return_tensors="pt").input_ids
        goal_t = text_model.encoder(input_ids).last_hidden_state.detach().cpu().numpy() ## Get the goal embedding
        goal_[:len(goal_t[0]), :] = goal_t[0][:cfg.max_block_size] ## Overwrite just the zeros up to the size of this vector, smaller vectors will have < max_block_size
    else:
        goal_ = " " * cfg.max_block_size
        goal_ = goal[:cfg.max_block_size] + goal_[len(goal):cfg.max_block_size]
        # legacy buffer-based encoding is not available here
        raise RuntimeError("Text encoding without model requires a buffer; pass model into get_text_tokens")
    return np.expand_dims(goal_, axis=0)

def get_blocked_mask(cfg, targets=None, T=0):
    ## Compute blocked masks
    c=192 ## Number of patches/channels in the image
    mask = torch.ones((1 + (c * cfg.policy.obs_stacking) + T + c, ), device=cfg.device) ## (1, T)
    if targets is None:
        pass
    elif (torch.rand(1)[0] > 0.66):  
        mask[1 + (c * cfg.policy.obs_stacking): 1 + (c * cfg.policy.obs_stacking) + T] = torch.zeros((1,T), device=cfg.device) ## Mask goal string
    elif (torch.rand(1)[0] > 0.33):
        mask[1 + (c * cfg.policy.obs_stacking) + T: 1 + (c * cfg.policy.obs_stacking) + T + c] = torch.zeros((1,c), device=cfg.device) ## Mask goal image

def eval_model_in_sim(cfg, model, device, log_dir, env, env_unwrapped,
                      wandb, iter_, tokenizer=None, text_model=None):
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    print("Evaluating model in sim environment")
    from collections import deque
    from einops import rearrange

    rewards = []
    for j in range(cfg.sim.eval_episodes): ## Better to eval over a few different goal configurations
        obs, reset_info = env.reset()
        obs_ = get_image_from_maniskill2_obs_dict(env_unwrapped, obs)[:,:,:3]
        obs_hist = deque(maxlen=cfg.policy.obs_stacking)
        last_action = np.zeros(cfg.action_dim)  # Track last action taken
        for _ in range(cfg.policy.obs_stacking):
            obs_hist.append(obs_)
        instruction = env_unwrapped.get_language_instruction()
        # print("Reset info", reset_info)
        print("Instruction", instruction)
        frames = []
        obs_list = []
        poses_list = []
        actions_list = []
        done, truncated, timeLimit, t = False, False, 100, 0
        txt_goal = get_text_tokens(cfg, tokenizer, text_model, instruction, model=model)
        # obs_hist.append(image) ## Add the new observation to the history buffer
        while not (done or truncated or (t > timeLimit)):
            # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
            # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
            # obs = [obs_["image"] for obs_ in obs] # obs is a list of dicts
            image = np.stack(obs_hist, axis=-1)  # stack along the last dimension
            image = rearrange(image, 'h w c t -> h w (c t)')  # add batch dimension

            # Prepare observations and goal arrays (keep them as numpy until conversion)
            obs_state_np = np.array(model.preprocess_state(image), dtype=np.float32)
            goal_state_np = np.array(model.preprocess_goal_image(image[:,:,:3]), dtype=np.float32)

            # Prepare last_action tensor if available
            last_action_tensor = None
            if last_action is not None:
                last_action_tensor = torch.from_numpy(np.array(last_action[:cfg.action_dim], dtype=np.float32)).unsqueeze(0).to(device)

            # Prepare text goal tensor (handle both numpy and torch input without copying unnecessarily)
            if isinstance(txt_goal, torch.Tensor):
                txt_goal_tensor = txt_goal.clone().detach().to(device)
            else:
                txt_goal_tensor = torch.from_numpy(np.array(txt_goal)).float().to(device)

            obs_tensor = torch.from_numpy(obs_state_np).unsqueeze(0).float().to(device)
            goal_tensor = torch.from_numpy(goal_state_np).unsqueeze(0).float().to(device)

            # Build pose tensor efficiently
            pose_np = np.array(obs["extra"]["tcp_pose"], dtype=np.float32)
            pose_tensor = torch.from_numpy(pose_np).unsqueeze(0).unsqueeze(0).to(device)

            action, loss = model.forward(
                obs_tensor,
                txt_goal_tensor,
                goal_tensor,
                mask_=True,  # Masks goal image
                pose=pose_tensor,
                last_action=last_action_tensor,
            )

            action = model.decode_action(action[0]).cpu().detach().numpy() ## Add in the gripper close action
            last_action = action.copy()  # Store for next iteration
            ## If the actions are stacked into a longer vector execute the sequence of actions
            for step_ in range(cfg.policy.action_stacking):
                act_ = action[cfg.action_dim*step_:(cfg.action_dim*(step_+1))]
                obs, reward, done, truncated, info = env.step(act_)
                image = get_image_from_maniskill2_obs_dict(env_unwrapped, obs)
                image = image[:,:,:3] ## Remove last dimension of image color
                # Store the original image for video before stacking/processing
                frames.append(image)
                obs_list.append(obs)
                poses_list.append(obs["extra"]["tcp_pose"])
                actions_list.append(act_)
                reward = -(np.linalg.norm(info["eof_to_obj1_diff"]) + np.linalg.norm(info["eof_to_obj1_diff"])) ## Use a shaped reward as distance between gripper and objects
                rewards.append(reward)
                t=t+1   
                if done or truncated:
                    break
        
    
    episode_stats = info.get('episode_stats', {})
    episode_stats['rewards'] = np.mean(rewards)
    episode_stats['observations'] = obs_list
    episode_stats['poses'] = poses_list
    episode_stats['actions'] = actions_list
    # print("Episode stats", episode_stats)
    print(f"avg reward {np.mean(episode_stats['rewards']):.8f}")
    if not cfg.testing:
        wandb.log({"avg reward": np.mean(rewards)})
    
    import os
    path_ = os.path.join(log_dir, f"simple-env-{iter_}.mp4")
    import imageio
    imageio.mimsave(path_, frames, fps=20)
    episode_stats['video_url'] = path_

    if not cfg.testing:
        try:
            # Explicit format to avoid W&B warning (defaulting to gif will be removed in v0.20.0)
            wandb.log({"example": wandb.Video(path_, format="mp4")})
        except Exception as e:
            print(f"Warning: failed to log video to wandb: {e}")

    return episode_stats

import gymnasium as gym
# --- History Stacking Wrapper ---
class DictWrapper(gym.ObservationWrapper):
    # from gymnasium.spaces import Box
    """
    A wrapper that grabs the observation from a specific key in the dictionary.
    """
    def __init__(self, env, obs_key=""):
        # gym.Wrapper.__init__(self, env)
        self.env = env
        self.observation_space = gym.spaces.Box( 
            low=0,
            high=255,
            shape=(256,256,3),  # Assuming the observation is an image of size 256x256 with 3 color channels
            dtype=np.uint8)
        self._obs_key = obs_key

    def observation(self, observation):
        """
        This method is called by the gym.ObservationWrapper after the environment's
        step or reset methods return an observation.
        """
        # Add the new observation to the history buffer
        return observation[self._obs_key]
    
    def step(self, action):
        """
        Step the environment and return the observation from the specified key.
        """
        obs, reward, done, info = self.env.step(action) ## LIBERO does not return truncated
        return obs[self._obs_key][::-1, :, :], reward, done, False, obs ## Not sure why the image was upside down.

    def reset(self, **kwargs):
        """
        Reset the environment and return the observation from the specified key.
        """
        obs = self.env.reset()
        return obs[self._obs_key][::-1, :, :], obs

def eval_libero(model, device, cfg, iter_=0, log_dir="./", 
                tokenizer=None, text_model=None, wandb=None):
        # cfg, model, device, log_dir, env, env_unwrapped, buffer,
        #               wandb, iter_, tokenizer=None, text_model=None):
    
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv, DenseRewardEnv
    import os
    from libero.libero.utils import get_libero_path
    from gymnasium.wrappers import FrameStackObservation
    from einops import rearrange
    import cv2

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = cfg.sim.task_set # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()
    
    # Load initial states and goal images from Hugging Face dataset if provided
    init_states_dataset = None
    if hasattr(cfg.sim, 'libero_init_state_hf_repo') and cfg.sim.libero_init_state_hf_repo:
        print(f"Loading initial states from Hugging Face: {cfg.sim.libero_init_state_hf_repo}")
        from datasets import load_dataset
        init_states_dataset = load_dataset(cfg.sim.libero_init_state_hf_repo, split='train')
        print(f"Loaded dataset with {len(init_states_dataset)} entries")
    elif hasattr(cfg.sim, 'libero_init_state_file') and cfg.sim.libero_init_state_file:
        print(f"Loading initial states from HDF5: {cfg.sim.libero_init_state_file}")
        init_states_dataset = h5py.File(hydra.utils.get_original_cwd()+cfg.sim.libero_init_state_file, 'r')
    else:
        print("No initial states dataset provided, using default initial states from task suite")    

    trajectory_data = []
    # retrieve a specific task
    tasks = cfg.sim.eval_tasks
    success_count = 0
    for idx, task_id in enumerate(tasks):
        task = task_suite.get_task(task_id)
        task_name = task.name
        instruction = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
            f"language instruction is {instruction}, and the bddl file is {task_bddl_file}")

        # step over the environment
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 256,
            "camera_widths": 256
        }
        env = DenseRewardEnv(**env_args) # env = OffScreenRenderEnv(**env_args)
        env.seed(0)
        
        # Load initial states from dataset if available, otherwise use default
        task_description = instruction.replace(" ", "_")
        task_demos = None
        if init_states_dataset is not None:
            task_demos = [item for item in init_states_dataset if item.get('task_description') == task_description]
            num_init_states = len(task_demos)
            if num_init_states > 0:
                print(f"Loaded {num_init_states} initial states from HF dataset for task: {task_description}")
            else:
                init_states = task_suite.get_task_init_states(task_id)
                num_init_states = len(init_states)
                print(f"Using default initial states for task: {task_description}")
        else:
            init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
            num_init_states = len(init_states)
            print(f"Using default initial states for task: {task_description}")
        
        # for init_state_id in range(len(init_states)):
        for init_state_id in range(min(1, num_init_states)):  ## Just do a couple different initializations for eval
            # Load init_state and goal_img from dataset or use default
            if init_states_dataset is not None:
                # Hugging Face dataset format
                if task_demos and init_state_id < len(task_demos):
                    demo = task_demos[init_state_id]
                    init_state = np.array(demo['init_state'])
                    goal_img = np.array(demo['goal_img']) if 'goal_img' in demo and demo['goal_img'] is not None else None
                    print(f"Loaded init_state and goal_img from HF dataset for demo {init_state_id}")
                else:
                    init_state = init_states[init_state_id]
                    goal_img = None
            else:
                init_state = init_states[init_state_id]
                goal_img = None
            
            env.reset()
            env.set_init_state(init_state)
            env_ = FrameStackObservation(DictWrapper(env, obs_key="agentview_image"), cfg.policy.obs_stacking) ## Stacking the observations
            obs, info = env_.reset()

            mask = get_blocked_mask(cfg, targets=None, T=0) ## Get the blocked mask
            
            txt_goal = get_text_tokens(cfg, tokenizer, text_model, instruction, model=model)
            
            # Use goal image from HDF5 if available, otherwise use first observation
            if goal_img is not None:
                image_goal = goal_img
                print(f"Using goal image from HDF5, shape: {image_goal.shape}")
            else:
                image_goal = obs.reshape((256, 256, 3*cfg.policy.obs_stacking))[:,:,:3]
                print("Using first observation as goal image")
            frames = []
            rewards = []
            infos = []
            obs_list = []
            poses_list = []
            actions_list = []
            last_action = np.zeros(cfg.action_dim)  # Track last action taken
            done, truncated, timeLimit, t, wait_steps = False, False, cfg.sim.episode_length, 0, 0

            while not (done or truncated or (t > (timeLimit + wait_steps))):
                ## Reshape the image to the correct size and stack the hostory on the last channel dimension
                # image = obs[0]
                if t < wait_steps: ## let object stabalize before acting.
                    obs, reward, done, truncated, info = env_.step([0,0,0,0,0,0,0])
                    t += 1
                    continue
                # obs = obs.reshape((128, 128, 3*cfg.policy.obs_stacking)) ## Assuming the observation is an image of size 128x128 with 3 color channels  
                obs = rearrange(obs, 't h w c -> h w (t c)', c=3, t=cfg.policy.obs_stacking) ## Rearranging the image to have the stacked history in the last channel dimension
                # image = obs[:,:,:3] ## Remove the last dimension of the image color
                obs_state = model.preprocess_state(obs)     # Skip preprocessing for SimpleWorldModel
                goal_state = model.preprocess_goal_image(image_goal)    # Preprocessing the goal image for both SimpleWorldModel and DreamerWorldModel
                # Build pose as a single numpy array first (avoids slow tensor-from-list warning)
                pose_np = np.concatenate(
                    (
                        np.asarray(info["robot0_eef_pos"], dtype=np.float32),
                        np.asarray(info["robot0_eef_quat"][:3], dtype=np.float32),
                        np.asarray([info["robot0_gripper_qpos"][0]], dtype=np.float32),
                    ),
                    axis=-1,
                )
                pose_tensor = torch.from_numpy(pose_np).to(device).view(1, 1, -1)
                pose_ = model.encode_pose(pose_tensor)  # (1, 1, pose_dim)
 
                # Prepare last_action tensor if available  
                last_action_tensor = None
                if last_action is not None:
                    last_action_np = np.array(last_action[:cfg.action_dim], dtype=np.float32)
                    last_action_tensor = model.encode_action(torch.from_numpy(last_action_np).unsqueeze(0).unsqueeze(0).to(device))

                # Prepare tensors efficiently and avoid copying tensors unnecessarily
                if isinstance(txt_goal, torch.Tensor):
                    text_goal_tensor = txt_goal.clone().detach().to(device)
                else:
                    text_goal_tensor = torch.from_numpy(np.array(txt_goal)).float().to(device)

                observations_tensor = torch.from_numpy(np.array([[obs_state]], dtype=np.float32)).to(device)
                goal_image_tensor = torch.from_numpy(np.array([goal_state], dtype=np.float32)).to(device)

                # print(f"Step {t}, obs_state shape: {observations_tensor.shape}, text_goal shape: {text_goal_tensor.shape}, goal_image shape: {goal_image_tensor.shape}, pose shape: {pose_.shape}, last_action shape: {last_action_tensor.shape if last_action_tensor is not None else None}")
                # Note that: return_full_sequence=False by default, so we only get the next action, not the whole sequence of future actions
                out = model.forward(
                    observations=observations_tensor,
                    text_goal=text_goal_tensor,
                    goal_image=goal_image_tensor,
                    mask_=True,
                    pose=pose_,
                    prev_actions=last_action_tensor,
                )
                action = model.decode_action(out['actions'][0]).cpu().detach().numpy()
                # action = out['actions'][0].cpu().detach().numpy()  # Assuming the model's output is already in the correct action space and does not require decoding
                last_action = action.copy()  # Store for next iteration 
                ## If the actions are stacked into a longer vector execute the sequence of actions
                for step_ in range(cfg.policy.action_stacking):
                    act_ = action[cfg.action_dim*step_:(cfg.action_dim*(step_+1))]
                    image = obs  # Resize to 128x128
                    frames.append(image)
                    pose_data = np.concatenate((info["robot0_eef_pos"], info["robot0_eef_quat"][:3], [(info["robot0_gripper_qpos"][0])]), axis=-1)
                    obs_list.append(cv2.resize(obs, (64, 64))) ## Resize obs for data

                    obs, reward, done, truncated, info = env_.step(act_)
                    # Store the original image for video before stacking/processing
                    poses_list.append(pose_data)
                    actions_list.append(act_)
                    # reward = -(np.linalg.norm(info["eof_to_obj1_diff"]) + np.linalg.norm(info["eof_to_obj1_diff"])) ## Use a shaped reward as distance between gripper and objects
                    rewards.append(reward)
                    infos.append(info)
                    t=t+1   
                    # print(f"Step {t}, reward: {reward:.4f}, done: {done}, truncated: {truncated}")
                    if done or truncated:
                        print("Episode finished with success after {} timesteps".format(step_))
                        break
                if done:
                    print("Episode finished with success after {} timesteps".format(step_))
                    success_count += 1
                    break
            trajectory_data.append({
                'task_id': task_id,
                'init_state_id': init_state_id,
                'rewards': rewards,
                'infos': infos,
                'observations': obs_list,
                'poses': poses_list,
                'actions': actions_list,
            })
            import os
            video_dir = os.path.join(log_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            sub_video_dir = os.path.join(video_dir, f"{cfg.experiment.name}")
            os.makedirs(sub_video_dir, exist_ok=True)
            path_ = os.path.join(sub_video_dir, f"libero-{iter_}-task-id-{task_id}-idx-{idx}-init-id-{init_state_id}.mp4")
            import imageio
            imageio.mimsave(path_, frames, fps=20)
    episode_stats = info.get('episode_stats', {})
    episode_stats['rewards'] = np.mean([np.mean(traj['rewards']) for traj in trajectory_data])
    episode_stats['video_url'] = path_
    episode_stats['traj'] = trajectory_data
    print(f"avg reward {np.mean([np.mean(traj['rewards']) for traj in trajectory_data]):.8f}")
    if not cfg.testing:
        wandb.log({"avg reward_"+str(task_id): np.mean([np.mean(traj['rewards']) for traj in trajectory_data]), "success_rate_"+str(task_id): success_count / len(tasks), "success_count_"+str(task_id): success_count})
    if not cfg.testing:
        wandb.log({"example": wandb.Video(path_, format="mp4")})
    env.close()
    
    # Close HDF5 file if it was opened
    if init_states_dataset is not None and isinstance(init_states_dataset, h5py.File):
        init_states_dataset.close()
        print("Closed HDF5 file")
    
    return episode_stats

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="./conf", config_name="64pix-pose")
def my_main(cfg: DictConfig):
    import torch
    import wandb
    from omegaconf import DictConfig, OmegaConf
    if not cfg.testing:
        import wandb
        # start a new wandb run to track this script
        wandb.init(
            project=cfg.experiment.project,
            # track hyperparameters and run metadata
            config=OmegaConf.to_container(cfg),
            name=cfg.experiment.name,
        )
        wandb.run.log_code(".")
    # ------------
    # Train and test splits
    # Loading data
    # create RLDS dataset builder
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print("Logging to:", log_dir)
    # model = GRP(cfg)
    # model_ = torch.load("/home/gberseth/playground/mini_grp/miniGRP.pth")
    model_dir = hydra.utils.get_original_cwd()+"/mini-grp/miniGRP.pth"
    print("Original working directory:", hydra.utils.get_original_cwd())
    print ("Loading model from:", model_dir)
    if "dataset" == cfg.model.type:
        ## load the dataset
        from dreamer_model_trainer import CircularBufferDataset
        from replay_model import ReplayModel
        model_ = ReplayModel(cfg)
        dataset_buffer = CircularBufferDataset(cfg=cfg)
        model_.set_dataset(dataset_buffer)
    else:
        from grp_model import GRP
        model_ = torch.load(model_dir, pickle_module=dill)
    # model_._cgf = cfg

    tokenizer = None
    text_model = None
    if cfg.dataset.encode_with_t5: ## Load T5 model
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        tokenizer = T5Tokenizer.from_pretrained(cfg.dataset.t5_version)
        text_model = T5ForConditionalGeneration.from_pretrained(cfg.dataset.t5_version)
    
    if "libero" in cfg.simEval:
        results = eval_libero(model_.to(cfg.device), device=cfg.device, cfg=cfg,
                          iter_=0, tokenizer=tokenizer, text_model=text_model, wandb=wandb,
                          log_dir=log_dir)
    if "simple_env" in cfg.simEval:
        import simpler_env
        task_name = "widowx_carrot_on_plate"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]
        if 'env' in locals():
            print("Closing existing env")
            env.close()
            del env
        env = simpler_env.make(task_name)
        env_unwrapped = env.env.env.env ## Updated gymnasium wrapper adds lots of wrappers.
        results = eval_model_in_sim(cfg, model_.to(cfg.device), device=cfg.device, log_dir=log_dir,
                                env=env, env_unwrapped=env_unwrapped,
                                wandb=wandb, iter_=0, tokenizer=tokenizer, text_model=text_model)
        print("results:", results)

    # cbuffer.save(cfg.dataset.to_name)


if __name__ == "__main__":
    results = my_main()
    print("results:", results)