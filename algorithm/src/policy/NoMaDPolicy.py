import numpy as np
import os
import yaml
import torch
import argparse
import time
from collections import deque
from pathlib import Path

from vint_train.utils import to_numpy, transform_images, load_model
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vint_train.training.train_utils import get_action
class NoMaDPolicy:
    """Visual navigation policy using NoMaD diffusion model"""
    
    def __init__(self, model_config_path, model_weights_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model configuration
        with open(model_config_path, 'r') as f:
            self.model_params = yaml.safe_load(f)
        
        self.context_size = self.model_params["context_size"]
        self.context_queue = deque(maxlen=self.context_size + 1)
        
        # Load model
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Model weights not found at {model_weights_path}")
        
        print(f"Loading model from {model_weights_path}")
        self.model = load_model(model_weights_path, self.model_params)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize noise scheduler
        self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    
    def add_observation(self, rgb_image):
        """Add observation to context queue"""
        # Convert numpy array to PIL Image
        from PIL import Image as PILImage
        if isinstance(rgb_image, np.ndarray):
            rgb_image = PILImage.fromarray(rgb_image)
        self.context_queue.append(rgb_image)
    
    def predict_waypoint(self, waypoint_index=2, num_samples=8 ,mask = None ,goal = None):
        """Predict waypoint using the diffusion model"""
        if len(self.context_queue) <= self.context_size:
            return None
        
        from PIL import Image as PILImage
        if isinstance(goal, np.ndarray):
            goal = PILImage.fromarray(goal)

        if goal is None:
            goal = torch.randn((1, 3, *self.model_params["image_size"])).to(self.device)
        else:
            goal = transform_images(
                [goal], 
                self.model_params["image_size"], 
                center_crop=False
            ).to(self.device)

        if mask is None:
            mask = torch.ones(1).long().to(self.device)  # Ignore the goal
        else:
            mask = torch.zeros(1).long().to(self.device)

        # Transform images
        obs_images = transform_images(
            list(self.context_queue), 
            self.model_params["image_size"], 
            center_crop=False
        )
        obs_images = obs_images.to(self.device)
        

        # Create fake goal (exploration mode)

        
        # Infer action
        with torch.no_grad():
            # Encode vision features
            obs_cond = self.model(
                'vision_encoder', 
                obs_img=obs_images, 
                goal_img=goal, 
                input_goal_mask=mask
            )
            
            # Repeat for multiple samples
            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(num_samples, 1, 1)
            
            # Initialize action from Gaussian noise
            noisy_action = torch.randn(
                (num_samples, self.model_params["len_traj_pred"], 2), 
                device=self.device
            )
            naction = noisy_action
            
            # Initialize scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
            
            # Diffusion denoising process
            start_time = time.time()
            for k in self.noise_scheduler.timesteps[:]:
                # Predict noise
                noise_pred = self.model(
                    'noise_pred_net',
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )
                
                # Inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
            
            elapsed = time.time() - start_time
            print(f"Inference time: {elapsed:.3f}s")
        
        # Convert to numpy
        naction = to_numpy(get_action(naction))
        
        # Select first sample (could use heuristic here)
        naction = naction[0]
        
        # Get chosen waypoint
        chosen_waypoint = naction[waypoint_index]
        
        return chosen_waypoint, naction
