#!/usr/bin/env python3
"""
Habitat Simulation Main Function with NoMaD Navigation Model
This script runs visual navigation in Habitat-Sim using the NoMaD diffusion policy.
"""

import habitat_sim
import habitat
from habitat.utils.visualizations import maps
import numpy as np
import cv2
import imageio
import os
import yaml
import torch
import argparse
import time
from collections import deque
from pathlib import Path

# Import vint_train utilities
from vint_train.utils import  to_numpy, transform_images, load_model
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vint_train.training.train_utils import get_action

# Import action conversion utilities
from src.controller.action_conversion import WaypointToActionConverter, VelocityController


class HabitatSimulator:
    """Wrapper class for Habitat-Sim configuration and trajectory visualization"""

    def __init__(self, scene_path, enable_physics=True, resolution=(480, 640)):
        self.scene_path = scene_path
        self.enable_physics = enable_physics
        self.resolution = resolution
        self.sim = None
        self.agent = None
        self.trajectory_points = []
        self.top_down_map = None

    def make_sim_config(self):
        """Create simulator configuration"""
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self.scene_path
        sim_cfg.enable_physics = self.enable_physics

        agent_cfg = habitat_sim.agent.AgentConfiguration()

        # RGB sensor
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "color_sensor"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = list(self.resolution)

        agent_cfg.sensor_specifications = [rgb_sensor]
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def initialize(self):
        """Initialize the simulator"""
        cfg = self.make_sim_config()
        self.sim = habitat_sim.Simulator(cfg)
        self.agent = self.sim.initialize_agent(0)

        # Set random starting position
        start_position = self.sim.pathfinder.get_random_navigable_point()
        state = self.agent.get_state()
        state.position = start_position
        self.agent.set_state(state)

        # Generate static top-down map
        agent_height = self.agent.get_state().position[1]
        self.top_down_map = maps.get_topdown_map(
            self.sim.pathfinder, height=agent_height, meters_per_pixel=0.05
        )
        self.top_down_map = maps.colorize_topdown_map(self.top_down_map)

    def step(self, action):
        """Execute an action in the simulator"""
        self.sim.step(action)

    def get_observation(self):
        """Get current RGB observation"""
        obs = self.sim.get_sensor_observations()
        rgb_img = obs["color_sensor"][..., :3]  # Remove alpha channel
        return rgb_img

    def get_agent_position(self):
        """Get current agent position"""
        agent_state = self.agent.get_state()
        return agent_state.position

    def world_to_map(self, position):
        """Convert 3D world coordinates to 2D map pixel coordinates"""
        grid_loc = maps.to_grid(
            realworld_x=position[2],
            realworld_y=position[0],
            grid_resolution=self.top_down_map.shape[0:2],
            sim=self.sim,
        )
        return grid_loc[::-1]  # Return as (x, y)

    def draw_trajectory(self, trajectory_points, color=(255, 0, 0)):
        """Draw trajectory on the map"""
        if len(trajectory_points) < 2:
            return self.top_down_map.copy()

        map_with_traj = self.top_down_map.copy()

        # Draw lines between points
        for i in range(1, len(trajectory_points)):
            pt1 = trajectory_points[i - 1]
            pt2 = trajectory_points[i]
            cv2.line(map_with_traj, pt1, pt2, color, thickness=2)

        # Draw current position as a circle
        cv2.circle(map_with_traj, trajectory_points[-1], 5, (0, 255, 0), -1)
        return map_with_traj

    def reset(self):
        self.sim.reset(self.sim._default_agent_id)

    def explore(self,policy):
        CONTEXT_WARMUP=12
        NUM_STEPS=600
        WAYPOINT_INDEX=2
        NUM_SAMPLES=8
        self.sim.reset(self.sim._default_agent_id)
        video_frames = []
        trajectory_grid_points = []
        trajectory_grid_waypoints = []
        obs_video = []
        print(f"Starting exploration for {NUM_STEPS} steps...")
        control = WaypointToActionConverter()
        # Exploration loop
        for step in range(NUM_STEPS ):
            # Get current observation
            rgb_img = self.get_observation()

            # Add to policy context
            policy.add_observation(rgb_img)

            # Get agent position and update trajectory
            position = self.get_agent_position()
            grid_loc = self.world_to_map(position)
            trajectory_grid_points.append(grid_loc)

            # Predict action after enough context
            action = []
            cumulate_action = []
            if step < CONTEXT_WARMUP:
                # Random exploration during warmup
                action.append(np.random.choice(["move_forward", "turn_left", "turn_right"]))
                if step % 10 == 0:
                    print(f"Step {step}: Warmup - {action}")
            else:
                # Use policy to predict waypoint
                if step % 4 == 0:
                    result = policy.predict_waypoint(waypoint_index=WAYPOINT_INDEX, num_samples=NUM_SAMPLES)
                    waypoint, trajectory = result
                    trajectory_grid_waypoints.append(waypoint)

                if result is not None:
                    action = control.sequential_convert(waypoint=waypoint)

                    if step % 4 == 0:
                        print(f"Step {step}: Predicted waypoint {waypoint} -> {action}")

            # Execute action
            for act in action:
                self.step(act)

            # Visualize
            # Draw trajectory on map
            map_frame = self.draw_trajectory(trajectory_grid_points)


            obs_video.append(rgb_img)
            video_frames.append(map_frame)



        print("✓ Exploration completed.")
        # Save video
        output_path1 = "src/results/habitat_obs_notebook.mp4"
        print(f"Saving video to {output_path1}...")
        imageio.mimsave(output_path1, obs_video, fps=10)
        print(f"✓ Video saved to {output_path1}")


    def close(self):
        """Close the simulator"""
        if self.sim:
            self.sim.close()


class NoMaDPolicy:
    """Visual navigation policy using NoMaD diffusion model"""

    def __init__(self, model_config_path, model_weights_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model configuration
        with open(model_config_path, "r") as f:
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
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def add_observation(self, rgb_image):
        """Add observation to context queue"""
        # Convert numpy array to PIL Image
        from PIL import Image as PILImage

        if isinstance(rgb_image, np.ndarray):
            rgb_image = PILImage.fromarray(rgb_image)
        self.context_queue.append(rgb_image)

    def predict_waypoint(self, waypoint_index=2, num_samples=8):
        """Predict waypoint using the diffusion model"""
        if len(self.context_queue) <= self.context_size:
            return None

        # Transform images
        obs_images = transform_images(
            list(self.context_queue), self.model_params["image_size"], center_crop=False
        )
        obs_images = obs_images.to(self.device)

        # Create fake goal (exploration mode)
        fake_goal = torch.randn((1, 3, *self.model_params["image_size"])).to(self.device)
        mask = torch.ones(1).long().to(self.device)  # Ignore the goal

        # Infer action
        with torch.no_grad():
            # Encode vision features
            obs_cond = self.model(
                "vision_encoder", obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask
            )

            # Repeat for multiple samples
            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(num_samples, 1, 1)

            # Initialize action from Gaussian noise
            noisy_action = torch.randn(
                (num_samples, self.model_params["len_traj_pred"], 2), device=self.device
            )
            naction = noisy_action

            # Initialize scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            # Diffusion denoising process
            start_time = time.time()
            for k in self.noise_scheduler.timesteps[:]:
                # Predict noise
                noise_pred = self.model(
                    "noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond
                )

                # Inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
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


def waypoint_to_habitat_action(waypoint, threshold_forward=0.1, threshold_turn=0.3):
    """
    Convert predicted waypoint to Habitat action.

    IMPORTANT: NoMaD outputs continuous waypoints [dx, dy] in meters (relative to robot),
    but Habitat uses discrete actions (turn_left, turn_right, move_forward).

    This is a SIMPLIFIED conversion that:
    1. Calculates the angle to the waypoint
    2. If angle is large -> turn first (pure rotation, no translation)
    3. If angle is small -> move forward (pure translation, no rotation)

    Limitations:
    - Habitat cannot do simultaneous rotation + translation
    - This greedy approach may not be optimal
    - For better results, use WaypointToActionConverter from action_conversion.py

    Args:
        waypoint: [dx, dy] where:
            - dx: lateral offset (+ right, - left) in meters
            - dy: forward offset (+ forward, - back) in meters
        threshold_forward: minimum forward distance to trigger move (meters)
        threshold_turn: minimum angle to trigger turn (radians, ~17 degrees by default)

    Returns:
        Habitat discrete action: "move_forward", "turn_left", or "turn_right"
    """
    dx, dy = waypoint

    # Calculate angle to waypoint (in robot's local frame)
    # atan2(lateral, forward) gives angle from forward axis
    angle_to_waypoint = np.arctan2(dx, dy)

    # If we need to turn significantly, turn first
    if abs(angle_to_waypoint) > threshold_turn:
        return "turn_right" if angle_to_waypoint > 0 else "turn_left"

    # If waypoint is in front and close enough, move forward
    if dy > threshold_forward:
        return "move_forward"

    # If waypoint is behind us, turn around
    if dy < -threshold_forward:
        return "turn_right"  # Turn to face the waypoint

    # Default: move forward
    return "move_forward"


def main(args):
    """Main function for Habitat simulation with NoMaD policy"""

    # Initialize Habitat simulator
    print("Initializing Habitat simulator...")
    habitat_sim = HabitatSimulator(
        scene_path=args.scene_path, enable_physics=True, resolution=(480, 640)
    )
    habitat_sim.initialize()
    print("Habitat simulator initialized.")

    # Initialize NoMaD policy
    print("Initializing NoMaD policy...")
    policy = NoMaDPolicy(
        model_config_path=args.model_config,
        model_weights_path=args.model_weights,
        device=args.device,
    )
    print("NoMaD policy initialized.")

    # Recording containers
    video_frames = []
    trajectory_grid_points = []

    print(f"Starting exploration for {args.num_steps} steps...")

    # Exploration loop
    for step in range(args.num_steps):
        # Get current observation
        rgb_img = habitat_sim.get_observation()

        # Add to policy context
        policy.add_observation(rgb_img)

        # Get agent position and update trajectory
        position = habitat_sim.get_agent_position()
        grid_loc = habitat_sim.world_to_map(position)
        trajectory_grid_points.append(grid_loc)

        # Predict action after enough context
        action = None
        if step < args.context_warmup:
            # Random exploration during warmup
            action = np.random.choice(["move_forward", "turn_left", "turn_right"])
            print(f"Step {step}: Warmup - {action}")
        else:
            # Use policy to predict waypoint
            result = policy.predict_waypoint(
                waypoint_index=args.waypoint_index, num_samples=args.num_samples
            )

            if result is not None:
                waypoint, trajectory = result
                action = waypoint_to_habitat_action(waypoint)
                print(f"Step {step}: Predicted waypoint {waypoint} -> {action}")
            else:
                action = "move_forward"
                print(f"Step {step}: Default action - {action}")

        # Execute action
        habitat_sim.step(action)

        # Visualize if enabled
        if args.save_video:
            # Draw trajectory on map
            map_frame = habitat_sim.draw_trajectory(trajectory_grid_points)

            # Resize and combine frames
            h_rgb, w_rgb = rgb_img.shape[:2]
            h_map, w_map = map_frame.shape[:2]
            scale = h_rgb / h_map
            new_w_map = int(w_map * scale)
            map_frame_resized = cv2.resize(map_frame, (new_w_map, h_rgb))

            # Stack side-by-side
            combined_frame = np.hstack((rgb_img, map_frame_resized))
            video_frames.append(combined_frame)

    print("Exploration completed.")

    # Save video
    if args.save_video and len(video_frames) > 0:
        output_path = args.output_video
        print(f"Saving video to {output_path}...")
        imageio.mimsave(output_path, video_frames, fps=args.video_fps)
        print(f"Video saved to {output_path}")

    # Cleanup
    habitat_sim.close()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Habitat Simulation with NoMaD Visual Navigation")

    # Scene configuration
    parser.add_argument(
        "--scene-path", type=str, required=True, help="Path to Habitat scene file (.glb)"
    )

    # Model configuration
    parser.add_argument(
        "--model-config", type=str, required=True, help="Path to model config YAML file"
    )
    parser.add_argument(
        "--model-weights", type=str, required=True, help="Path to model checkpoint file (.pth)"
    )

    # Navigation parameters
    parser.add_argument(
        "--num-steps", type=int, default=1000, help="Number of simulation steps (default: 1000)"
    )
    parser.add_argument(
        "--waypoint-index",
        "-w",
        type=int,
        default=2,
        help="Index of waypoint for navigation (default: 2)",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=8,
        help="Number of action samples from diffusion model (default: 8)",
    )
    parser.add_argument(
        "--context-warmup",
        type=int,
        default=10,
        help="Number of random steps before using policy (default: 10)",
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run model on (default: cuda)",
    )

    # Video output
    parser.add_argument("--save-video", action="store_true", help="Save visualization video")
    parser.add_argument(
        "--output-video",
        type=str,
        default="habitat_exploration.mp4",
        help="Output video filename (default: habitat_exploration.mp4)",
    )
    parser.add_argument("--video-fps", type=int, default=10, help="Video FPS (default: 10)")

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.scene_path):
        raise FileNotFoundError(f"Scene file not found: {args.scene_path}")
    if not os.path.exists(args.model_config):
        raise FileNotFoundError(f"Model config not found: {args.model_config}")
    if not os.path.exists(args.model_weights):
        raise FileNotFoundError(f"Model weights not found: {args.model_weights}")

    main(args)
