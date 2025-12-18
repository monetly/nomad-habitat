import habitat_sim
from habitat.utils.visualizations import maps
from src.controller.action_conversion import WaypointToActionConverter
import numpy as np
import imageio
import cv2

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

    def explore(self,policy,
                mode:str='continuous',
                CONTEXT_WARMUP=12,
                NUM_STEPS=600,
                NUM_SAMPLES=8):
        
        WAYPOINT_INDEX=2
        video_frames = []
        trajectory_grid_points = []
        obs_video = []
        print(f"Starting exploration for {NUM_STEPS} steps...")
        control = WaypointToActionConverter(sim=self)
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
            if step < CONTEXT_WARMUP:
                # Random exploration during warmup
                action=np.random.choice(["move_forward", "turn_left", "turn_right"])
                print(f"Step {step}: Warmup - {action}")
                self.step(action)
            else:
                result = policy.predict_waypoint(waypoint_index=WAYPOINT_INDEX, num_samples=NUM_SAMPLES)
                waypoint, _ = result
                waypoint/= np.linalg.norm(waypoint)*3
                control(waypoint)


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