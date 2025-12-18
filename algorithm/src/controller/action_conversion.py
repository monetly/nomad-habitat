#!/usr/bin/env python3
"""
Action conversion utilities for NoMaD waypoints to Habitat actions.

NoMaD outputs continuous waypoints [dx, dy] in meters relative to robot.
Habitat uses discrete actions: move_forward, turn_left, turn_right.

This module provides different strategies to bridge this gap.
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, List
import habitat_sim
import quaternion as qt

ROBOT_CONFIG = "src/config/robot.yaml"
DISCRETE_CONFIG = "src/config/discrete.yaml"

with Path(ROBOT_CONFIG).open("r", encoding="utf-8") as f:
    robot_config = yaml.safe_load(f)

with Path(DISCRETE_CONFIG).open("r", encoding="utf-8") as f:
    discrete_config = yaml.safe_load(f)


def arc_approximation_circle(waypoint: np.ndarray, step: int = 4) -> np.ndarray:
    """
    用圆弧（圆心在原点，半径常量）按角度均分逼近终点 (dx, dy)。
    参数:
        waypoint: np.ndarray([dx, dy])
        step: 分段数
    返回:
        (step+1, 2) 的点序列（包含起点和终点）
    """
    dx, dy = float(waypoint[0]), float(waypoint[1])
    r = np.hypot(dx, dy)                 # 位移模长，也是该圆的半径（圆心在原点的前提下）
    theta = np.arctan2(dy, dx)           # 终点的极角
    angles = np.linspace(0.0, theta, step + 1)
    points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    return points


class WaypointToActionConverter:
    """
    Converts continuous waypoints to discrete Habitat actions.

    NoMaD waypoint convention:
    - dy: lateral offset (+ = left, - = right)
    - dx: forward offset (+ = forward, - = backward)
    - Units: meters
    """

    def __init__(
        self,
        turn_threshold: float = discrete_config["turn_threshold"],  # radians (~17 degrees)
        forward_threshold: float = discrete_config["forward_threshold"],  # meters
        turn_angle: float = discrete_config["turn_angle"],  # Habitat default: 10 degrees
        forward_distance: float = discrete_config[
            "forward_distance"
        ],  # Habitat default: 0.25 meters
        mode: str = "continuous",
        sim: habitat_sim.sim=None
    ):
        """
        Args:
            turn_threshold: Angle threshold to trigger turn action (radians)
            forward_threshold: Distance threshold to trigger forward action (meters)
            turn_angle: How much Habitat turns per turn action (radians)
            forward_distance: How much Habitat moves per forward action (meters)
        """
        # self.turn_threshold = turn_threshold
        self.turn_threshold = np.deg2rad(turn_angle)
        self.forward_threshold = forward_threshold
        # self.turn_angle = turn_angle
        self.turn_angle = np.deg2rad(turn_angle)
        self.forward_distance = forward_distance
        self.rate = robot_config["frame_rate"]
        self.max_angle_per_step = robot_config["max_w"] / self.rate
        self.max_distance_per_step = robot_config["max_v"] / self.rate
        self.mode = mode
        self.sim=sim


    def sequential_convert(self
    , waypoint: np.ndarray, current_heading: float = 0.0) -> List[str]:
        """
        Sequential conversion: plan a sequence of discrete actions.

        Strategy:
        1. Turn to face the waypoint (discretized)
        2. Move forward to reach it (discretized)

        Args:
            waypoint: [dx, dy] in meters
            current_heading: Current heading angle in radians (0 = north)

        Returns:
            List of Habitat actions to execute
        """
        dx, dy = waypoint

        # 计算目标方向角（相对于机器人坐标系）
        angle_to_waypoint = np.arctan2(dy, dx)
        # 计算需要转动的角度差
        angle_diff = angle_to_waypoint - current_heading
        # 计算需要的离散转向次数
        num_turns = int(min(self.max_angle_per_step, abs(angle_diff)) / self.turn_angle)

        if abs(angle_diff) <= self.turn_threshold:
            return ["move_forward"]

        turn_action = "turn_right" if angle_diff < 0 else "turn_left"

        # 
        actions = [turn_action] * num_turns + ["move_forward"]
        if self.sim is not None:
            for action in actions:
                self.sim.step(action)
        return actions

    #     def sequential_convert(
    #     self,
    #     waypoint: np.ndarray,
    #     current_heading: float = 0.0 # 假设 waypoint 是局部坐标，这里通常保持 0
    # ) -> List[str]:
    #         dx, dy = waypoint

    #         # 1. 计算距离和需要的移动步数 (假设每步移动 0.25m)
    #         dist = np.linalg.norm([dx, dy])
    #         MOVE_STEP_SIZE = 0.25
    #         num_moves = int(np.ceil(dist / MOVE_STEP_SIZE))

    #         # 2. 计算角度
    #         angle_to_waypoint = np.arctan2(dy, dx)
    #         angle_diff = angle_to_waypoint - current_heading

    #         # 3. 角度归一化到 [-pi, pi] (关键修复)
    #         angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

    #         # 4. 计算转向次数
    #         # 注意：这里逻辑从"限制最大角度"改为"计算实际需要的次数"
    #         # 假设 self.turn_angle 是每次动作转的角度 (例如 10度 或 30度)
    #         turn_angle_rad = np.radians(self.turn_angle) if self.turn_angle > 1 else self.turn_angle
    #         num_turns = int(abs(angle_diff) / turn_angle_rad)

    #         turn_action = "turn_right" if angle_diff < 0 else "turn_left"

    #         # 5. 构建序列
    #         # 如果角度偏差太大，先不移动，只转向；或者转向后移动
    #         actions = [turn_action] * num_turns + ["move_forward"] * num_moves

    #         # 简单的防抖动：如果就在脚下，不要转圈
    #         if dist < 0.1:
    #             return []

    #         return actions
    def __call__(self, 
                 waypoint: np.array,
                 step :int = 4):
        if self.mode == 'discrete':
            self.sequential_convert(waypoint)
        elif self.mode == 'continuous':
            dx,dy = waypoint

            state=self.sim.agent.get_state()
            rotation = state.rotation
            quat_target = qt.from_rotation_vector(np.array([0, np.arctan2(dy,dx), 0]))
            rotation*= quat_target
            state.rotation = rotation
            state.position+=np.array([dy ,0 ,-dx])
            self.sim.agent.set_state(state)

        return

            


    def proportional_convert(self, waypoint: np.ndarray) -> Tuple[str, dict]:
        """
        Proportional conversion with metadata.

        Strategy:
        - Use proportional control logic
        - Return action + debug info

        Args:
            waypoint: [dx, dy] in meters

        Returns:
            (action, metadata_dict)
        """
        dx, dy = waypoint

        # Calculate angle and distance
        angle_to_waypoint = np.arctan2(dx, dy)
        distance_to_waypoint = np.sqrt(dx**2 + dy**2)

        # Proportional gains (tune these)
        k_angle = 2.0  # Angular gain
        k_distance = 1.0  # Distance gain

        # Calculate "control effort"
        turn_effort = k_angle * abs(angle_to_waypoint)
        forward_effort = k_distance * distance_to_waypoint if dy > 0 else 0

        metadata = {
            "angle_to_waypoint": angle_to_waypoint,
            "distance_to_waypoint": distance_to_waypoint,
            "turn_effort": turn_effort,
            "forward_effort": forward_effort,
        }

        # Decision logic
        if turn_effort > forward_effort and abs(angle_to_waypoint) > self.turn_threshold:
            action = "turn_right" if angle_to_waypoint > 0 else "turn_left"
        elif dy > self.forward_threshold:
            action = "move_forward"
        elif dy < -self.forward_threshold:
            # Waypoint is behind, turn around
            action = "turn_right"
        else:
            action = "move_forward"

        return action, metadata


class VelocityController:
    """
    Alternative approach: Convert waypoints to velocity commands,
    then simulate those velocities in Habitat.

    This is more accurate but requires custom Habitat integration.
    """

    def __init__(
        self,
        max_v: float = 0.5,  # max linear velocity (m/s)
        max_w: float = 1.0,  # max angular velocity (rad/s)
        dt: float = 0.1,  # control timestep (seconds)
    ):
        self.max_v = max_v
        self.max_w = max_w
        self.dt = dt

    def waypoint_to_velocity(self, waypoint: np.ndarray) -> Tuple[float, float]:
        """
        Convert waypoint to (v, w) velocity commands.

        Uses a simple proportional controller.

        Args:
            waypoint: [dx, dy] in meters

        Returns:
            (v, w): linear velocity (m/s), angular velocity (rad/s)
        """
        dx, dy = waypoint

        # Calculate angle to waypoint
        angle_to_waypoint = np.arctan2(dx, dy)
        distance_to_waypoint = np.sqrt(dx**2 + dy**2)

        # Proportional control gains
        k_v = 1.0  # Linear velocity gain
        k_w = 2.0  # Angular velocity gain

        # Calculate velocities
        v = k_v * distance_to_waypoint * np.cos(angle_to_waypoint)
        w = k_w * angle_to_waypoint

        # Clip to max velocities
        v = np.clip(v, -self.max_v, self.max_v)
        w = np.clip(w, -self.max_w, self.max_w)

        return v, w

    def velocity_to_habitat_actions(self, v: float, w: float, turn_threshold: float = 0.2) -> str:
        """
        Convert velocity commands to Habitat discrete actions.

        Args:
            v: linear velocity (m/s)
            w: angular velocity (rad/s)
            turn_threshold: threshold for turning (rad/s)

        Returns:
            Habitat action string
        """
        # If angular velocity is dominant, turn
        if abs(w) > turn_threshold:
            return "turn_right" if w > 0 else "turn_left"

        # Otherwise, move forward if v > 0
        if v > 0.05:
            return "move_forward"

        # Default
        return "move_forward"


def compare_conversion_methods(waypoint: np.ndarray):
    """Utility to compare different conversion methods."""
    print(f"Waypoint: dx={waypoint[0]:.2f}, dy={waypoint[1]:.2f}")
    print("-" * 50)

    # Simple converter
    simple_conv = WaypointToActionConverter()
    simple_action = simple_conv.simple_convert(waypoint)
    print(f"Simple method:        {simple_action}")

    # Sequential converter
    seq_actions = simple_conv.sequential_convert(waypoint)
    print(
        f"Sequential method:    {seq_actions[:5]}..."
        if len(seq_actions) > 5
        else f"Sequential method:    {seq_actions}"
    )

    # Proportional converter
    prop_action, metadata = simple_conv.proportional_convert(waypoint)
    print(f"Proportional method:  {prop_action}")
    print(f"  - Angle: {np.degrees(metadata['angle_to_waypoint']):.1f}°")
    print(f"  - Distance: {metadata['distance_to_waypoint']:.2f}m")

    # Velocity controller
    vel_ctrl = VelocityController()
    v, w = vel_ctrl.waypoint_to_velocity(waypoint)
    vel_action = vel_ctrl.velocity_to_habitat_actions(v, w)
    print(f"Velocity method:      {vel_action}")
    print(f"  - v: {v:.2f} m/s, w: {w:.2f} rad/s")


if __name__ == "__main__":
    print("Testing Action Conversion Methods")
    print("=" * 50)

    # Test cases
    test_waypoints = [
        np.array([0.0, 1.0]),  # Straight ahead
        np.array([1.0, 1.0]),  # Forward-right diagonal
        np.array([-1.0, 1.0]),  # Forward-left diagonal
        np.array([2.0, 0.5]),  # Mostly right
        np.array([-2.0, 0.5]),  # Mostly left
        np.array([0.0, -1.0]),  # Behind
    ]

    for waypoint in test_waypoints:
        print()
        compare_conversion_methods(waypoint)
        print()
