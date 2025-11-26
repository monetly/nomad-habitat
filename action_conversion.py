#!/usr/bin/env python3
"""
Action conversion utilities for NoMaD waypoints to Habitat actions.

NoMaD outputs continuous waypoints [dx, dy] in meters relative to robot.
Habitat uses discrete actions: move_forward, turn_left, turn_right.

This module provides different strategies to bridge this gap.
"""

import numpy as np
from typing import Tuple, List


class WaypointToActionConverter:
    """
    Converts continuous waypoints to discrete Habitat actions.
    
    NoMaD waypoint convention:
    - dx: lateral offset (+ = right, - = left)
    - dy: forward offset (+ = forward, - = backward)
    - Units: meters
    """
    
    def __init__(
        self,
        turn_threshold: float = 0.3,  # radians (~17 degrees)
        forward_threshold: float = 0.1,  # meters
        turn_angle: float = 0.261799,  # Habitat default: 15 degrees
        forward_distance: float = 0.25  # Habitat default: 0.25 meters
    ):
        """
        Args:
            turn_threshold: Angle threshold to trigger turn action (radians)
            forward_threshold: Distance threshold to trigger forward action (meters)
            turn_angle: How much Habitat turns per turn action (radians)
            forward_distance: How much Habitat moves per forward action (meters)
        """
        self.turn_threshold = turn_threshold
        self.forward_threshold = forward_threshold
        self.turn_angle = turn_angle
        self.forward_distance = forward_distance
    
    def simple_convert(self, waypoint: np.ndarray) -> str:
        """
        Simple greedy conversion: either turn or move forward.
        
        Strategy:
        1. Calculate angle to waypoint
        2. If angle is large, turn first
        3. Otherwise, move forward
        
        Args:
            waypoint: [dx, dy] in meters
            
        Returns:
            Habitat action string
        """
        dx, dy = waypoint
        
        # Calculate angle to waypoint (in robot's frame)
        angle_to_waypoint = np.arctan2(dx, dy)  # atan2(lateral, forward)
        
        # If we need to turn significantly, turn first
        if abs(angle_to_waypoint) > self.turn_threshold:
            return "turn_right" if angle_to_waypoint > 0 else "turn_left"
        
        # If waypoint is mostly forward and close enough, move forward
        if dy > self.forward_threshold:
            return "move_forward"
        
        # If waypoint is behind us, turn around
        if dy < -self.forward_threshold:
            return "turn_right"  # or "turn_left", doesn't matter
        
        # Default: move forward
        return "move_forward"
    
    def sequential_convert(
        self, 
        waypoint: np.ndarray,
        current_heading: float = 0.0
    ) -> List[str]:
        """
        Sequential conversion: plan a sequence of actions.
        
        Strategy:
        1. Turn to face the waypoint
        2. Move forward to reach it
        
        Args:
            waypoint: [dx, dy] in meters
            current_heading: Current heading angle in radians (0 = north)
            
        Returns:
            List of Habitat actions to execute
        """
        dx, dy = waypoint
        
        # Calculate angle to waypoint
        angle_to_waypoint = np.arctan2(dx, dy)
        
        # Calculate number of turns needed
        num_turns = int(abs(angle_to_waypoint) / self.turn_angle)
        turn_action = "turn_right" if angle_to_waypoint > 0 else "turn_left"
        
        # Calculate number of forward steps needed
        distance_to_waypoint = np.sqrt(dx**2 + dy**2)
        num_forward = int(distance_to_waypoint / self.forward_distance)
        
        # Build action sequence
        actions = [turn_action] * num_turns + ["move_forward"] * num_forward
        
        # Ensure at least one action
        if len(actions) == 0:
            actions = ["move_forward"]
        
        return actions
    
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
            "forward_effort": forward_effort
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
        dt: float = 0.1      # control timestep (seconds)
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
    
    def velocity_to_habitat_actions(
        self, 
        v: float, 
        w: float,
        turn_threshold: float = 0.2
    ) -> str:
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
    print(f"Sequential method:    {seq_actions[:5]}..." if len(seq_actions) > 5 else f"Sequential method:    {seq_actions}")
    
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
        np.array([0.0, 1.0]),   # Straight ahead
        np.array([1.0, 1.0]),   # Forward-right diagonal
        np.array([-1.0, 1.0]),  # Forward-left diagonal
        np.array([2.0, 0.5]),   # Mostly right
        np.array([-2.0, 0.5]),  # Mostly left
        np.array([0.0, -1.0]),  # Behind
    ]
    
    for waypoint in test_waypoints:
        print()
        compare_conversion_methods(waypoint)
        print()
