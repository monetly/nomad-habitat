import habitat_sim
import numpy as np
import quaternion  # habitat-sim 依赖的 numpy-quaternion

class SimShortestPathFollower:
    """
    一个基于 habitat_sim 底层 PathFinder 的跟随器。
    用于验证：给定一个目标点，自动生成动作走到那里。
    """
    def __init__(self, sim, agent_id=0, goal_radius=0.15):
        self.sim = sim
        self.agent_id = agent_id
        self.goal_radius = goal_radius
        # 动作名称映射，需要与你的 HabitatSimulator 配置一致
        self.action_map = {
            "forward": "move_forward",
            "left": "turn_left",
            "right": "turn_right",
            "stop": None
        }

    def get_next_action(self, target_pos):
        """
        计算到达 target_pos 的下一步动作。
        """
        agent_state = self.sim.get_agent(self.agent_id).get_state()
        current_pos = agent_state.position

        # 1. 检查是否到达
        dist_to_target = np.linalg.norm(current_pos - target_pos)
        if dist_to_target < self.goal_radius:
            return self._compute_action_to_waypoint(agent_state, agent_state.position)

        # 2. 使用 PathFinder 规划路径
        path = habitat_sim.ShortestPath()
        path.requested_start = current_pos
        path.requested_end = target_pos
        
        found = self.sim.pathfinder.find_path(path)
        
        if not found or len(path.points) < 2:
            # 如果找不到路（比如点在障碍物内），尝试直线逼近或停止
            # 这里简单处理：如果找不到路但距离很近，尝试调整角度
            return self._compute_action_to_waypoint(agent_state, agent_state.position)

        # 3. 获取路径上的下一个导航点 (next waypoint)
        # path.points[0] 是当前位置，path.points[1] 是下一步要走的位置
        next_waypoint = path.points[1]
        
        # 4. 计算控制指令 (Geometric Controller)
        return self._compute_action_to_waypoint(agent_state, next_waypoint)

    def _compute_action_to_waypoint(self, agent_state, next_waypoint):
        """
        简单的 P-Controller：先转弯对准，再前进。
        """
        # 计算相对向量
        rel_pos = next_waypoint - agent_state.position
        
        # 将全局向量转换到 Agent 局部坐标系
        # Habitat 中 quaternion 是 (w, x, y, z)
        rot_inv = agent_state.rotation.inverse()
        local_rel_pos = quaternion.rotate_vectors(rot_inv, rel_pos)
        
        # local_rel_pos: -Z 是前方, X 是右方 (Habitat 标准)
        forward = -local_rel_pos[2]
        right = local_rel_pos[0]
        
        # 计算角度误差 (atan2(y, x))
        # 在这里，x轴是前方，y轴是右方（为了计算方便，我们映射一下）
        # 目标角度：相对于正前方的偏角
        angle_error = np.arctan2(right, forward)
        
        # 阈值设置
        TURN_THRESHOLD = np.deg2rad(10)  # 10度以内认为对准了
        
        if angle_error > TURN_THRESHOLD:
            return self.action_map["right"]
        elif angle_error < -TURN_THRESHOLD:
            return self.action_map["left"]
        else:
            return self.action_map["forward"]

    def local_to_global(self, local_waypoint):
        """
        工具函数：将模型预测的局部 Waypoint (dx, dy) 转为全局坐标
        假设 local_waypoint = [forward_dist, right_dist] 或 [x, y]
        注意：NoMaD 通常输出 [x, y]，其中 x 是前方，y 是左方/右方，需确认你的模型定义。
        这里假设：input [x_forward, y_left] (米)
        """
        agent_state = self.sim.get_agent(self.agent_id).get_state()
        
        # 构造局部向量 (Habitat: -Z Forward, X Right, Y Up)
        # 假设 local_waypoint[0] 是前方(x), local_waypoint[1] 是左方(y)
        # 则对应 Habitat 局部向量: -z = x_model, -x = y_model (左是负右)
        
        x_fwd = local_waypoint[0]
        y_left = local_waypoint[1]
        
        # Habitat 局部向量: [Right, Up, Back]
        # 前进 x_fwd -> Back = -x_fwd
        # 左移 y_left -> Right = -y_left
        local_vec = np.array([-y_left, 0.0, -x_fwd])
        
        # 旋转到全局
        global_vec = quaternion.rotate_vectors(agent_state.rotation, local_vec)
        target_pos = agent_state.position + global_vec
        
        # 投影到 NavMesh (确保点在可达区域)
        snapped_pos = self.sim.pathfinder.snap_point(target_pos)
        
        # 如果投影失败（比如点在墙里），使用未投影的点（PathFinder 会尝试处理）
        if np.isnan(snapped_pos[0]):
            return target_pos
        return snapped_pos