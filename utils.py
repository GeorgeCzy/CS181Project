import numpy as np
import random
import copy
from typing import Tuple, List, Optional, Dict, Any
import os
from base import Board, Player
from collections import namedtuple
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class BaseReward:
    """奖励函数基类"""
    
    def get_piece_value(self, strength: int) -> float:
        """根据棋子强度返回价值权重"""
        # 象和鼠有特殊价值，中等强度棋子也很重要
        value_map = {1: 1.8, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5, 6: 3.0, 7: 3.5, 8: 4.0}
        return value_map.get(strength, 1.0)

    def calculate_reward(
        self,
        board_before: Board,
        board_after: Board,
        action: Tuple,
        player_id: int,
        done: bool,
    ) -> float:
        """计算奖励值"""
        raise NotImplementedError


class RewardFunction(BaseReward):
    """斗兽棋智能奖励函数"""

    def __init__(self):
        # 奖励权重配置
        self.weights = {
            "win_game": 100.0,  # 获胜
            "lose_game": -100.0,  # 失败
            "draw_game": 0.0,  # 平局
            "capture_piece": 10.0,  # 吃掉对方棋子基础奖励
            "be_captured": -8.0,  # 被吃掉基础惩罚
            "mutual_destruction": -0.5,  # 同归于尽
            "reveal_piece": 1.0,  # 翻开棋子
            "survival_penalty": -0.1,
        }

    # def can_capture(self, attacker_strength: int, defender_strength: int) -> bool:
    #     """判断攻击方是否能吃掉防守方"""
    #     if attacker_strength == 8 and defender_strength == 1:  # 象吃鼠 x
    #         return False
    #     if attacker_strength > defender_strength:
    #         return True
    #     if attacker_strength == 1 and defender_strength == 8:  # 鼠吃象
    #         return True
    #     return False

    def find_closest_enemy(
        self, board: Board, pos: Tuple[int, int], player_id: int
    ) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """
        找到距离指定位置最近的敌方棋子
        返回: (敌方位置, 距离) 或 (None, None)
        """
        current_piece = board.get_piece(pos[0], pos[1])
        if not current_piece:
            return None, None

        min_distance = float("inf")
        closest_enemy_pos = None

        # 遍历棋盘寻找已翻开的敌方棋子
        for r in range(7):
            for c in range(8):
                enemy = board.get_piece(r, c)
                if enemy and enemy.player != player_id and enemy.revealed:
                    dist = manhattan_distance(pos, (r, c))
                    # 更新最近敌人
                    if dist < min_distance:
                        min_distance = dist
                        closest_enemy_pos = (r, c)

        return closest_enemy_pos, min_distance if closest_enemy_pos else None

    def evaluate_position(
        self, board: Board, pos: Tuple[int, int], player_id: int
    ) -> Tuple[float, float]:
        """
        评估位置的威胁和机会程度
        Returns:
            Tuple[float, float]: (威胁值, 机会值)
        """
        piece = board.get_piece(pos[0], pos[1])
        if not piece or not piece.revealed:
            return 0.0, 0.0

        enemy_pos, distance = self.find_closest_enemy(board, pos, player_id)
        if not enemy_pos or not distance:
            return 0.0, 0.0

        enemy = board.get_piece(enemy_pos[0], enemy_pos[1])
        threat, opportunity = 0.0, 0.0

        # 威胁评估: 敌人能吃掉我方棋子
        if piece.compare_strength(enemy) == -1:
            # 威胁随距离增加而减小
            threat = 4.0 / (distance + 1)
            # 对高价值棋子增加威胁程度
            if self.get_piece_value(piece.strength) >= 3.0:
                threat *= 1.5
            threat = -threat  # 转换为负值

        # 机会评估: 我方棋子能吃掉敌人
        if piece.compare_strength(enemy) == 1:
            # 机会随距离增加而减小
            opportunity = 3.0 / (distance + 1)
            # 对高价值敌方棋子增加机会价值
            if self.get_piece_value(enemy.strength) >= 3.0:
                opportunity *= 1.5

        return threat, opportunity

    def _evaluate_revealed_piece(
        self,
        pos: Tuple[int, int],
        threats: Dict[Tuple[int, int], int],
        opportunities: Dict[Tuple[int, int], int],
        strength: int,
        is_self: bool,
    ) -> float:
        """
        评估翻开棋子的价值
        Args:
            threats: 预计算的威胁字典
            opportunities: 预计算的机会字典
        """
        value = 0.0

        # 根据敌我计算基础分
        base_value = self.weights["reveal"] if is_self else -self.weights["reveal"]

        threat_value = threats.get(
            pos, 0
        )  # 需要改，如果是敌人，则威胁的是附近的己方棋子
        opportunity_value = opportunities.get(pos, 0)  # 同样，机会也是己方棋子的

        value = base_value + threat_value + opportunity_value

        return value * self.get_piece_value(strength)

    def estimate_reveal_reward(
        self,
        board: Board,
        pos: Tuple[int, int],
        player_id: int,
        threats: Dict[Tuple[int, int], int],
        opportunities: Dict[Tuple[int, int], int],
    ) -> float:
        """估算翻开棋子的期望奖励"""
        # 获取所有未翻开棋子的可能性
        unrevealed_pieces = []
        for r in range(7):
            for c in range(8):
                piece = board.get_piece(r, c)
                if piece and not piece.revealed:
                    unrevealed_pieces.append((piece.player, piece.strength))

        if not unrevealed_pieces:
            return 0.0

        # 计算期望奖励
        total_reward = 0.0
        probability = 1.0 / len(unrevealed_pieces)

        for player, strength in unrevealed_pieces:
            is_self = player == player_id
            reward = self._evaluate_revealed_piece(
                pos, threats, opportunities, strength, is_self
            )
            total_reward += reward * probability

        return total_reward

    def _is_defensive_move(
        self,
        board_after: Board,
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
        player_id: int,
        threats_before: Dict[Tuple[int, int], int],
        threats_after: Dict[Tuple[int, int], int],
    ) -> bool:
        """
        判断是否为防守性移动
        Args:
            threats_before: 移动前的威胁字典
            threats_after: 移动后的威胁字典
        """
        piece = board_after.get_piece(end_pos[0], end_pos[1])
        if not piece or piece.player != player_id or not piece.revealed:
            return False

        # 检查移动是否减少了威胁
        old_threats = threats_before.get(start_pos, 0)
        new_threats = threats_after.get(end_pos, 0)

        # 高价值棋子(强度>=3)的威胁减少
        if new_threats < old_threats and self.get_piece_value(piece.strength) >= 3.0:
            return True

        return False

    def calculate_reward(
        self,
        board_before: Board,
        board_after: Board,
        action: Tuple[str, Tuple[int, int], Optional[Tuple[int, int]]],
        player_id: int,
        result: int,
    ) -> float:
        """计算奖励值"""
        action_type, pos1, pos2 = action
        total_reward = self.weights["survival_penalty"]

        if result == player_id:
            return self.weights["win_game"]
        if result == 1 - player_id:
            return self.weights["lose_game"]
        if result == 2:
            return self.weights["draw_game"]

        # 翻开棋子的动作
        if action_type == "reveal":
            r, c = pos1
            piece = board_after.get_piece(r, c)
            if piece:
                # 使用合并后的评估函数
                threat, opportunity = self.evaluate_position(
                    board_after, pos1, player_id
                )
                # 由于翻开后处于后手,威胁的影响更大
                total_reward = self.weights["reveal_piece"] + threat * 1.2 + opportunity

        # 移动棋子的动作
        else:  # move
            start_pos, end_pos = pos1, pos2
            moving_piece_before = board_before.get_piece(start_pos[0], start_pos[1])
            target_piece_before = board_before.get_piece(end_pos[0], end_pos[1])
            piece_after = board_after.get_piece(end_pos[0], end_pos[1])

            # 吃子奖励计算
            if target_piece_before and target_piece_before.player != player_id:
                if piece_after and piece_after.player == player_id:
                    # 成功吃子
                    capture_value = self.get_piece_value(target_piece_before.strength)
                    total_reward += self.weights["capture_piece"] * capture_value
                elif not piece_after:
                    # 同归于尽
                    total_reward += self.weights["mutual_destruction"]
                else:
                    # 被吃
                    lost_value = self.get_piece_value(moving_piece_before.strength)
                    total_reward += self.weights["be_captured"] * lost_value

            # 评估移动后的位置价值变化
            if piece_after and piece_after.player == player_id:
                # 使用合并后的评估函数
                old_threat, old_opportunity = self.evaluate_position(
                    board_before, start_pos, player_id
                )
                new_threat, new_opportunity = self.evaluate_position(
                    board_after, end_pos, player_id
                )

                # 威胁减少或机会增加时给予奖励
                if new_threat > old_threat:  # threat是负值
                    total_reward += (new_threat - old_threat) * 1.2
                if new_opportunity > old_opportunity:
                    total_reward += (new_opportunity - old_opportunity) * 1.5

        return total_reward


# class SimpleReward(BaseReward):
#     """斗兽棋简单奖励函数"""

#     def __init__(self):
#         self.weights = {
#             "win_game": 10.0,  # 获胜
#             "lose_game": -10.0,  # 失败
#             "draw_game": 0.0,  # 平局
#             "capture_piece": 1.0,  # 吃掉对方棋子基础奖励
#             "be_captured": -1.0,  # 被吃掉基础惩罚
#             "reveal_piece": 0.5,  # 翻开棋子
#             "survival_penalty": -0.1,
#         }

#     def calculate_reward(
#         self,
#         board_before: Board,
#         board_after: Board,
#         action: Tuple[str, Tuple[int, int], Optional[Tuple[int, int]]],
#         player_id: int,
#         result: int,
#     ) -> float:
#         """计算简单奖励值"""
#         action_type, pos1, pos2 = action

#         if result == player_id:
#             return self.weights["win_game"]
#         if result == 1 - player_id:
#             return self.weights["lose_game"]
#         if result == 2:
#             return self.weights["draw_game"]

#         total_reward = 0.0

#         if action_type == "reveal":
#             total_reward += self.weights["reveal_piece"]

#         elif action_type == "move":
#             start_pos, end_pos = pos1, pos2
#             target_piece = board_after.get_piece(end_pos[0], end_pos[1])

#             if target_piece and target_piece.player != player_id:
#                 total_reward += self.weights["capture_piece"]
#             elif not target_piece:
#                 total_reward += self.weights["be_captured"]

#         return total_reward

class SimpleReward:
    """重新设计的简单奖励函数"""

    def __init__(self):
        self.weights = {
            "win_game": 100.0,      # 获胜
            "lose_game": -100.0,    # 失败
            "draw_game": 0.0,       # 平局
            "capture_piece": 5.0,   # 吃掉对方棋子基础奖励
            "be_captured": -5.0,    # 被吃掉基础惩罚
            "mutual_destruction": -0.5, # 同归于尽：中性偏负
            "step_penalty": -0.02,  # 每步小惩罚，避免拖延
        }

    def count_immediate_threats_opportunities(self, board: Board, player_id: int) -> Tuple[float, float]:
        """计算当前棋盘上的即时威胁和机会总和"""
        total_threats = 0.0
        total_opportunities = 0.0
        
        my_pieces = board.get_player_pieces(player_id)
        
        for r, c in my_pieces:
            piece = board.get_piece(r, c)
            if piece and piece.revealed:
                # 检查四周的直接威胁和机会
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 7 and 0 <= nc < 8:
                        neighbor = board.get_piece(nr, nc)
                        if neighbor and neighbor.revealed and neighbor.player != player_id:
                            compare_result = piece.compare_strength(neighbor)
                            if compare_result == -1:  # 邻居能吃掉我
                                total_threats += self.get_piece_value(piece.strength)
                            elif compare_result == 1:  # 我能吃掉邻居
                                total_opportunities += self.get_piece_value(neighbor.strength)
        
        return total_threats, total_opportunities

    def calculate_reward(
        self,
        board_before: Board,
        board_after: Board,
        action: Tuple[str, Tuple[int, int], Optional[Tuple[int, int]]],
        player_id: int,
        result: int,
    ) -> float:
        """计算重新设计的奖励值"""
        try:
            # 处理动作解包
            if len(action) == 2:
                action_type, pos1 = action
                pos2 = None
            elif len(action) == 3:
                action_type, pos1, pos2 = action
            else:
                return -2.0

            # 游戏结束奖励（大奖励，稀疏）
            if result == player_id:
                return self.weights["win_game"]
            if result == 1 - player_id:
                return self.weights["lose_game"]
            if result == 2:
                return self.weights["draw_game"]

            # 基础步数惩罚，避免拖延
            total_reward = self.weights["step_penalty"]

            # 动作特定奖励
            if action_type == "reveal":
                # 翻开动作本身不给特殊奖励，让威胁机会变化来评估
                pass
                
            elif action_type == "move" and pos2 is not None:
                start_pos, end_pos = pos1, pos2
                moving_piece_before = board_before.get_piece(start_pos[0], start_pos[1])
                target_piece_before = board_before.get_piece(end_pos[0], end_pos[1])
                piece_after = board_after.get_piece(end_pos[0], end_pos[1])

                # 吃子奖励计算
                if target_piece_before and target_piece_before.player != player_id:
                    target_value = self.get_piece_value(target_piece_before.strength)
                    my_value = self.get_piece_value(moving_piece_before.strength)
                    
                    if piece_after and piece_after.player == player_id:
                        # 成功吃子
                        total_reward += self.weights["capture_piece"] * target_value
                        
                        # 额外奖励：用低价值棋子吃高价值棋子
                        if target_value > my_value:
                            total_reward += (target_value - my_value) * 0.5
                            
                    elif not piece_after:
                        # 同归于尽：精确的价值交换评估
                        value_diff = target_value - my_value
                        if value_diff > 0:
                            # 有利交换
                            total_reward += self.weights["mutual_destruction"] + value_diff * 1.0
                        elif value_diff == 0:
                            # 等价交换，中性偏负
                            total_reward += self.weights["mutual_destruction"]
                        else:
                            # 不利交换
                            total_reward += self.weights["mutual_destruction"] + value_diff * 1.0  # value_diff是负数
                    else:
                        # 被吃掉
                        total_reward += self.weights["be_captured"] * my_value

            # === 统一的威胁机会变化评估（适用于所有动作类型）===
            threats_before, opportunities_before = self.count_immediate_threats_opportunities(board_before, player_id)
            threats_after, opportunities_after = self.count_immediate_threats_opportunities(board_after, player_id)
            
            # 威胁减少是好事，机会增加是好事
            threat_improvement = threats_before - threats_after  # 威胁减少为正值
            opportunity_improvement = opportunities_after - opportunities_before  # 机会增加为正值
            
            # 综合战略改善奖励
            strategic_improvement = threat_improvement + opportunity_improvement
            total_reward += strategic_improvement * 0.4

            return total_reward
            
        except Exception as e:
            print(f"计算奖励时出错: {e}, 动作: {action}")
            raise e
        
class ImprovedReward(BaseReward):
    """改进的奖励函数 - 解决反复移动问题"""

    def __init__(self):
        self.weights = {
            "win_game": 50.0,        # 降低获胜奖励，避免过大梯度
            "lose_game": -50.0,      # 降低失败惩罚
            "draw_game": 0.0,
            "capture_piece": 3.0,    # 降低吃子奖励
            "be_captured": -3.0,     # 降低被吃惩罚
            "mutual_destruction": -0.5,
            "step_penalty": -0.1,    # 增加步数惩罚，强制快速结束
            "repetition_penalty": -1.0,  # 减少重复动作惩罚
            "progress_reward": 0.5,      # 减少进度奖励
            "position_improvement": 0.2, # 减少位置改善奖励
            "no_progress_penalty": -0.5, # 新增：无进展惩罚
        }
        
        # 记录最近的动作历史
        self.action_history = []
        self.position_history = []
        self.max_history = 10
        self.last_capture_step = 0  # 记录上次吃子的步数
        self.current_step = 0       # 当前步数

    def detect_repetition(self, action: Tuple) -> float:
        """检测重复动作并返回惩罚"""
        if len(action) < 3:
            return 0.0
            
        action_type, pos1, pos2 = action
        
        if action_type != "move" or pos2 is None:
            return 0.0
        
        # 检查是否在最近几步中有相同或相反的移动
        repetition_penalty = 0.0
        
        # 当前移动
        current_move = (pos1, pos2)
        reverse_move = (pos2, pos1)
        
        # 检查历史中的重复
        recent_actions = self.action_history[-4:]  # 只检查最近4步，减少惩罚
        
        for i, hist_action in enumerate(recent_actions):
            if len(hist_action) >= 3 and hist_action[0] == "move":
                hist_move = (hist_action[1], hist_action[2])
                
                # 完全相同的移动
                if hist_move == current_move:
                    repetition_penalty += self.weights["repetition_penalty"] * (1.0 - i * 0.2)
                
                # 相反的移动（往返移动）
                elif hist_move == reverse_move:
                    repetition_penalty += self.weights["repetition_penalty"] * 1.2 * (1.0 - i * 0.2)
        
        return repetition_penalty

    def calculate_position_value(self, board, pos: Tuple[int, int], player_id: int) -> float:
        """计算位置的战略价值"""
        r, c = pos
        value = 0.0
        
        # 简化中心位置价值计算
        center_distance = abs(r - 3) + abs(c - 3.5)
        center_value = max(0, 1 - center_distance * 0.1)  # 降低中心价值
        value += center_value
        
        # 检查周围的威胁和机会
        piece = board.get_piece(r, c)
        if piece and piece.player == player_id and piece.revealed:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 7 and 0 <= nc < 8:
                    neighbor = board.get_piece(nr, nc)
                    if neighbor and neighbor.revealed and neighbor.player != player_id:
                        # 能攻击敌人增加价值
                        if piece.compare_strength(neighbor) == 1:
                            value += 1.0  # 降低攻击价值
                        # 被威胁减少价值
                        elif piece.compare_strength(neighbor) == -1:
                            value -= 0.8  # 降低威胁惩罚
        
        return value

    def calculate_progress_reward(self, board_before, board_after, 
                                action: Tuple, player_id: int) -> float:
        """计算游戏进度奖励"""
        progress_reward = 0.0
        
        # 1. 翻开新棋子给予奖励
        if len(action) >= 2 and action[0] == "reveal":
            progress_reward += self.weights["progress_reward"]
        
        # 2. 移动到新位置给予奖励
        elif len(action) >= 3 and action[0] == "move":
            pos1, pos2 = action[1], action[2]
            
            # 简化位置历史检查
            if pos2 not in self.position_history[-10:]:  # 减少到最近10步
                progress_reward += self.weights["progress_reward"] * 0.3
            
            # 简化位置价值改善奖励
            old_value = self.calculate_position_value(board_before, pos1, player_id)
            new_value = self.calculate_position_value(board_after, pos2, player_id)
            
            if new_value > old_value:
                progress_reward += self.weights["position_improvement"] * (new_value - old_value)
        
        return progress_reward

    def update_history(self, action: Tuple):
        """更新动作和位置历史"""
        self.action_history.append(action)
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
        
        # 记录移动的目标位置
        if len(action) >= 3 and action[0] == "move":
            self.position_history.append(action[2])
            if len(self.position_history) > self.max_history * 2:
                self.position_history.pop(0)

    def calculate_reward(
        self,
        board_before,
        board_after,
        action: Tuple[str, Tuple[int, int], Optional[Tuple[int, int]]],
        player_id: int,
        result: int,
    ) -> float:
        """改进的奖励计算 - 重点解决步长过长问题"""
        try:
            self.current_step += 1
            
            # 处理动作解包
            if len(action) == 2:
                action_type, pos1 = action
                pos2 = None
                action = (action_type, pos1, pos2)
            elif len(action) == 3:
                action_type, pos1, pos2 = action
            else:
                return -2.0

            # 游戏结束奖励
            if result == player_id:
                # 奖励快速获胜
                step_bonus = max(0, (500 - self.current_step) * 0.02)
                return self.weights["win_game"] + step_bonus
            if result == 1 - player_id:
                return self.weights["lose_game"]
            if result == 2:
                return self.weights["draw_game"]

            # 基础步数惩罚（递增）
            step_penalty = self.weights["step_penalty"]
            if self.current_step > 200:
                step_penalty *= 2  # 200步后加倍惩罚
            if self.current_step > 400:
                step_penalty *= 3  # 400步后三倍惩罚
                
            total_reward = step_penalty

            # 无进展惩罚：如果很长时间没有吃子
            steps_since_capture = self.current_step - self.last_capture_step
            if steps_since_capture > 50:
                total_reward += self.weights["no_progress_penalty"] * (steps_since_capture / 50.0)

            # 检测重复动作惩罚
            repetition_penalty = self.detect_repetition(action)
            total_reward += repetition_penalty

            # 简化进度奖励
            progress_reward = self.calculate_progress_reward(board_before, board_after, action, player_id)
            total_reward += progress_reward

            # 处理移动动作的战斗奖励
            if action_type == "move" and pos2 is not None:
                start_pos, end_pos = pos1, pos2
                moving_piece_before = board_before.get_piece(start_pos[0], start_pos[1])
                target_piece_before = board_before.get_piece(end_pos[0], end_pos[1])
                piece_after = board_after.get_piece(end_pos[0], end_pos[1])

                # 战斗结果奖励
                if target_piece_before and target_piece_before.player != player_id:
                    target_value = self.get_piece_value(target_piece_before.strength)
                    my_value = self.get_piece_value(moving_piece_before.strength)
                    
                    # 记录吃子时间
                    self.last_capture_step = self.current_step
                    
                    if piece_after and piece_after.player == player_id:
                        # 成功吃子
                        total_reward += self.weights["capture_piece"] * target_value
                        if target_value > my_value:
                            total_reward += (target_value - my_value) * 0.3  # 降低额外奖励
                    elif not piece_after:
                        # 同归于尽
                        value_diff = target_value - my_value
                        total_reward += self.weights["mutual_destruction"] + value_diff * 0.5
                    else:
                        # 被吃掉
                        total_reward += self.weights["be_captured"] * my_value

            # 更新历史记录
            self.update_history(action)

            # 限制奖励范围，避免梯度爆炸
            total_reward = np.clip(total_reward, -10.0, 10.0)

            return total_reward
            
        except Exception as e:
            print(f"计算奖励时出错: {e}, 动作: {action}")
            return -2.0

    def reset_history(self):
        """重置历史记录（每局游戏开始时调用）"""
        self.action_history = []
        self.position_history = []
        self.last_capture_step = 0
        self.current_step = 0


class AggressiveReward(BaseReward):
    """激进的奖励函数 - 强制学习攻击性策略"""

    def __init__(self):
        self.weights = {
            "win_game": 100.0,
            "lose_game": -100.0,
            "draw_game": -10.0,        # 平局也是负奖励，鼓励主动进攻
            "capture_piece": 15.0,     # 大幅提高吃子奖励
            "be_captured": -8.0,
            "mutual_destruction": 2.0,  # 同归于尽变为正奖励（鼓励交换）
            "step_penalty": -0.2,      # 增加步数惩罚
            "repetition_penalty": -3.0, # 大幅增加重复惩罚
            "no_progress_penalty": -2.0, # 增加无进展惩罚
            "reveal_bonus": 2.0,       # 翻开奖励
            "attack_bonus": 5.0,       # 新增：主动攻击奖励
            "retreat_penalty": -2.0,   # 新增：后退惩罚
        }
        
        self.action_history = []
        self.position_history = []
        self.last_capture_step = 0
        self.current_step = 0
        self.board_center = (3, 4)  # 棋盘中心
        
    def is_attacking_move(self, board_before: Board, action: Tuple, player_id: int) -> bool:
        """判断是否为攻击性移动"""
        if len(action) < 3 or action[0] != "move":
            return False
            
        _, pos1, pos2 = action
        if pos2 is None:
            return False
            
        # 检查目标位置是否有敌方棋子
        target_piece = board_before.get_piece(pos2[0], pos2[1])
        if target_piece and target_piece.player != player_id:
            return True
            
        # 检查是否向敌方棋子靠近
        moving_piece = board_before.get_piece(pos1[0], pos1[1])
        if not moving_piece or not moving_piece.revealed:
            return False
            
        # 找最近的敌方棋子，看是否在靠近
        min_dist_before = float('inf')
        min_dist_after = float('inf')
        
        for r in range(7):
            for c in range(8):
                piece = board_before.get_piece(r, c)
                if piece and piece.player != player_id and piece.revealed:
                    dist_before = abs(r - pos1[0]) + abs(c - pos1[1])
                    dist_after = abs(r - pos2[0]) + abs(c - pos2[1])
                    min_dist_before = min(min_dist_before, dist_before)
                    min_dist_after = min(min_dist_after, dist_after)
        
        return min_dist_after < min_dist_before

    def is_retreating_move(self, board_before: Board, action: Tuple, player_id: int) -> bool:
        """判断是否为后退移动"""
        if len(action) < 3 or action[0] != "move":
            return False
            
        _, pos1, pos2 = action
        if pos2 is None:
            return False
            
        moving_piece = board_before.get_piece(pos1[0], pos1[1])
        if not moving_piece or not moving_piece.revealed:
            return False
            
        # 检查是否远离所有敌方棋子
        total_dist_before = 0
        total_dist_after = 0
        enemy_count = 0
        
        for r in range(7):
            for c in range(8):
                piece = board_before.get_piece(r, c)
                if piece and piece.player != player_id and piece.revealed:
                    dist_before = abs(r - pos1[0]) + abs(c - pos1[1])
                    dist_after = abs(r - pos2[0]) + abs(c - pos2[1])
                    total_dist_before += dist_before
                    total_dist_after += dist_after
                    enemy_count += 1
        
        if enemy_count == 0:
            return False
            
        avg_dist_before = total_dist_before / enemy_count
        avg_dist_after = total_dist_after / enemy_count
        
        return avg_dist_after > avg_dist_before + 0.5  # 明显远离

    def detect_repetition(self, action: Tuple) -> float:
        """更严厉的重复检测"""
        if len(action) < 3 or action[0] != "move":
            return 0.0
            
        _, pos1, pos2 = action
        if pos2 is None:
            return 0.0
        
        current_move = (pos1, pos2)
        reverse_move = (pos2, pos1)
        
        repetition_penalty = 0.0
        recent_actions = self.action_history[-8:]  # 检查最近8步
        
        for i, hist_action in enumerate(recent_actions):
            if len(hist_action) >= 3 and hist_action[0] == "move":
                hist_move = (hist_action[1], hist_action[2])
                
                if hist_move == current_move:
                    # 完全相同的移动，惩罚递增
                    repetition_penalty += self.weights["repetition_penalty"] * (2 ** (len(recent_actions) - i))
                elif hist_move == reverse_move:
                    # 往返移动，更严重的惩罚
                    repetition_penalty += self.weights["repetition_penalty"] * 2 * (2 ** (len(recent_actions) - i))
        
        return repetition_penalty

    def calculate_reward(self, board_before, board_after, action: Tuple, player_id: int, result: int) -> float:
        """激进的奖励计算"""
        try:
            self.current_step += 1
            
            # 处理动作解包
            if len(action) == 2:
                action_type, pos1 = action
                pos2 = None
                action = (action_type, pos1, pos2)
            elif len(action) == 3:
                action_type, pos1, pos2 = action
            else:
                return -5.0

            # 游戏结束奖励
            if result == player_id:
                step_bonus = max(0, (300 - self.current_step) * 0.1)  # 更大的快速获胜奖励
                return self.weights["win_game"] + step_bonus
            if result == 1 - player_id:
                return self.weights["lose_game"]
            if result == 2:
                return self.weights["draw_game"]

            # 基础步数惩罚（更严厉）
            step_penalty = self.weights["step_penalty"]
            if self.current_step > 150:
                step_penalty *= 3
            if self.current_step > 300:
                step_penalty *= 5
                
            total_reward = step_penalty

            # 无进展惩罚
            steps_since_capture = self.current_step - self.last_capture_step
            if steps_since_capture > 30:  # 降低到30步
                total_reward += self.weights["no_progress_penalty"] * (steps_since_capture / 30.0)

            # 重复动作惩罚
            repetition_penalty = self.detect_repetition(action)
            total_reward += repetition_penalty

            # 动作特定奖励
            if action_type == "reveal":
                total_reward += self.weights["reveal_bonus"]
                
            elif action_type == "move" and pos2 is not None:
                start_pos, end_pos = pos1, pos2
                moving_piece_before = board_before.get_piece(start_pos[0], start_pos[1])
                target_piece_before = board_before.get_piece(end_pos[0], end_pos[1])
                piece_after = board_after.get_piece(end_pos[0], end_pos[1])

                # 战斗奖励
                if target_piece_before and target_piece_before.player != player_id:
                    target_value = self.get_piece_value(target_piece_before.strength)
                    my_value = self.get_piece_value(moving_piece_before.strength)
                    
                    self.last_capture_step = self.current_step
                    
                    if piece_after and piece_after.player == player_id:
                        # 成功吃子 - 大奖励
                        total_reward += self.weights["capture_piece"] * target_value
                        total_reward += self.weights["attack_bonus"]  # 额外攻击奖励
                        
                        if target_value > my_value:
                            total_reward += (target_value - my_value) * 2.0  # 更大的以小博大奖励
                            
                    elif not piece_after:
                        # 同归于尽 - 现在是正奖励
                        total_reward += self.weights["mutual_destruction"] * min(target_value, my_value)
                    else:
                        # 被吃掉 - 仍然惩罚，但减少惩罚以鼓励尝试
                        total_reward += self.weights["be_captured"] * my_value * 0.5
                
                # 移动行为分析
                else:
                    # 检查是否为攻击性移动
                    if self.is_attacking_move(board_before, action, player_id):
                        total_reward += self.weights["attack_bonus"] * 0.5  # 较小的攻击奖励
                    
                    # 检查是否为后退移动
                    elif self.is_retreating_move(board_before, action, player_id):
                        total_reward += self.weights["retreat_penalty"]

            # 更新历史
            self.action_history.append(action)
            if len(self.action_history) > 10:
                self.action_history.pop(0)
            
            if action_type == "move" and pos2 is not None:
                self.position_history.append(pos2)
                if len(self.position_history) > 20:
                    self.position_history.pop(0)

            # 限制奖励范围
            return np.clip(total_reward, -20.0, 20.0)
            
        except Exception as e:
            print(f"计算奖励时出错: {e}, 动作: {action}")
            return -5.0

    def reset_history(self):
        """重置历史记录"""
        self.action_history = []
        self.position_history = []
        self.last_capture_step = 0
        self.current_step = 0


class FeatureExtractor:
    """改进的特征提取器"""
    
    def __init__(self):
        self.reward_function = RewardFunction()
    
    def extract_features(self, board: Board, player_id: int, action: Optional[Tuple] = None) -> np.ndarray:
        """提取状态-动作特征"""
        features = []
        
        # 1. 基础棋盘特征 (8维)
        features.extend(self._extract_board_features(board, player_id))
        
        # 2. 威胁和机会特征 (8维) 
        features.extend(self._extract_threat_opportunity_features(board, player_id))
        
        # 3. 位置控制特征 (4维)
        features.extend(self._extract_position_features(board, player_id))
        
        # 4. 动作特征 (6维)
        if action:
            features.extend(self._extract_action_features(board, player_id, action))
        else:
            features.extend([0.0] * 6)
            
        return np.array(features, dtype=np.float32)
    
    def _extract_board_features(self, board: Board, player_id: int) -> List[float]:
        """提取基础棋盘特征 (8维)"""
        features = []
        
        my_pieces = board.get_player_pieces(player_id)
        opponent_pieces = board.get_player_pieces(1 - player_id)
        
        # 棋子数量特征
        features.extend([
            len(my_pieces) / 16.0,  # 己方棋子数量比例
            len(opponent_pieces) / 16.0,  # 对方棋子数量比例
        ])
        
        # 翻开状态特征
        my_revealed = [r for r, c in my_pieces if board.get_piece(r, c).revealed]
        opp_revealed = [r for r, c in opponent_pieces if board.get_piece(r, c).revealed]
        
        features.extend([
            len(my_revealed) / max(len(my_pieces), 1),  # 己方已翻开比例
            len(opp_revealed) / max(len(opponent_pieces), 1),  # 对方已翻开比例
        ])
        
        # 价值特征
        my_total_value = sum(
            self.reward_function.get_piece_value(board.get_piece(r, c).strength)
            for r, c in my_pieces if board.get_piece(r, c).revealed
        )
        opp_total_value = sum(
            self.reward_function.get_piece_value(board.get_piece(r, c).strength) 
            for r, c in opponent_pieces if board.get_piece(r, c).revealed
        )
        
        total_value = my_total_value + opp_total_value + 1e-6
        features.extend([
            my_total_value / total_value,  # 己方价值比例
            opp_total_value / total_value,  # 对方价值比例
        ])
        
        # 强度分布特征
        my_high_value = sum(1 for r, c in my_pieces 
                           if board.get_piece(r, c).revealed and 
                           self.reward_function.get_piece_value(board.get_piece(r, c).strength) >= 3.0)
        opp_high_value = sum(1 for r, c in opponent_pieces
                            if board.get_piece(r, c).revealed and
                            self.reward_function.get_piece_value(board.get_piece(r, c).strength) >= 3.0)
        
        features.extend([
            my_high_value / max(len(my_revealed), 1),  # 己方高价值棋子比例
            opp_high_value / max(len(opp_revealed), 1),  # 对方高价值棋子比例  
        ])
        
        return features
    
    def _extract_threat_opportunity_features(self, board: Board, player_id: int) -> List[float]:
        """提取威胁和机会特征 (8维)"""
        features = []
        
        max_threat = 0
        max_opportunity = 0
        total_threat = 0
        total_opportunity = 0
        threatened_pieces = 0
        hunting_pieces = 0
        min_enemy_distance = float('inf')
        
        # 评估每个己方已翻开棋子
        my_revealed_pieces = [(r, c) for r, c in board.get_player_pieces(player_id)
                             if board.get_piece(r, c).revealed]
        
        for r, c in my_revealed_pieces:
            piece = board.get_piece(r, c)
            enemy_pos, distance = self.reward_function.find_closest_enemy(board, (r, c), player_id)
            
            if enemy_pos and distance:
                enemy = board.get_piece(enemy_pos[0], enemy_pos[1])
                min_enemy_distance = min(min_enemy_distance, distance)
                
                # 威胁评估
                compare_strength = piece.compare_strength(enemy)
                if compare_strength == -1:  # 敌人能吃掉我方棋子
                    threat = 4.0 / (distance + 1)
                    if self.reward_function.get_piece_value(piece.strength) >= 3.0:
                        threat *= 1.5
                    total_threat += threat
                    max_threat = max(max_threat, threat)
                    threatened_pieces += 1
                
                # 机会评估  
                if compare_strength == 1:  # 我方棋子能吃掉敌人
                    opportunity = 3.0 / (distance + 1)
                    if self.reward_function.get_piece_value(enemy.strength) >= 3.0:
                        opportunity *= 1.5
                    total_opportunity += opportunity
                    max_opportunity = max(max_opportunity, opportunity)
                    hunting_pieces += 1
        
        piece_count = len(my_revealed_pieces)
        features.extend([
            max_threat,  # 最大威胁
            max_opportunity,  # 最大机会
            total_threat / max(piece_count, 1),  # 平均威胁
            total_opportunity / max(piece_count, 1),  # 平均机会
            1.0 / (min_enemy_distance + 1) if min_enemy_distance != float('inf') else 0,  # 最近敌人
            threatened_pieces / max(piece_count, 1),  # 受威胁棋子比例
            hunting_pieces / max(piece_count, 1),  # 有机会棋子比例
            (total_opportunity - total_threat),  # 净机会值
        ])
        
        return features
    
    def _extract_position_features(self, board: Board, player_id: int) -> List[float]:
        """重新设计位置控制特征 (4维)"""
        features = []
        
        my_pieces = board.get_player_pieces(player_id)
        
        # 1. 中心控制程度
        center_positions = [(2, 3), (2, 4), (3, 3), (3, 4), (4, 3), (4, 4)]
        my_center_control = sum(1 for r, c in my_pieces if (r, c) in center_positions)
        
        # 2. 棋子分散度（避免扎堆）
        if len(my_pieces) > 1:
            total_distance = 0
            count = 0
            for i, (r1, c1) in enumerate(my_pieces):
                for r2, c2 in my_pieces[i+1:]:
                    total_distance += manhattan_distance((r1, c1), (r2, c2))
                    count += 1
            avg_distance = total_distance / count if count > 0 else 0
            dispersion = avg_distance / 12.0  # 归一化
        else:
            dispersion = 0
        
        # 3. 区域平衡（在上下两个区域的分布）
        upper_pieces = sum(1 for r, c in my_pieces if r <= 2)  # 上半区
        lower_pieces = sum(1 for r, c in my_pieces if r >= 4)  # 下半区
        middle_pieces = len(my_pieces) - upper_pieces - lower_pieces
        
        # 计算分布平衡度
        total_pieces = len(my_pieces)
        if total_pieces > 0:
            balance = 1.0 - abs(upper_pieces - lower_pieces) / total_pieces
        else:
            balance = 0.0
        
        # 4. 边缘风险（太多棋子在边缘容易被围攻）
        edge_positions = [(r, c) for r in [0, 6] for c in range(8)] + \
                        [(r, c) for r in range(1, 6) for c in [0, 7]]
        edge_pieces = sum(1 for r, c in my_pieces if (r, c) in edge_positions)
        edge_risk = edge_pieces / max(len(my_pieces), 1)
        
        features.extend([
            my_center_control / 6.0,   # 中心控制比例 [0,1]
            dispersion,                # 棋子分散度 [0,1]
            balance,                   # 区域平衡度 [0,1]
            1.0 - edge_risk           # 边缘安全度 [0,1]
        ])
        
        return features
    
    def _extract_action_features(self, board: Board, player_id: int, action: Tuple) -> List[float]:
        """提取动作特征 (6维)"""
        features = []
        action_type, pos1, pos2 = action
        
        if action_type == "reveal":
            r, c = pos1
            # 翻开位置的战略价值
            center_positions = [(2, 3), (2, 4), (3, 3), (3, 4), (4, 3), (4, 4)]
            is_center = 1.0 if (r, c) in center_positions else 0.0
            
            # 附近敌人数量
            nearby_enemies = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 7 and 0 <= nc < 8:
                        piece = board.get_piece(nr, nc)
                        if piece and piece.player != player_id and piece.revealed:
                            nearby_enemies += 1
            
            features.extend([
                1.0, 0.0,  # 动作类型：翻开
                is_center,  # 是否中心位置
                nearby_enemies / 8.0,  # 附近敌人密度
                r / 6.0, c / 7.0  # 位置归一化
            ])
            
        else:  # move
            sr, sc = pos1
            er, ec = pos2
            
            # 移动方向特征
            dr, dc = er - sr, ec - sc
            move_direction = 0.0
            if dr == -1: move_direction = 0.25  # 上
            elif dr == 1: move_direction = 0.5   # 下  
            elif dc == -1: move_direction = 0.75 # 左
            elif dc == 1: move_direction = 1.0   # 右
            
            # 目标位置价值
            target_piece = board.get_piece(er, ec)
            target_value = 0.0
            if target_piece and target_piece.player != player_id:
                target_value = self.reward_function.get_piece_value(target_piece.strength) / 4.0
            
            features.extend([
                0.0, 1.0,  # 动作类型：移动
                move_direction,  # 移动方向
                target_value,  # 目标价值
                abs(dr) + abs(dc),  # 移动距离（总是1，但保持一致性）
                (er + ec) / 13.0  # 目标位置综合坐标
            ])
        
        return features


def save_model_data(data: Dict, filename: str, save_path: str = "model_data/"):
    """统一的模型保存函数"""
    import os
    import pickle

    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)

    try:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"模型已保存到 {filepath}")
    except Exception as e:
        print(f"保存模型失败: {e}")


def load_model_data(filename: str, save_path: str = "model_data/") -> Optional[Dict]:
    """统一的模型加载函数"""
    import os
    import pickle

    filepath = os.path.join(save_path, filename)

    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"模型已从 {filepath} 加载")
        return data
    except FileNotFoundError:
        print(f"模型文件 {filepath} 不存在")
        return None
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None


# 经验回放缓冲区
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'result'])

class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha      # 优先级指数
        self.beta = beta        # 重要性采样指数
        self.beta_increment = 0.001
        self.buffer = []
        self.priorities = []
        self.pos = 0
    
    def push(self, state, action, reward, next_state, result):
        """添加新经验"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(state, action, reward, next_state, result))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = Experience(state, action, reward, next_state, result)
            self.priorities[self.pos] = max_priority
        
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """采样经验"""
        if len(self.buffer) == 0:
            return []
            
        # 计算采样概率
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 计算重要性权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(device)
        
        # 获取经验
        experiences = [self.buffer[idx] for idx in indices]
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6  # 添加小值防止优先级为0
    
    def __len__(self):
        return len(self.buffer)