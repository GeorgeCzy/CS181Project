import numpy as np
import random
import copy
from typing import Tuple, List, Optional, Dict, Any
import os
from base import Board, Player

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class RewardFunction:
    """斗兽棋智能奖励函数"""
    
    def __init__(self):
        # 奖励权重配置
        self.weights = {
            'win_game': 100.0,          # 获胜
            'lose_game': -100.0,        # 失败
            'draw_game': 0.0,           # 平局
            'capture_piece': 10.0,       # 吃掉对方棋子基础奖励
            'be_captured': -8.0,        # 被吃掉基础惩罚
            'mutual_destruction': -0.5,  # 同归于尽
            'reveal_piece': 1.0,        # 翻开棋子
            'survival_penalty': -0.1
        }
    
    def can_capture(self, attacker_strength: int, defender_strength: int) -> bool:
        """判断攻击方是否能吃掉防守方"""
        if attacker_strength == 8 and defender_strength == 1:  # 象吃鼠 x
            return False
        if attacker_strength > defender_strength:
            return True
        if attacker_strength == 1 and defender_strength == 8:  # 鼠吃象
            return True
        return False
    
    def get_piece_value(self, strength: int) -> float:
        """根据棋子强度返回价值权重"""
        # 象和鼠有特殊价值，中等强度棋子也很重要
        value_map = {1: 3.0, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5, 6: 3.0, 7: 3.5, 8: 4.0}
        return value_map.get(strength, 1.0)

    def find_closest_enemy(self, board: Board, pos: Tuple[int, int], player_id: int) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """
        找到距离指定位置最近的敌方棋子
        返回: (敌方位置, 距离) 或 (None, None)
        """
        current_piece = board.get_piece(pos[0], pos[1])
        if not current_piece:
            return None, None
            
        min_distance = float('inf')
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

    def evaluate_position(self, board: Board, pos: Tuple[int, int], player_id: int) -> Tuple[float, float]:
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
        if self.can_capture(enemy.strength, piece.strength):
            # 威胁随距离增加而减小
            threat = 4.0 / (distance + 1)
            # 对高价值棋子增加威胁程度
            if self.get_piece_value(piece.strength) >= 3.0:
                threat *= 1.5
            threat = -threat  # 转换为负值
        
        # 机会评估: 我方棋子能吃掉敌人
        if self.can_capture(piece.strength, enemy.strength):
            # 机会随距离增加而减小
            opportunity = 3.0 / (distance + 1)
            # 对高价值敌方棋子增加机会价值
            if self.get_piece_value(enemy.strength) >= 3.0:
                opportunity *= 1.5
        
        return threat, opportunity
    
    def _evaluate_revealed_piece(self, pos: Tuple[int, int], 
                            threats: Dict[Tuple[int, int], int],
                            opportunities: Dict[Tuple[int, int], int],
                            strength: int, is_self: bool) -> float:
        """
        评估翻开棋子的价值
        Args:
            threats: 预计算的威胁字典
            opportunities: 预计算的机会字典
        """
        value = 0.0
        
        # 根据敌我计算基础分
        base_value = self.weights['reveal'] if is_self else -self.weights['reveal']
        
        threat_value = threats.get(pos, 0) # 需要改，如果是敌人，则威胁的是附近的己方棋子
        opportunity_value = opportunities.get(pos, 0) # 同样，机会也是己方棋子的
        
        value = base_value + threat_value + opportunity_value
        
        return value * self.get_piece_value(strength)

    def estimate_reveal_reward(self, board: Board, pos: Tuple[int, int], player_id: int,
                            threats: Dict[Tuple[int, int], int],
                            opportunities: Dict[Tuple[int, int], int]) -> float:
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
            is_self = (player == player_id)
            reward = self._evaluate_revealed_piece(pos, threats, opportunities, strength, is_self)
            total_reward += reward * probability
            
        return total_reward
    
    def _is_defensive_move(self, board_after: Board, start_pos: Tuple[int, int], 
                        end_pos: Tuple[int, int], player_id: int,
                        threats_before: Dict[Tuple[int, int], int],
                        threats_after: Dict[Tuple[int, int], int]) -> bool:
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


    def calculate_reward(self, board_before: Board, board_after: Board, 
                        action: Tuple[str, Tuple[int, int], Optional[Tuple[int, int]]],
                        player_id: int, result: int) -> float:
        """计算奖励值"""
        action_type, pos1, pos2 = action
        total_reward = self.weights['survival_penalty']

        if result == player_id:
            return self.weights['win_game']
        if result == 1 - player_id:
            return self.weights['lose_game']
        if result == 2:
            return self.weights['draw_game']         
        
        # 翻开棋子的动作
        if action_type == "reveal":
            r, c = pos1
            piece = board_after.get_piece(r, c)
            if piece:
                # 使用合并后的评估函数
                threat, opportunity = self.evaluate_position(board_after, pos1, player_id)
                # 由于翻开后处于后手,威胁的影响更大
                total_reward = self.weights['reveal_piece'] + threat * 1.2 + opportunity
        
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
                    total_reward += self.weights['capture_piece'] * capture_value
                elif not piece_after:
                    # 同归于尽
                    total_reward += self.weights['mutual_destruction']
                else:
                    # 被吃
                    lost_value = self.get_piece_value(moving_piece_before.strength)
                    total_reward += self.weights['be_captured'] * lost_value
            
            # 评估移动后的位置价值变化
            if piece_after and piece_after.player == player_id:
                # 使用合并后的评估函数
                old_threat, old_opportunity = self.evaluate_position(board_before, start_pos, player_id)
                new_threat, new_opportunity = self.evaluate_position(board_after, end_pos, player_id)
                
                # 威胁减少或机会增加时给予奖励
                if new_threat > old_threat:  # threat是负值
                    total_reward += (new_threat - old_threat) * 1.2
                if new_opportunity > old_opportunity:
                    total_reward += (new_opportunity - old_opportunity) * 1.5
        
        return total_reward
    

class SimpleReward:
    """斗兽棋简单奖励函数"""
    
    def __init__(self):
        self.weights = {
            'win_game': 10.0,          # 获胜
            'lose_game': -10.0,        # 失败
            'draw_game': 0.0,           # 平局
            'capture_piece': 1.0,       # 吃掉对方棋子基础奖励
            'be_captured': -1.0,        # 被吃掉基础惩罚
            'reveal_piece': 0.5,         # 翻开棋子
            'survival_penalty': -0.1
        }
    
    def calculate_reward(self, board_before: Board, board_after: Board, 
                        action: Tuple[str, Tuple[int, int], Optional[Tuple[int, int]]],
                        player_id: int, result: int) -> float:
        """计算简单奖励值"""
        action_type, pos1, pos2 = action
        
        if result == player_id:
            return self.weights['win_game']
        if result == 1 - player_id:
            return self.weights['lose_game']
        if result == 2:
            return self.weights['draw_game']
        
        total_reward = 0.0
        
        if action_type == "reveal":
            total_reward += self.weights['reveal_piece']
        
        elif action_type == "move":
            start_pos, end_pos = pos1, pos2
            target_piece = board_after.get_piece(end_pos[0], end_pos[1])
            
            if target_piece and target_piece.player != player_id:
                total_reward += self.weights['capture_piece']
            elif not target_piece:
                total_reward += self.weights['be_captured']
        
        return total_reward

    
class Environment:
    """斗兽棋强化学习环境"""
    
    def __init__(self):
        self.board = None
        self.current_player = 0  # 0: RL agent, 1: opponent
        self.reward_function = RewardFunction()
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置环境，返回初始状态"""
        self.board = Board()
        self.current_player = 0
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        将棋盘状态转换为数值特征向量
        状态包括：
        - 棋盘上每个位置的棋子信息 (玩家, 强度, 是否翻开)
        - 当前玩家
        - 游戏阶段信息
        """
        state = np.zeros((7, 8, 4))  # ROWS x COLS x features
        
        for r in range(7):
            for c in range(8):
                piece = self.board.get_piece(r, c)
                if piece:
                    state[r, c, 0] = piece.player  # 玩家 (0 或 1)
                    state[r, c, 1] = piece.strength / 8.0  # 强度标准化
                    state[r, c, 2] = 1 if piece.revealed else 0  # 是否翻开
                    state[r, c, 3] = 1  # 有棋子
                else:
                    state[r, c, :] = 0  # 空位置
        
        # 展平状态向量并添加当前玩家信息
        flat_state = state.flatten()
        current_player_feature = np.array([self.current_player])
        
        return np.concatenate([flat_state, current_player_feature])
    
    def get_valid_actions(self, player_id: int) -> List[Tuple[str, Tuple[int, int], Optional[Tuple[int, int]]]]:
        """获取指定玩家的所有有效动作"""
        actions = []
        
        # 翻开棋子动作
        for r in range(7):
            for c in range(8):
                piece = self.board.get_piece(r, c)
                if piece and piece.player == player_id and not piece.revealed:
                    actions.append(("reveal", (r, c), None))
        
        # 移动棋子动作
        for r in range(7):
            for c in range(8):
                piece = self.board.get_piece(r, c)
                if piece and piece.player == player_id and piece.revealed:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 7 and 0 <= nc < 8:
                            # 使用临时棋盘测试移动是否有效
                            temp_board = copy.deepcopy(self.board)
                            if temp_board.try_move((r, c), (nr, nc)):
                                actions.append(("move", (r, c), (nr, nc)))
        
        return actions
    
    def _check_game_over(self): # return -1, 0, 1, 2 : None, 0 win, 1 win, draw
        """Checks for win/loss conditions and terminates the game if met."""
        red_pieces = self.board.get_player_pieces(0)
        blue_pieces = self.board.get_player_pieces(1)

        if not red_pieces:
            return 1  # Blue wins
        elif not blue_pieces:
            return 0  # Red wins
        elif len(red_pieces) == 1 and len(blue_pieces) == 1:
            rr, rc = red_pieces[0]
            br, bc = blue_pieces[0]
            red_piece = self.board.get_piece(rr, rc)
            blue_piece = self.board.get_piece(br, bc)

            # 只有当两个棋子都已翻开时，才能判断是否能互相捕获
            if red_piece.revealed and blue_piece.revealed:
                can_red_attack = (red_piece.strength > blue_piece.strength) or \
                                (red_piece.strength == 1 and blue_piece.strength == 8)
                can_blue_attack = (blue_piece.strength > red_piece.strength) or \
                                (blue_piece.strength == 1 and red_piece.strength == 8)

                # 如果它们相邻，且双方都无法捕获对方，则为和棋
                if self.board.is_adjacent((rr, rc), (br, bc)):
                    if not can_red_attack and not can_blue_attack:
                        return 2  # Draw
                else: # 如果不相邻，也无法捕获，则为和棋
                    if not can_red_attack and not can_blue_attack:
                        return 2  # Draw
            # 如果有一方或双方未翻开，游戏继续 (因为信息不完全，未来可能仍有变化)
        return -1  # Game continues
    
    def step(self, action: Tuple[str, Tuple[int, int], Optional[Tuple[int, int]]]) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作，使用智能奖励函数"""
        import copy
        
        action_type, pos1, pos2 = action
        board_before = copy.deepcopy(self.board)
        result = -1 # -1: game continues, 0: 0 wins, 1: 1 wins, 2: draw
        info = {}
        
        # 验证动作有效性
        valid_actions = self.get_valid_actions(self.current_player)
        if action not in valid_actions:
            # 无效动作
            reward = self.reward_function.weights['invalid_action']
            return self.get_state(), reward, False, {"invalid": True}
        
        # 执行动作
        success = False
        if action_type == "reveal":
            r, c = pos1
            piece = self.board.get_piece(r, c)
            if piece and piece.player == self.current_player and not piece.revealed:
                piece.revealed = True
                success = True
                
        elif action_type == "move":
            success = self.board.try_move(pos1, pos2)
        
        if not success:
            reward = self.reward_function.weights['invalid_action']
            return self.get_state(), reward, False, {"invalid": True}
        
        # # 检查游戏结束
        # red_pieces = self.board.get_player_pieces(0)
        # blue_pieces = self.board.get_player_pieces(1)
        
        # if not red_pieces or not blue_pieces:
        #     done = True
        # elif len(red_pieces) == 1 and len(blue_pieces) == 1:
        #     rr, rc = red_pieces[0]
        #     br, bc = blue_pieces[0]
        #     if not self.board.is_adjacent((rr, rc), (br, bc)):
        #         done = True

        result = self._check_game_over()
        
        # 使用智能奖励函数计算奖励
        reward = self.reward_function.calculate_reward(
            board_before, self.board, action, self.current_player, result
        )
        
        # 切换玩家
        self.current_player = 1 - self.current_player
        
        return self.get_state(), reward, result, info