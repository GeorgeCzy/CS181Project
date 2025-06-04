import numpy as np
import random
import math
import copy
from typing import List, Tuple, Dict, Optional
from base import Board, Player
from utils import RewardFunction
# import AgentFight as AG

class MCTSNode:
    """蒙特卡洛树搜索节点"""
    def __init__(self, state: Board, parent=None, action=None, player_id: int = 0):
        self.state = state
        self.parent = parent
        self.action = action  # 到达该节点的动作
        self.player_id = player_id
        self.children: List[MCTSNode] = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = self._get_valid_actions()
        
    def _get_valid_actions(self) -> List[Tuple]:
        """获取所有有效动作"""
        actions = []
        
        # 翻开动作
        for r in range(7):
            for c in range(8):
                piece = self.state.get_piece(r, c)
                if piece and piece.player == self.player_id and not piece.revealed:
                    actions.append(("reveal", (r, c), None))
        
        # 移动动作
        for r in range(7):
            for c in range(8):
                piece = self.state.get_piece(r, c)
                if piece and piece.player == self.player_id and piece.revealed:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 7 and 0 <= nc < 8:
                            temp_board = copy.deepcopy(self.state)
                            if temp_board.try_move((r, c), (nr, nc)):
                                actions.append(("move", (r, c), (nr, nc)))
                                
        return actions
    
    def ucb_value(self, exploration_constant: float = 1.414) -> float:
        """计算UCB值"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

class MCTSAgent(Player):
    """基于MCTS的智能体"""
    
    def __init__(self, player_id: int, simulation_limit: int = 100, 
                 exploration_constant: float = 1.414):
        super().__init__(player_id)
        self.simulation_limit = simulation_limit
        self.exploration_constant = exploration_constant
        self.reward_function = RewardFunction()
        self.ai_type = f"MCTS (s={simulation_limit})"
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """选择阶段: 使用UCB选择最优子节点"""
        while not self._is_terminal(node) and not node.untried_actions:
            if not node.children:
                return node
                
            # 选择UCB值最高的子节点
            node = max(node.children, key=lambda n: n.ucb_value(self.exploration_constant))
        return node
    
    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """扩展阶段: 创建新的子节点"""
        if not node.untried_actions:
            return None
            
        action = node.untried_actions.pop()
        new_state = copy.deepcopy(node.state)
        
        # 执行动作
        success = False
        if action[0] == "reveal":
            r, c = action[1]
            piece = new_state.get_piece(r, c)
            if piece and piece.player == node.player_id and not piece.revealed:
                piece.revealed = True
                success = True
        else:  # move
            success = new_state.try_move(action[1], action[2])
            
        if not success:
            return None
            
        # 创建新节点
        new_node = MCTSNode(
            state=new_state,
            parent=node,
            action=action,
            player_id=1 - node.player_id  # 切换玩家
        )
        node.children.append(new_node)
        return new_node
    
    def _simulate(self, node: MCTSNode) -> int:
        """模拟阶段: 使用启发式随机策略模拟到游戏结束"""
        state = copy.deepcopy(node.state)
        current_player = node.player_id
        last_move = None  # 记录上一步移动
        repeated_moves = 0  # 记录重复移动次数
        
        while True:
            result = self._check_game_result(state)
            if result != -1:  # 游戏结束
                if result == self.player_id:
                    return 1  # 胜利
                elif result == 1 - self.player_id:
                    return 0  # 失败
                else:
                    return 0.5  # 平局
            
            # 获取所有可能的动作
            valid_actions = []
            reveal_actions = []
            move_actions = []
            
            for r in range(7):
                for c in range(8):
                    piece = state.get_piece(r, c)
                    if piece and piece.player == current_player:
                        if not piece.revealed:
                            reveal_actions.append(("reveal", (r, c), None))
                        else:
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < 7 and 0 <= nc < 8:
                                    temp_board = copy.deepcopy(state)
                                    if temp_board.try_move((r, c), (nr, nc)):
                                        move_actions.append(("move", (r, c), (nr, nc)))
            
            # 优先考虑翻开动作
            if reveal_actions:
                valid_actions = reveal_actions
            else:
                valid_actions = move_actions
            
            if not valid_actions:
                return 0.5  # 无法行动，视为平局
            
            # 使用启发式策略选择动作
            action = None
            if random.random() < 0.7:  # 70%概率使用启发式策略
                best_value = float('-inf')
                for a in valid_actions:
                    value = 0
                    
                    # 避免重复移动
                    if last_move and a[0] == "move":
                        if (a[1] == last_move[2] and a[2] == last_move[1]) or \
                        (a[1] == last_move[1] and a[2] == last_move[2]):
                            value -= 2.0  # 重复移动的惩罚
                    
                    # 评估动作价值
                    if a[0] == "reveal":
                        value += 1.0  # 鼓励翻开棋子
                    else:  # move
                        start_pos, end_pos = a[1], a[2]
                        moving_piece = state.get_piece(start_pos[0], start_pos[1])
                        target_pos = state.get_piece(end_pos[0], end_pos[1])
                        
                        if target_pos and target_pos.player != current_player:
                            # 评估吃子价值
                            if target_pos.revealed:
                                if self.reward_function.can_capture(moving_piece.strength, target_pos.strength):
                                    value += 3.0 * self.reward_function.get_piece_value(target_pos.strength)
                        
                        # 考虑威胁和机会
                        threat, opportunity = self.reward_function.evaluate_position(state, end_pos, current_player)
                        value += opportunity - abs(threat)
                    
                    if value > best_value:
                        best_value = value
                        action = a
            
            # 如果启发式策略没有找到好的动作，随机选择
            if not action:
                action = random.choice(valid_actions)
            
            # 执行动作
            if action[0] == "reveal":
                r, c = action[1]
                piece = state.get_piece(r, c)
                piece.revealed = True
                last_move = None
                repeated_moves = 0
            else:
                success = state.try_move(action[1], action[2])
                if last_move and ((action[1] == last_move[2] and action[2] == last_move[1]) or \
                                (action[1] == last_move[1] and action[2] == last_move[2])):
                    repeated_moves += 1
                    if repeated_moves >= 3:  # 连续重复3次就平局
                        return 0.5
                else:
                    repeated_moves = 0
                last_move = action
            
            current_player = 1 - current_player
    
    def _backpropagate(self, node: MCTSNode, result: float):
        """回传阶段: 更新节点统计信息"""
        while node:
            node.visits += 1
            if node.player_id == self.player_id:
                node.wins += result
            node = node.parent
    
    def _is_terminal(self, node: MCTSNode) -> bool:
        """检查是否为终止节点"""
        return self._check_game_result(node.state) != -1
    
    def _check_game_result(self, board: Board) -> int:
        """检查游戏结果"""
        red_pieces = board.get_player_pieces(0)
        blue_pieces = board.get_player_pieces(1)
        
        if not red_pieces:
            return 1  # 蓝方胜
        if not blue_pieces:
            return 0  # 红方胜
            
        # 处理特殊终局情况
        if len(red_pieces) == 1 and len(blue_pieces) == 1:
            rr, rc = red_pieces[0]
            br, bc = blue_pieces[0]
            red_piece = board.get_piece(rr, rc)
            blue_piece = board.get_piece(br, bc)
            
            if red_piece.revealed and blue_piece.revealed:
                can_red_attack = self.reward_function.can_capture(red_piece.strength, blue_piece.strength)
                can_blue_attack = self.reward_function.can_capture(blue_piece.strength, red_piece.strength)
                
                if not can_red_attack and not can_blue_attack:
                    return 2  # 平局
                    
        return -1  # 游戏继续

    def choose_action(self, board: Board) -> Optional[Tuple]:
        """选择最佳动作"""
        root = MCTSNode(state=copy.deepcopy(board), player_id=self.player_id)
        
        # 运行MCTS模拟
        for _ in range(self.simulation_limit):
            node = self._select(root)
            if not self._is_terminal(node):
                new_node = self._expand(node)
                if new_node:
                    result = self._simulate(new_node)
                    self._backpropagate(new_node, result)
            
        # 选择访问次数最多的动作
        if not root.children:
            return None
            
        best_child = max(root.children, key=lambda n: n.visits)
        return best_child.action
    
    def take_turn(self, board: Board) -> bool:
        """执行一个回合"""
        action = self.choose_action(board)
        if not action:
            return False
            
        action_type, pos1, pos2 = action
        if action_type == "reveal":
            r, c = pos1
            piece = board.get_piece(r, c)
            if piece and piece.player == self.player_id and not piece.revealed:
                piece.revealed = True
                return True
        else:  # move
            return board.try_move(pos1, pos2)
        
        return False
    
# if __name__ == "__main__":
#     from AgentFight import RandomPlayer

#     mcts_agent = MCTSAgent(
#         player_id=0,
#         simulation_limit=100,  # 每步模拟次数
#         exploration_constant=1.414  # UCB参数
#     )

#     random_agent = RandomPlayer(player_id=1)
    
#     game = AG.Game(mcts_agent, random_agent)
#     game.run()