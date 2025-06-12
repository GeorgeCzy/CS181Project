import numpy as np
import random
import math
import copy
from typing import List, Tuple, Dict, Optional
from base import Board, Player, BaseTrainer, compare_strength
from utils import RewardFunction, save_model_data, load_model_data

class MCTSNode:
    """蒙特卡洛树搜索节点"""
    def __init__(self, state: Board, parent=None, action=None, player_id: int = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.player_id = player_id
        self.children: List[MCTSNode] = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = self._get_valid_actions()
        
    def _get_valid_actions(self) -> List[Tuple]:
        """获取所有有效动作"""
        return self.state.get_all_possible_moves(self.player_id)
    
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
        
        # 用于保存的数据
        self.tree_statistics = {
            'total_simulations': 0,
            'average_tree_depth': 0.0,
            'best_moves_history': []
        }
    
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
            if piece and not piece.revealed:
                piece.reveal()
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
    
    def _simulate(self, node: MCTSNode) -> float:
        """模拟阶段: 使用启发式随机策略模拟到游戏结束"""
        state = copy.deepcopy(node.state)
        current_player = node.player_id
        last_move = None
        repeated_moves = 0
        max_simulation_steps = 200  # 防止无限循环
        steps = 0
        
        while steps < max_simulation_steps:
            result = self._check_game_result(state)
            if result != -1:  # 游戏结束
                if result == self.player_id:
                    return 1.0  # 胜利
                elif result == 1 - self.player_id:
                    return 0.0  # 失败
                else:
                    return 0.5  # 平局
            
            # 获取所有可能的动作
            valid_actions = state.get_all_possible_moves(current_player)
            
            if not valid_actions:
                return 0.5  # 无法行动，视为平局
            
            # 使用启发式策略选择动作
            action = self._select_simulation_action(state, valid_actions, current_player, last_move)
            
            # 执行动作
            success = False
            if action[0] == "reveal":
                r, c = action[1]
                piece = state.get_piece(r, c)
                if piece and not piece.revealed:
                    piece.reveal()
                    success = True
                    last_move = None
                    repeated_moves = 0
            else:  # move
                success = state.try_move(action[1], action[2])
                if success:
                    # 检查重复移动
                    if (last_move and action[0] == "move" and last_move[0] == "move" and
                        ((action[1] == last_move[2] and action[2] == last_move[1]) or
                         (action[1] == last_move[1] and action[2] == last_move[2]))):
                        repeated_moves += 1
                        if repeated_moves >= 3:
                            return 0.5  # 连续重复移动视为平局
                    else:
                        repeated_moves = 0
                    last_move = action
            
            if not success:
                return 0.5  # 动作失败，视为平局
            
            current_player = 1 - current_player
            steps += 1
        
        return 0.5  # 超出最大步数，视为平局
    
    def _select_simulation_action(self, state: Board, valid_actions: List[Tuple], 
                                 current_player: int, last_move: Optional[Tuple]) -> Tuple:
        """为模拟选择动作（启发式策略）"""
        if random.random() < 0.7:  # 70%概率使用启发式策略
            best_value = float('-inf')
            best_action = None
            
            for action in valid_actions:
                value = 0.0
                
                # 避免重复移动
                if (last_move and action[0] == "move" and last_move[0] == "move" and
                    ((action[1] == last_move[2] and action[2] == last_move[1]) or
                     (action[1] == last_move[1] and action[2] == last_move[2]))):
                    value -= 2.0
                
                # 评估动作价值
                if action[0] == "reveal":
                    value += 1.0  # 鼓励翻开棋子
                else:  # move
                    start_pos, end_pos = action[1], action[2]
                    moving_piece = state.get_piece(start_pos[0], start_pos[1])
                    target_piece = state.get_piece(end_pos[0], end_pos[1])
                    
                    if target_piece and target_piece.player != current_player:
                        # 评估吃子价值
                        if target_piece.revealed:
                            if compare_strength(moving_piece.strength, target_piece.strength) == 1:
                                value += 3.0 * self.reward_function.get_piece_value(target_piece.strength)
                    
                    # 考虑威胁和机会
                    threat, opportunity = self.reward_function.evaluate_position(state, end_pos, current_player)
                    value += opportunity - abs(threat)
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            return best_action if best_action else random.choice(valid_actions)
        
        return random.choice(valid_actions)
    
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
            
        if len(red_pieces) == 1 and len(blue_pieces) == 1:
            rr, rc = red_pieces[0]
            br, bc = blue_pieces[0]
            red_piece = board.get_piece(rr, rc)
            blue_piece = board.get_piece(br, bc)
               
            if compare_strength(red_piece.strength, blue_piece.strength) == 0:
                return 2  # 平局
                    
        return -1  # 游戏继续

    def choose_action(self, board: Board, valid_actions: List[Tuple]) -> Tuple:
        """选择最佳动作"""
        if not valid_actions:
            return None
            
        root = MCTSNode(state=copy.deepcopy(board), player_id=self.player_id)
        
        # 运行MCTS模拟
        for _ in range(self.simulation_limit):
            node = self._select(root)
            if not self._is_terminal(node):
                new_node = self._expand(node)
                if new_node:
                    result = self._simulate(new_node)
                    self._backpropagate(new_node, result)
            
            self.tree_statistics['total_simulations'] += 1
        
        # 选择访问次数最多的动作
        if not root.children:
            return random.choice(valid_actions)
            
        best_child = max(root.children, key=lambda n: n.visits)
        
        # 记录最佳移动
        self.tree_statistics['best_moves_history'].append(best_child.action)
        if len(self.tree_statistics['best_moves_history']) > 1000:
            self.tree_statistics['best_moves_history'].pop(0)
        
        return best_child.action
    
    def take_turn(self, board: Board) -> bool:
        """执行一个回合"""
        valid_actions = board.get_all_possible_moves(self.player_id)
        
        if not valid_actions:
            return False
            
        action = self.choose_action(board, valid_actions)
        if not action:
            return False
            
        action_type, pos1, pos2 = action
        if action_type == "reveal":
            r, c = pos1
            piece = board.get_piece(r, c)
            if piece and not piece.revealed:
                piece.reveal()
                return True
        else:  # move
            return board.try_move(pos1, pos2)
        
        return False
    
    def save_model(self, filename: str):
        """保存模型"""
        data = {
            'simulation_limit': self.simulation_limit,
            'exploration_constant': self.exploration_constant,
            'tree_statistics': self.tree_statistics,
            'stats': self.training_stats
        }
        save_model_data(data, f"{filename}.pkl")
    
    def load_model(self, filename: str):
        """加载模型"""
        data = load_model_data(f"{filename}.pkl")
        if data:
            self.simulation_limit = data.get('simulation_limit', self.simulation_limit)
            self.exploration_constant = data.get('exploration_constant', self.exploration_constant)
            self.tree_statistics = data.get('tree_statistics', self.tree_statistics)
            self.training_stats = data.get('stats', self.training_stats)

class MCTSTrainer(BaseTrainer):
    """MCTS训练器"""
    
    def __init__(self, agent: MCTSAgent, opponent_agent: Player, **kwargs):
        super().__init__(agent, opponent_agent, **kwargs)
        
    def _agent_choose_action(self, board: Board, valid_actions: List[Tuple]) -> Tuple:
        """智能体选择动作"""
        return self.agent.choose_action(board, valid_actions)
    
    def _agent_update(self, board_before: Board, action: Tuple, reward: float, 
                     board_after: Board, result: int):
        """MCTS不需要在线学习，但可以记录统计信息"""
        pass
    
    def save_model(self, filename: str):
        """保存模型"""
        self.agent.save_model(filename)

def train_or_load_model(force_retrain=False, episodes=100):
    """训练或加载MCTS模型"""
    from AgentFight import RandomPlayer
    # from base import Game
    import os
    
    model_name = "final_MCTSAgent"
    
    # 创建智能体
    mcts_agent = MCTSAgent(
        player_id=0,
        simulation_limit=100,
        exploration_constant=1.414
    )
    random_opponent = RandomPlayer(player_id=1)
    
    # 检查是否存在已训练的模型
    model_path = os.path.join("model_data", f"{model_name}.pkl")
    model_exists = os.path.exists(model_path)
    
    if model_exists and not force_retrain:
        print(f"发现已训练的模型: {model_path}")
        if mcts_agent.load_model(model_name):
            print("模型加载成功!")
        else:
            print("模型加载失败，将重新训练...")
            model_exists = False
    
    if not model_exists or force_retrain:
        if force_retrain:
            print("强制重新训练模型...")
        else:
            print("未找到已训练模型，开始训练...")
        
        # 训练 (MCTS训练较少轮次)
        trainer = MCTSTrainer(mcts_agent, random_opponent)
        history = trainer.train(episodes=episodes)
        
        print(f"训练完成! 最终胜率: {mcts_agent.get_stats()['win_rate']:.3f}")
    
    return mcts_agent, random_opponent

# 使用示例
if __name__ == "__main__":
    import argparse
    from base import Game
    
    parser = argparse.ArgumentParser(description='MCTS Agent 训练和测试')
    parser.add_argument('--retrain', action='store_true', help='强制重新训练模型')
    parser.add_argument('--episodes', type=int, default=100, help='训练回合数')
    parser.add_argument('--test-games', type=int, default=1, help='测试游戏数量')
    parser.add_argument('--no-display', action='store_true', help='不显示游戏界面')
    
    args = parser.parse_args()
    
    # 训练或加载模型
    mcts_agent, random_opponent = train_or_load_model(
        force_retrain=args.retrain, 
        episodes=args.episodes
    )
    
    # 测试
    print(f"\n开始测试 {args.test_games} 场游戏...")
    wins = 0
    
    for i in range(args.test_games):
        game = Game(mcts_agent, random_opponent, display=not args.no_display, delay=0.5)
        result = game.run()
        
        if result == 0:  # MCTS agent wins
            wins += 1
            
        print(f"游戏 {i+1}: {'胜利' if result == 0 else '失败' if result == 1 else '平局'}")
    
    print(f"\n测试结果: {wins}/{args.test_games} 胜率: {wins/args.test_games:.3f}")
    
# python MCTS.py --retrain --episodes 100 --test-games 3