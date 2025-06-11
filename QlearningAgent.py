import numpy as np
import random
import copy
from typing import Tuple, List, Dict
from base import Player, BaseTrainer, GameEnvironment, Board
from utils import save_model_data, load_model_data

class QLearningAgent(Player):
    """Q-learning智能体"""
    
    def __init__(self, player_id: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        super().__init__(player_id)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        
        self.ai_type = f"QL (ε={epsilon:.2f})" if epsilon > 0 else "QL (Trained)"
    
    def _board_to_state(self, board: Board) -> np.ndarray:
        """将Board对象转换为状态数组（仅在需要时使用）"""
        env = GameEnvironment()
        env.board = board
        env.current_player = self.player_id
        return env.get_state()
        
    def get_state_key(self, board: Board) -> str:
        """将Board转换为可哈希的键"""
        state = self._board_to_state(board)
        return str(state.round(3).tolist())
    
    def get_action_key(self, action: Tuple) -> str:
        """将动作转换为可哈希的键"""
        return str(action)
    
    def choose_action(self, board: Board, valid_actions: List[Tuple]) -> Tuple:
        """使用epsilon-greedy策略选择动作 - 直接使用Board对象"""
        state_key = self.get_state_key(board)
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            best_action = None
            best_q_value = float('-inf')
            
            for action in valid_actions:
                action_key = self.get_action_key(action)
                q_value = self.q_table[state_key].get(action_key, 0.0)
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            return best_action if best_action else random.choice(valid_actions)
    
    def update_q_value(self, board_before: Board, action: Tuple, reward: float, 
                      board_after: Board, next_valid_actions: List[Tuple]):
        """更新Q值 - 直接使用Board对象"""
        state_key = self.get_state_key(board_before)
        action_key = self.get_action_key(action)
        next_state_key = self.get_state_key(board_after)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
        
        max_next_q = 0.0
        if next_valid_actions and next_state_key in self.q_table:
            max_next_q = max([
                self.q_table[next_state_key].get(self.get_action_key(a), 0.0)
                for a in next_valid_actions
            ])
        
        current_q = self.q_table[state_key][action_key]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state_key][action_key] = new_q
    
    def take_turn(self, board: Board) -> bool:
        """为游戏集成实现的take_turn方法"""
        valid_actions = board.get_all_possible_moves(self.player_id)
        
        if not valid_actions:
            return False
        
        action = self.choose_action(board, valid_actions)
        
        action_type, pos1, pos2 = action
        
        if action_type == "reveal":
            r, c = pos1
            piece = board.get_piece(r, c)
            if piece and piece.player == self.player_id and not piece.revealed:
                piece.revealed = True
                return True
                
        elif action_type == "move":
            if board.try_move(pos1, pos2):
                return True
        
        return False
    
    def save_model(self, filename: str):
        """保存模型"""
        data = {
            'q_table': self.q_table,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'stats': self.training_stats
        }
        save_model_data(data, f"{filename}.pkl")
    
    def load_model(self, filename: str):
        """加载模型"""
        data = load_model_data(f"{filename}.pkl")
        if data:
            self.q_table = data.get('q_table', {})
            self.learning_rate = data.get('learning_rate', self.learning_rate)
            self.discount_factor = data.get('discount_factor', self.discount_factor)
            self.epsilon = data.get('epsilon', self.epsilon)
            self.training_stats = data.get('stats', self.training_stats)
            return True
        return False

class QLearningTrainer(BaseTrainer):
    """Q-learning训练器"""
    
    def _agent_choose_action(self, board: Board, valid_actions: List[Tuple]) -> Tuple:
        """智能体选择动作"""
        return self.agent.choose_action(board, valid_actions)
    
    def _agent_update(self, board_before: Board, action: Tuple, reward: float, 
                     board_after: Board, result: int):
        """更新智能体"""
        if result == -1:
            next_valid_actions = board_after.get_all_possible_moves(self.agent.player_id)
        else:
            next_valid_actions = []
        
        self.agent.update_q_value(board_before, action, reward, board_after, next_valid_actions)
    
    def save_model(self, filename: str):
        """保存模型"""
        self.agent.save_model(filename)


def train_or_load_model(force_retrain=False, episodes=1000):
    """训练或加载Q-Learning模型"""
    from AgentFight import RandomPlayer
    # from base import Game
    import os
    
    model_name = "final_QLearningAgent"
    
    # 创建智能体
    ql_agent = QLearningAgent(player_id=0, learning_rate=0.1, epsilon=0.1)
    random_opponent = RandomPlayer(player_id=1)
    
    # 检查是否存在已训练的模型
    model_path = os.path.join("model_data", f"{model_name}.pkl")
    model_exists = os.path.exists(model_path)
    
    if model_exists and not force_retrain:
        print(f"发现已训练的模型: {model_path}")
        if ql_agent.load_model(model_name):
            print("模型加载成功!")
            # 设置为测试模式（不探索）
            ql_agent.epsilon = 0.0
            ql_agent.ai_type = "QL (Trained)"
        else:
            print("模型加载失败，将重新训练...")
            model_exists = False
    
    if not model_exists or force_retrain:
        if force_retrain:
            print("强制重新训练模型...")
        else:
            print("未找到已训练模型，开始训练...")
        
        # 训练
        trainer = QLearningTrainer(ql_agent, random_opponent)
        history = trainer.train(episodes=episodes)
        
        print(f"训练完成! 最终胜率: {ql_agent.get_stats()['win_rate']:.3f}")
        
        # 设置为测试模式
        ql_agent.epsilon = 0.0
        ql_agent.ai_type = "QL (Trained)"
    
    return ql_agent, random_opponent

# 使用示例
if __name__ == "__main__":
    from base import Game
    import argparse
    
    parser = argparse.ArgumentParser(description='Q-Learning Agent 训练和测试')
    parser.add_argument('--retrain', action='store_true', help='强制重新训练模型')
    parser.add_argument('--episodes', type=int, default=1000, help='训练回合数')
    parser.add_argument('--test-games', type=int, default=1, help='测试游戏数量')
    parser.add_argument('--no-display', action='store_true', help='不显示游戏界面')
    
    args = parser.parse_args()
    
    # 训练或加载模型
    ql_agent, random_opponent = train_or_load_model(
        force_retrain=args.retrain, 
        episodes=args.episodes
    )
    
    # 测试
    print(f"\n开始测试 {args.test_games} 场游戏...")
    wins = 0
    
    for i in range(args.test_games):
        game = Game(ql_agent, random_opponent, display=not args.no_display, delay=0.5)
        result = game.run()
        
        if result == 0:  # QL agent wins
            wins += 1
            
        print(f"游戏 {i+1}: {'胜利' if result == 0 else '失败' if result == 1 else '平局'}")
    
    print(f"\n测试结果: {wins}/{args.test_games} 胜率: {wins/args.test_games:.3f}")
    
# python QlearningAgent.py --retrain --episodes 1000 --test-games 10