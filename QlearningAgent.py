import numpy as np
import random
import copy
from typing import Tuple, List, Optional, Dict, Any
import os
import pickle
from utils import RewardFunction, Environment
from base import Board, Player


class QLearningAgent(Player):
    """Q-learning智能体"""
    
    def __init__(self, player_id: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        super().__init__(player_id)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}  # 状态-动作价值表
        self.last_state = None
        self.last_action = None
        
        if epsilon > 0:
            self.ai_type = f"QL (ε={epsilon:.2f})"
        else:
            self.ai_type = "QL (Trained)"
        
    def get_state_key(self, state: np.ndarray) -> str:
        """将状态转换为可哈希的键"""
        return str(state.round(3).tolist())  # 保留3位小数避免浮点精度问题
    
    def get_action_key(self, action: Tuple) -> str:
        """将动作转换为可哈希的键"""
        return str(action)
    
    def choose_action(self, state: np.ndarray, valid_actions: List[Tuple]) -> Tuple:
        """使用epsilon-greedy策略选择动作"""
        state_key = self.get_state_key(state)
        
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.choice(valid_actions)
        else:
            # 利用：选择Q值最高的动作
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
            
            if best_action is None:
                best_action = random.choice(valid_actions)
            
            return best_action
    
    def update_q_value(self, state: np.ndarray, action: Tuple, reward: float, 
                      next_state: np.ndarray, next_valid_actions: List[Tuple]):
        """更新Q值"""
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)
        next_state_key = self.get_state_key(next_state)
        
        # 初始化Q表
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
        
        # 计算下一状态的最大Q值
        max_next_q = 0.0
        if next_valid_actions and next_state_key in self.q_table:
            max_next_q = max([
                self.q_table[next_state_key].get(self.get_action_key(a), 0.0)
                for a in next_valid_actions
            ])
        
        # Q-learning更新公式
        current_q = self.q_table[state_key][action_key]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state_key][action_key] = new_q
    
    def take_turn(self, board) -> bool:
        """为游戏集成实现的take_turn方法"""
        env = Environment()
        env.board = copy.deepcopy(board)
        env.current_player = self.player_id
        
        state = env.get_state()
        valid_actions = env.get_valid_actions(self.player_id)
        
        if not valid_actions:
            return False
        
        action = self.choose_action(state, valid_actions)
        
        # 执行动作
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
    
    def save_q_table(self, filename: str):
        """保存Q表"""
        try:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            with open(filename, 'wb') as f:
                pickle.dump(self.q_table, f)
        except Exception as e:
            print(f"保存Q表失败: {e}")

    
    def load_q_table(self, filename: str):
        """加载Q表"""
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print(f"Q表文件 {filename} 不存在, 从空Q表开始")

class QLearningTrainer:
    """Q-learning训练器"""
    
    def __init__(self, agent: QLearningAgent, opponent_agent: Player):
        self.agent = agent
        self.opponent_agent = opponent_agent
        self.env = Environment()
        self.wins = 0
        self.losses = 0
        
    def train_episode(self, opponent_agent = None) -> Tuple[float, int]:
        """训练一个回合，返回总奖励和步数"""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        if opponent_agent is not None:
            self.opponent_agent = opponent_agent
        
        while True:
            if self.env.current_player == self.agent.player_id:
                # RL智能体回合
                valid_actions = self.env.get_valid_actions(self.agent.player_id)
                
                if not valid_actions:
                    break
                
                action = self.agent.choose_action(state, valid_actions)
                next_state, reward, result, _ = self.env.step(action)
                
                # 更新Q值
                if result == -1:
                    next_valid_actions = self.env.get_valid_actions(self.agent.player_id)
                else:
                    next_valid_actions = []
                
                self.agent.update_q_value(state, action, reward, next_state, next_valid_actions)
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if result != -1:
                    break
            else:
                # 对手回合
                if isinstance(self.opponent_agent, QLearningAgent):
                    valid_actions = self.env.get_valid_actions(self.opponent_agent.player_id)
                    if valid_actions:
                        action = self.opponent_agent.choose_action(state, valid_actions)
                        state, _, result, _ = self.env.step(action)
                else:
                    # 随机对手或其他类型
                    if self.opponent_agent.take_turn(self.env.board):
                        state = self.env.get_state()
                        self.env.current_player = 1 - self.env.current_player
                
                if result != -1:
                    break

            if steps >= 1000:
                result = 2
                break
        
        return total_reward, steps, result
    
    def train(self, opponent_agent = None, episodes: int = 10000, save_interval: int = 1000):
        """训练指定回合数"""
        print(f"开始训练 {episodes} 回合...")
        save_path = r"model_data/"
        
        for episode in range(episodes):
            total_reward, steps, result = self.train_episode(opponent_agent)

            if result == self.agent.player_id:
                self.wins += 1
            elif result == 1 - self.agent.player_id:
                self.losses += 1
            
            if episode % 10 == 0:
                print(f"回合 {episode}: 奖励 = {total_reward:.2f}, 步数 = {steps}, 胜 = {self.wins}, 负 = {self.losses}")
                self.wins = 0
                self.losses = 0
            
            if episode % save_interval == 0:
                self.agent.save_q_table(save_path + f"q_table_episode_{episode}.pkl")
        
        print("训练完成！")
        self.agent.save_q_table(save_path + "final_q_table.pkl")

# 使用示例
if __name__ == "__main__":
    from new_sim import RandomPlayer
    
    # 创建智能体
    rl_agent = QLearningAgent(player_id=0, learning_rate=0.1, epsilon=0.1)
    random_opponent = RandomPlayer(player_id=1)
    
    # 创建训练器
    trainer = QLearningTrainer(rl_agent, random_opponent)
    
    # 开始训练
    trainer.train(episodes=5000)
    
    print("训练完成！可以在游戏中使用训练好的智能体了。")