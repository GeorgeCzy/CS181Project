import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
from collections import deque, namedtuple
from typing import Tuple, List, Optional, Dict, Any
from new_sim import Board, Player

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 经验回放缓冲区
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    """深度Q网络"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super(DQN, self).__init__()
        
        # 网络架构
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DoushouqiEnv:
    """斗兽棋强化学习环境 - 优化版"""
    
    def __init__(self):
        self.board = None
        self.current_player = 0
        self.action_space_size = self._calculate_action_space_size()
        self.reset()
    
    def _calculate_action_space_size(self):
        """计算动作空间大小"""
        # 翻开动作: 7*8 = 56
        # 移动动作: 7*8*4 = 224 (每个位置最多4个方向)
        return 56 + 224
    
    def reset(self) -> torch.Tensor:
        """重置环境"""
        self.board = Board()
        self.current_player = 0
        return self.get_state()
    
    def get_state(self) -> torch.Tensor:
        """获取当前状态的张量表示"""
        # 创建多通道状态表示
        state = np.zeros((7, 8, 6))  # 6个通道
        
        for r in range(7):
            for c in range(8):
                piece = self.board.get_piece(r, c)
                if piece:
                    # 通道0-1: 玩家0和玩家1的棋子位置
                    state[r, c, piece.player] = 1
                    
                    # 通道2: 棋子强度
                    state[r, c, 2] = piece.strength / 8.0
                    
                    # 通道3: 是否翻开
                    state[r, c, 3] = 1 if piece.revealed else 0
                    
                    # 通道4: 我方棋子
                    if piece.player == self.current_player:
                        state[r, c, 4] = 1
                    
                    # 通道5: 对方棋子  
                    if piece.player != self.current_player:
                        state[r, c, 5] = 1
        
        # 展平并转换为torch张量
        flat_state = state.flatten()
        return torch.FloatTensor(flat_state).to(device)
    
    def action_to_index(self, action: Tuple) -> int:
        """将动作转换为索引"""
        action_type, pos1, pos2 = action
        
        if action_type == "reveal":
            r, c = pos1
            return r * 8 + c  # 0-55
        
        elif action_type == "move":
            r1, c1 = pos1
            r2, c2 = pos2
            
            # 计算移动方向
            dr, dc = r2 - r1, c2 - c1
            direction_map = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
            direction = direction_map.get((dr, dc), 0)
            
            return 56 + r1 * 8 * 4 + c1 * 4 + direction  # 56-279
        
        return 0
    
    def index_to_action(self, index: int) -> Optional[Tuple]:
        """将索引转换为动作"""
        if index < 56:  # 翻开动作
            r, c = divmod(index, 8)
            return ("reveal", (r, c), None)
        
        else:  # 移动动作
            move_index = index - 56
            r1 = move_index // (8 * 4)
            remaining = move_index % (8 * 4)
            c1 = remaining // 4
            direction = remaining % 4
            
            direction_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
            dr, dc = direction_map[direction]
            r2, c2 = r1 + dr, c1 + dc
            
            if 0 <= r2 < 7 and 0 <= c2 < 8:
                return ("move", (r1, c1), (r2, c2))
        
        return None
    
    def get_valid_actions(self, player_id: int) -> List[int]:
        """获取有效动作索引列表"""
        valid_indices = []
        
        # 翻开动作
        for r in range(7):
            for c in range(8):
                piece = self.board.get_piece(r, c)
                if piece and piece.player == player_id and not piece.revealed:
                    index = self.action_to_index(("reveal", (r, c), None))
                    valid_indices.append(index)
        
        # 移动动作
        for r in range(7):
            for c in range(8):
                piece = self.board.get_piece(r, c)
                if piece and piece.player == player_id and piece.revealed:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 7 and 0 <= nc < 8:
                            temp_board = copy.deepcopy(self.board)
                            if temp_board.try_move((r, c), (nr, nc)):
                                action = ("move", (r, c), (nr, nc))
                                index = self.action_to_index(action)
                                valid_indices.append(index)
        
        return valid_indices
    
    def step(self, action_index: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """执行动作"""
        action = self.index_to_action(action_index)
        if action is None:
            return self.get_state(), -1.0, False, {"invalid": True}
        
        action_type, pos1, pos2 = action
        reward = 0
        done = False
        info = {}
        
        # 验证动作有效性
        valid_actions = self.get_valid_actions(self.current_player)
        if action_index not in valid_actions:
            return self.get_state(), -1.0, False, {"invalid": True}
        
        # 执行动作
        if action_type == "reveal":
            r, c = pos1
            piece = self.board.get_piece(r, c)
            if piece and piece.player == self.current_player and not piece.revealed:
                piece.revealed = True
                reward = 0.1
            
        elif action_type == "move":
            start_pos, end_pos = pos1, pos2
            piece_before = self.board.get_piece(end_pos[0], end_pos[1])
            
            if self.board.try_move(start_pos, end_pos):
                piece_after = self.board.get_piece(end_pos[0], end_pos[1])
                
                if piece_before and piece_before.player != self.current_player:
                    if piece_after and piece_after.player == self.current_player:
                        reward = piece_before.strength * 0.2
                    else:
                        reward = -0.3
                else:
                    reward = 0.05
        
        # 检查游戏结束
        red_pieces = self.board.get_player_pieces(0)
        blue_pieces = self.board.get_player_pieces(1)
        
        if not red_pieces:
            done = True
            reward += 10 if self.current_player == 1 else -10
        elif not blue_pieces:
            done = True
            reward += 10 if self.current_player == 0 else -10
        elif len(red_pieces) == 1 and len(blue_pieces) == 1:
            rr, rc = red_pieces[0]
            br, bc = blue_pieces[0]
            if not self.board.is_adjacent((rr, rc), (br, bc)):
                done = True
                reward += 0
        
        # 切换玩家
        self.current_player = 1 - self.current_player
        
        return self.get_state(), reward, done, info

class DQNAgent(Player):
    """DQN智能体"""
    
    def __init__(self, player_id: int, state_size: int = 336, action_size: int = 280,
                 learning_rate: float = 1e-4, epsilon: float = 0.1, 
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 batch_size: int = 64, memory_size: int = 10000):
        
        super().__init__(player_id)
        
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # 创建主网络和目标网络
        self.q_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(memory_size)
        
        # 更新目标网络
        self.update_target_network()
        
        # 训练统计
        self.losses = []
        
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def choose_action(self, state: torch.Tensor, valid_actions: List[int]) -> int:
        """选择动作"""
        if not valid_actions:
            return 0
            
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            
            # 创建掩码，只考虑有效动作
            masked_q_values = torch.full((self.action_size,), float('-inf')).to(device)
            for action in valid_actions:
                masked_q_values[action] = q_values[0][action]
            
            return masked_q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """经验回放学习"""
        if len(self.memory) < self.batch_size:
            return
        
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # 转换为张量
        state_batch = torch.stack(batch.state)
        action_batch = torch.LongTensor(batch.action).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        next_state_batch = torch.stack(batch.next_state)
        done_batch = torch.BoolTensor(batch.done).to(device)
        
        # 计算当前Q值
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (0.99 * next_q_values * ~done_batch)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 记录损失
        self.losses.append(loss.item())
        
        # 衰减epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def take_turn(self, board) -> bool:
        """游戏集成接口"""
        env = DoushouqiEnv()
        env.board = copy.deepcopy(board)
        env.current_player = self.player_id
        
        state = env.get_state()
        valid_actions = env.get_valid_actions(self.player_id)
        
        if not valid_actions:
            return False
        
        # 测试时不探索
        old_epsilon = self.epsilon
        self.epsilon = 0.0
        
        action_index = self.choose_action(state, valid_actions)
        action = env.index_to_action(action_index)
        
        self.epsilon = old_epsilon
        
        if action:
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
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        try:
            checkpoint = torch.load(filepath, map_location=device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.losses = checkpoint.get('losses', [])
            print(f"模型已从 {filepath} 加载")
        except FileNotFoundError:
            print(f"模型文件 {filepath} 不存在")

class DQNTrainer:
    """DQN训练器"""
    
    def __init__(self, agent: DQNAgent, opponent_agent: Player):
        self.agent = agent
        self.opponent_agent = opponent_agent
        self.env = DoushouqiEnv()
        
    def train_episode(self) -> Tuple[float, int]:
        """训练一个回合"""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            if self.env.current_player == self.agent.player_id:
                # DQN智能体回合
                valid_actions = self.env.get_valid_actions(self.agent.player_id)
                
                if not valid_actions:
                    break
                
                action = self.agent.choose_action(state, valid_actions)
                next_state, reward, done, info = self.env.step(action)
                
                # 存储经验
                self.agent.store_experience(state, action, reward, next_state, done)
                
                # 学习
                self.agent.replay()
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            else:
                # 对手回合
                self.opponent_agent.take_turn(self.env.board)
                state = self.env.get_state()
                self.env.current_player = 1 - self.env.current_player
        
        return total_reward, steps
    
    def train(self, episodes: int = 10000, target_update_freq: int = 100, save_interval: int = 1000):
        """训练"""
        print(f"开始DQN训练 {episodes} 回合，使用设备: {device}")
        
        rewards_history = []
        
        save_path = r"model_data/"
        
        for episode in range(episodes):
            total_reward, steps = self.train_episode()
            rewards_history.append(total_reward)
            
            # 更新目标网络
            if episode % target_update_freq == 0:
                self.agent.update_target_network()
            
            # 打印进度
            if episode % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:]) if rewards_history else 0
                avg_loss = np.mean(self.agent.losses[-100:]) if self.agent.losses else 0
                print(f"回合 {episode}: 平均奖励 = {avg_reward:.2f}, "
                      f"平均损失 = {avg_loss:.4f}, epsilon = {self.agent.epsilon:.3f}")
            
            # 保存模型
            if episode % save_interval == 0:
                self.agent.save_model(save_path + f"dqn_model_episode_{episode}.pth")
        
        print("训练完成！")
        self.agent.save_model(save_path + "final_dqn_model.pth")
        
        return rewards_history

# 使用示例
if __name__ == "__main__":
    from AgentFight import RandomPlayer
    
    # 创建DQN智能体
    dqn_agent = DQNAgent(
        player_id=1, 
        learning_rate=1e-4, 
        epsilon=0.9,
        epsilon_decay=0.995,
        batch_size=64
    )
    random_opponent = RandomPlayer(player_id=0)
    
    # 创建训练器
    trainer = DQNTrainer(dqn_agent, random_opponent)
    
    # 开始训练
    trainer.train(episodes=5000, save_interval=1000)  
    
    print("DQN训练完成!")