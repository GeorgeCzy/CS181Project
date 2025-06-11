import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from datetime import datetime
import copy
from collections import deque, namedtuple
from typing import Tuple, List, Optional, Dict, Any
from base import Board, Player, BaseTrainer
from utils import SimpleReward, save_model_data, load_model_data
import time

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

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

# class ReplayBuffer:
#     """经验回放缓冲区"""
#     def __init__(self, capacity: int):
#         self.buffer = deque(maxlen=capacity)
    
#     def push(self, state, action, reward, next_state, result):
#         self.buffer.append(Experience(state, action, reward, next_state, result))
    
#     def sample(self, batch_size: int):
#         return random.sample(self.buffer, batch_size)
    
#     def __len__(self):
#         return len(self.buffer)

class DQN(nn.Module):
    """改进的深度Q网络 - Dueling DQN"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super(DQN, self).__init__()
        
        # 共享特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Dueling DQN: 价值流和优势流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling DQN 公式: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class DoubleDQN(nn.Module):
    """Double DQN 网络"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super(DoubleDQN, self).__init__()
        
        # 更深的网络
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
    def forward(self, x):
        return self.network(x)
    

class DQNAgent(Player):
    """改进的 DQN 智能体 - 支持 Dueling + Double DQN"""
    
    def __init__(self, player_id: int, state_size: int = 336, action_size: int = 280,
                 learning_rate: float = 1e-4, epsilon: float = 0.1, 
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 batch_size: int = 64, memory_size: int = 10000,
                 use_dueling: bool = True, use_double: bool = True):
        
        super().__init__(player_id)
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate  # 保存初始学习率
        self.lr_decay_rate = 0.999
        self.lr_min = 0.0001
        self.lr_max = learning_rate * 2
        self.adaptive_lr = True
        self.lr_adjustment_frequency = 50  # 每50个episode调整一次
        # self.performance_window = 20       # 基于最近20个episode的表现
        
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.use_dueling = use_dueling
        self.use_double = use_double
        self.reward_buffer = deque(maxlen=1000)  # 用于奖励归一化
        
        self.training_mode = True
        self.test_mode_epsilon = 0.0  # 测试模式下的epsilon
        
        # 创建网络
        if use_dueling:
            self.q_network = DQN(state_size, action_size).to(device)
            self.target_network = DQN(state_size, action_size).to(device)
        else:
            self.q_network = DoubleDQN(state_size, action_size).to(device)
            self.target_network = DoubleDQN(state_size, action_size).to(device)
            
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # 经验回放缓冲区
        self.memory = PrioritizedReplayBuffer(memory_size)
        
        # 使用改进的奖励函数
        self.reward_function = SimpleReward()
        
        # 更新目标网络
        self.update_target_network()
        
        # 训练统计
        self.losses = []
        self.episode_count = 0
        
        # 学习率调度器
        self.base_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=500, gamma=0.95
        )
                
        ai_type = "D3QN" if use_dueling and use_double else "DuelDQN" if use_dueling else "DoubleDQN" if use_double else "DQN"
        self.ai_type = f"{ai_type} (ε={epsilon:.2f})" if epsilon > 0 else f"{ai_type} (Trained)"
        
    def _board_to_state(self, board: Board) -> torch.Tensor:
        """将Board对象转换为状态张量"""
        # 创建多通道状态表示
        state = np.zeros((7, 8, 6))  # 6个通道
        
        for r in range(7):
            for c in range(8):
                piece = board.get_piece(r, c)
                if piece:
                    # 通道0-1: 玩家0和玩家1的棋子位置
                    state[r, c, piece.player] = 1
                    
                    # 通道2: 棋子强度
                    state[r, c, 2] = piece.strength / 8.0
                    
                    # 通道3: 是否翻开
                    state[r, c, 3] = 1 if piece.revealed else 0
                    
                    # 通道4: 我方棋子
                    if piece.player == self.player_id:
                        state[r, c, 4] = 1
                    
                    # 通道5: 对方棋子  
                    if piece.player != self.player_id:
                        state[r, c, 5] = 1
        
        # 展平并转换为torch张量
        flat_state = state.flatten()
        return torch.FloatTensor(flat_state).to(device)
    
    def _action_to_index(self, action: Tuple) -> int:
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
    
    def _index_to_action(self, index: int) -> Optional[Tuple]:
        """将索引转换为动作，确保返回正确的三元组格式"""
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
            
            # 验证目标位置是否在棋盘范围内
            if 0 <= r2 < 7 and 0 <= c2 < 8:
                return ("move", (r1, c1), (r2, c2))
        
        # 如果索引无效，返回一个默认的翻开动作
        return ("reveal", (0, 0), None)
    
    def _get_valid_action_indices(self, board: Board) -> List[int]:
        """获取有效动作索引列表"""
        valid_indices = []
        valid_actions = board.get_all_possible_moves(self.player_id)
        
        for action in valid_actions:
            index = self._action_to_index(action)
            valid_indices.append(index)
        
        return valid_indices
    
    def _validate_action(self, action: Tuple) -> bool:
        """验证动作格式是否正确"""
        if not isinstance(action, tuple) or len(action) != 3:
            return False
        
        action_type, pos1, pos2 = action
        
        if action_type not in ["reveal", "move"]:
            return False
        
        if not isinstance(pos1, tuple) or len(pos1) != 2:
            return False
        
        if action_type == "move" and (not isinstance(pos2, tuple) or len(pos2) != 2):
            return False
        
        return True
    
    def _normalize_action(self, action: Tuple) -> Tuple:
        """将动作标准化为3元组格式"""
        if len(action) == 2:
            action_type, pos1 = action
            if action_type == "reveal":
                return (action_type, pos1, None)
            else:
                # 这是错误的2元组move动作，返回默认
                return ("reveal", (0, 0), None)
        elif len(action) == 3:
            return action
        else:
            return ("reveal", (0, 0), None)

    def choose_action(self, board: Board, valid_actions: List[Tuple]) -> Tuple:
        """修复版本 - 标准化动作格式"""
        if not valid_actions:
            all_actions = board.get_all_possible_moves(self.player_id)
            if not all_actions:
                return ("reveal", (0, 0), None)
            valid_actions = all_actions
        
        # 标准化所有动作为3元组格式
        normalized_actions = [self._normalize_action(action) for action in valid_actions]
        
        # 验证标准化后的动作
        validated_actions = [action for action in normalized_actions if self._validate_action(action)]
        
        if not validated_actions:
            print(f"错误: 无法验证任何动作")
            return ("reveal", (0, 0), None)
        
        if random.random() < self.epsilon:
            return random.choice(validated_actions)
        
        state = self._board_to_state(board)
        valid_indices = [self._action_to_index(action) for action in validated_actions]
        
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            
            # 创建掩码，只考虑有效动作
            masked_q_values = torch.full((self.action_size,), float('-inf')).to(device)
            for idx in valid_indices:
                if 0 <= idx < self.action_size:  # 确保索引有效
                    masked_q_values[idx] = q_values[0][idx]
            
            best_index = masked_q_values.argmax().item()
            best_action = self._index_to_action(best_index)
            
            # 确保返回的动作在有效动作列表中
            if best_action and self._validate_action(best_action) and best_action in validated_actions:
                return best_action
            else:
                return random.choice(validated_actions)
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def normalize_reward(self, reward: float) -> float:
        """奖励归一化，避免训练不稳定"""
        self.reward_buffer.append(reward)
        
        if len(self.reward_buffer) < 100:
            return np.tanh(reward / 10.0)  # 早期简单归一化
        
        # 基于历史奖励的动态归一化
        mean_reward = np.mean(self.reward_buffer)
        std_reward = np.std(self.reward_buffer) + 1e-6
        
        normalized = (reward - mean_reward) / std_reward
        return np.tanh(normalized)  # 限制在[-1,1]
    
    def store_experience(self, board_before: Board, action: Tuple, reward: float, 
                        board_after: Board, result: int):
        """存储经验时进行奖励归一化"""
        normalized_reward = self.normalize_reward(reward)
        
        state = self._board_to_state(board_before)
        next_state = self._board_to_state(board_after)
        action_index = self._action_to_index(action)
        
        self.memory.push(state, action_index, normalized_reward, next_state, result)
        
    def replay(self):
        """改进的经验回放学习"""
        if len(self.memory) < self.batch_size:
            return
        
        # 采样经验
        experiences, indices, weights = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # 转换为张量
        state_batch = torch.stack(batch.state)
        action_batch = torch.LongTensor(batch.action).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        next_state_batch = torch.stack(batch.next_state)
        result_batch = torch.LongTensor(batch.result).to(device)
        
        # 计算当前Q值
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            if self.use_double:
                # Double DQN: 使用主网络选择动作，目标网络评估价值
                next_actions = self.q_network(next_state_batch).argmax(dim=1)
                next_q_values = self.target_network(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                # 标准 DQN
                next_q_values = self.target_network(next_state_batch).max(1)[0]
            
            # 使用更小的折扣因子，更注重即时奖励
            gamma = 0.95
            target_q_values = reward_batch + (gamma * next_q_values * (result_batch == -1).float())
        
        # 计算TD误差
        td_errors = (target_q_values - current_q_values.squeeze()).detach()
        
        # 更新优先级
        self.memory.update_priorities(indices, td_errors.abs().cpu().numpy())
        
        # 计算Huber损失（对异常值更鲁棒）
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values, reduction='none')
        weighted_loss = (loss * weights).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.5)
        
        self.optimizer.step()
        # 移除这里的调度器调用，改为在episode级别控制
        # self.base_scheduler.step()
        
        # 记录损失
        self.losses.append(weighted_loss.item())
        
    def get_learning_rate(self) -> float:
        """获取当前学习率"""
        # 从优化器中获取实际的学习率
        return self.optimizer.param_groups[0]['lr']
    
    def update_learning_rate(self, win_rate: float = None):
        """混合学习率调整策略 - 修复频率控制"""
        if not hasattr(self, 'training_stats'):
            return
            
        episodes = self.training_stats['episodes']
        current_lr = self.get_learning_rate()
        
        # 每隔一定episode数进行自适应调整
        if self.adaptive_lr and episodes % self.lr_adjustment_frequency == 0 and win_rate is not None:
            
            # 基于表现的自适应调整
            if win_rate < 0.25:
                # 表现很差，显著提高学习率
                multiplier = 1.2
                print(f"表现较差 (胜率={win_rate:.3f})，提高学习率")
                
            elif win_rate < 0.4:
                # 表现不佳，适度提高学习率
                multiplier = 1.1
                print(f"表现一般 (胜率={win_rate:.3f})，轻微提高学习率")
                
            elif win_rate > 0.8:
                # 表现很好，降低学习率以稳定策略
                multiplier = 0.9
                print(f"表现优秀 (胜率={win_rate:.3f})，降低学习率以稳定")
                
            elif win_rate > 0.6:
                # 表现良好，轻微降低学习率
                multiplier = 0.95
                print(f"表现良好 (胜率={win_rate:.3f})，轻微降低学习率")
                
            else:
                # 表现适中，保持当前学习率
                multiplier = 1.0
            
            # 计算新学习率
            new_lr = current_lr * multiplier
            
            # 限制学习率范围
            new_lr = max(self.lr_min, min(new_lr, self.lr_max))
            
            # 更新学习率
            if abs(new_lr - current_lr) > 1e-7:  # 只有变化足够大才更新
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"学习率调整: {current_lr:.6f} -> {new_lr:.6f}")
        
        # 基础调度器提供兜底的衰减 - 降低频率
        elif episodes % 200 == 0:  # 每200个episode执行基础衰减（原来是100）
            old_lr = current_lr
            self.base_scheduler.step()
            new_lr = self.get_learning_rate()
            
            # 防止学习率过低
            if new_lr < self.lr_min:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_min
                print(f"学习率限制在最小值: {self.lr_min:.6f}")
            elif abs(new_lr - old_lr) > 1e-7:
                print(f"基础调度器调整学习率: {old_lr:.6f} -> {new_lr:.6f}")
    
    def disable_adaptive_lr(self):
        """禁用自适应学习率，只使用基础调度器"""
        self.adaptive_lr = False
        print("已禁用自适应学习率调整")
    
    def enable_adaptive_lr(self):
        """启用自适应学习率"""
        self.adaptive_lr = True
        print("已启用自适应学习率调整")
    
    def decay_epsilon(self, win_rate: float = None):
        """改进的epsilon衰减"""
        if win_rate is not None:
            if win_rate < 0.3:
                decay = 0.999  # 胜率低时减慢衰减
            elif win_rate > 0.7:
                decay = 0.95   # 胜率高时加快衰减
            else:
                decay = self.epsilon_decay
        else:
            # 自适应衰减
            if self.episode_count < 100:
                decay = 0.999  # 早期慢衰减
            elif self.episode_count < 500:
                decay = 0.995  # 中期正常衰减
            else:
                decay = 0.99   # 后期快衰减
            
        self.epsilon = max(self.epsilon_min, self.epsilon * decay)
        self.episode_count += 1
        
    def set_training_mode(self, training: bool = True):
        """设置训练/测试模式"""
        self.training_mode = training
        if not training:
            # 测试模式：保存当前epsilon，设置为0
            self._saved_epsilon = self.epsilon
            self.epsilon = self.test_mode_epsilon
            print(f"切换到测试模式，epsilon: {self.epsilon}")
        else:
            # 训练模式：恢复保存的epsilon
            if hasattr(self, '_saved_epsilon'):
                self.epsilon = self._saved_epsilon
                print(f"切换到训练模式，epsilon: {self.epsilon}")
    
    def take_turn(self, board: Board) -> bool:
        """游戏回合方法 - 测试模式下不进行学习"""
        valid_actions = board.get_all_possible_moves(self.player_id)
        
        if not valid_actions:
            return False
        
        # 在测试模式下，强制epsilon=0（纯贪心策略）
        original_epsilon = self.epsilon
        if not self.training_mode:
            self.epsilon = 0.0
        
        action = self.choose_action(board, valid_actions)
        
        # 恢复原始epsilon
        if not self.training_mode:
            self.epsilon = original_epsilon
        
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
    
    def save_model(self, filename: str):
        """保存模型 - 只保存模型相关数据"""
        model_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.base_scheduler.state_dict(),  # 新增
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'hyperparameters': {  # 新增：保存超参数
                'learning_rate': self.learning_rate,
                'initial_learning_rate': self.initial_learning_rate,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'batch_size': self.batch_size,
                'use_dueling': self.use_dueling,
                'use_double': self.use_double,
                'lr_min': self.lr_min,
                'lr_max': self.lr_max
            },
            'model_info': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'ai_type': self.ai_type
            }
        }
        save_model_data(model_data, f"{filename}.pkl")
        print(f"模型已保存到: {filename}.pkl")
    
    def load_model(self, filename: str) -> bool:
        """加载模型"""
        data = load_model_data(f"{filename}.pkl")
        if data:
            try:
                self.q_network.load_state_dict(data['q_network_state_dict'])
                self.target_network.load_state_dict(data['target_network_state_dict'])
                self.optimizer.load_state_dict(data['optimizer_state_dict'])
                
                # 加载调度器状态（如果存在）
                if 'scheduler_state_dict' in data:
                    self.base_scheduler.load_state_dict(data['scheduler_state_dict'])
                
                self.epsilon = data.get('epsilon', self.epsilon)
                self.episode_count = data.get('episode_count', 0)
                
                # 加载超参数（如果存在）
                if 'hyperparameters' in data:
                    hp = data['hyperparameters']
                    self.learning_rate = hp.get('learning_rate', self.learning_rate)
                    self.initial_learning_rate = hp.get('initial_learning_rate', self.initial_learning_rate)
                    self.epsilon_decay = hp.get('epsilon_decay', self.epsilon_decay)
                    self.epsilon_min = hp.get('epsilon_min', self.epsilon_min)
                
                print(f"模型加载成功: {filename}.pkl")
                return True
            except Exception as e:
                print(f"模型加载失败: {e}")
                return False
        return False

class DQNTrainer(BaseTrainer):
    """DQN训练器"""
    
    def __init__(self, agent: DQNAgent, opponent_agent: Player, **kwargs):
        super().__init__(agent, opponent_agent, **kwargs)
        self.target_update_freq = 100
        
    def _agent_choose_action(self, board: Board, valid_actions: List[Tuple]) -> Tuple:
        """智能体选择动作"""
        return self.agent.choose_action(board, valid_actions)
    
    def _agent_update(self, board_before: Board, action: Tuple, reward: float, 
                     board_after: Board, result: int):
        """更新智能体"""
        # 存储经验
        self.agent.store_experience(board_before, action, reward, board_after, result)
        
        # 学习
        self.agent.replay()
        
        # 更新目标网络
        episodes = self.agent.training_stats['episodes']
        if episodes % self.target_update_freq == 0:
            self.agent.update_target_network()
    
    def save_model(self, filename: str):
        """保存模型"""
        self.agent.save_model(filename)



# def train_or_load_model(force_retrain=False, episodes=2000, lr_strategy="hybrid", print_interval=50):
#     """训练或加载改进的DQN模型 - 使用独立数据管理"""
#     from AgentFight import RandomPlayer
#     from training_data_manager import TrainingDataManager
#     import os
    
#     model_name = "final_D3QNAgent"
    
#     # 创建改进的智能体
#     dqn_agent = DQNAgent(
#         player_id=0, 
#         learning_rate=5e-4,
#         epsilon=0.9,
#         epsilon_min=0.05,
#         epsilon_decay=0.995,
#         batch_size=128,
#         memory_size=50000,
#         use_dueling=True,
#         use_double=True
#     )
#     random_opponent = RandomPlayer(player_id=1)
    
#     # 设置学习率策略
#     if lr_strategy == "adaptive":
#         dqn_agent.enable_adaptive_lr()
#         print("使用自适应学习率策略")
#     elif lr_strategy == "fixed":
#         dqn_agent.disable_adaptive_lr()
#         print("使用固定衰减学习率策略")
#     else:  # hybrid
#         print("使用混合学习率策略")
    
#     # 检查是否存在已训练的模型
#     model_path = os.path.join("model_data", f"{model_name}.pkl")
#     model_exists = os.path.exists(model_path)
    
#     if model_exists and not force_retrain:
#         print(f"发现已训练的模型: {model_path}")
#         if dqn_agent.load_model(model_name):
#             print("模型加载成功!")
#             dqn_agent.epsilon = 0.0
#             dqn_agent.ai_type = "D3QN (Trained)"
#         else:
#             print("模型加载失败，将重新训练...")
#             model_exists = False
    
#     if not model_exists or force_retrain:
#         if force_retrain:
#             print("强制重新训练模型...")
#         else:
#             print("未找到已训练模型，开始训练...")
        
#         # 创建数据管理器
#         data_manager = TrainingDataManager()
#         session_name = f"D3QN_{lr_strategy}_{episodes}eps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         data_manager.start_session(dqn_agent, session_name)
        
#         trainer = DQNTrainer(dqn_agent, random_opponent)
        
#         # 使用课程式训练，传入打印间隔
#         combined_history = train_with_curriculum(dqn_agent, random_opponent, episodes, data_manager, print_interval)
        
#         # 结束数据记录会话
#         final_stats = {
#             'training_episodes': episodes,
#             'lr_strategy': lr_strategy,
#             'final_epsilon': dqn_agent.epsilon,
#             'final_learning_rate': dqn_agent.get_learning_rate()
#         }
#         data_manager.end_session(dqn_agent, final_stats)
        
#         # 绘制训练历史
#         data_manager.plot_training_history()
        
#         # 保存模型（不包含训练数据）
#         dqn_agent.save_model(model_name)
        
#         print(f"训练完成! 最终epsilon: {dqn_agent.epsilon:.3f}")
#         print(f"最终胜率: {dqn_agent.get_stats()['win_rate']:.3f}")
        
#         # 设置为测试模式
#         dqn_agent.epsilon = 0.0
#         dqn_agent.ai_type = "D3QN (Trained)"
    
#     return dqn_agent, random_opponent

# def train_with_curriculum(dqn_agent, opponent, episodes=2000, data_manager=None, print_interval=50):
#     """课程式训练 - 带数据记录版本，改进学习率控制和进度输出
    
#     Args:
#         dqn_agent: DQN智能体
#         opponent: 对手
#         episodes: 总训练回合数
#         data_manager: 数据管理器
#         print_interval: 打印进度的间隔（默认每50回合）
#     """
#     trainer = DQNTrainer(dqn_agent, opponent)
    
#     total_episodes = episodes
    
#     # 添加计时记录
#     training_start_time = time.time()
#     phase_times = {}  # 记录各阶段耗时
    
#     # 阶段1: 快速探索 (30%)
#     phase1_episodes = int(total_episodes * 0.3)
#     print(f"阶段1: 快速探索学习 ({phase1_episodes} episodes) - 每{print_interval}回合输出进度")
#     print(f"epsilon固定在: {dqn_agent.epsilon:.3f}")
#     print(f"初始学习率: {dqn_agent.get_learning_rate():.6f}")
    
#     # 保存原始epsilon衰减设置并禁用
#     original_epsilon_decay = dqn_agent.epsilon_decay
#     dqn_agent.epsilon_decay = 1.0  # 禁用epsilon衰减
    
#     trainer.target_update_freq = 30
    
#     # 阶段1训练开始计时
#     phase1_start_time = time.time()
    
#     # 用于计算批次统计的变量
#     batch_wins = 0
#     batch_loses = 0
#     batch_draws = 0
#     batch_steps = []
#     batch_start_episode = 0
    
#     for episode in range(phase1_episodes):
#         episode_start_time = time.time()
#         total_reward, steps, result = trainer.train_episode()
#         episode_end_time = time.time()
#         episode_time = episode_end_time - episode_start_time
        
#         # 记录批次数据
#         batch_steps.append(steps)
#         if result == 0:  # 智能体胜利
#             batch_wins += 1
#         elif result == 1:  # 对手胜利
#             batch_loses += 1
#         else:  # 平局
#             batch_draws += 1        
        
#         # 定期输出进度
#         if episode % print_interval == 0 or episode == phase1_episodes - 1:
#             # 计算这个批次的统计
#             batch_episodes = episode - batch_start_episode + 1
#             batch_win_rate = (batch_wins + batch_draws / 2) / batch_episodes
#             avg_steps = sum(batch_steps) / len(batch_steps) if batch_steps else 0
            
#             # 计算这批次的耗时
#             current_time = time.time()
#             if episode == 0:
#                 batch_time = current_time - phase1_start_time
#             else:
#                 # 计算从上次输出到现在的时间
#                 episodes_since_start = episode + 1
#                 total_elapsed = current_time - phase1_start_time
#                 if episode >= print_interval:
#                     # 估算这个批次的时间
#                     avg_time_per_episode = total_elapsed / episodes_since_start
#                     batch_time = print_interval * avg_time_per_episode
#                 else:
#                     batch_time = total_elapsed
            
#             avg_time_per_episode = batch_time / batch_episodes if batch_episodes > 0 else 0
            
#             param_info = f", ε = {dqn_agent.epsilon:.3f}, lr = {dqn_agent.get_learning_rate():.6f}"
#             time_info = f", 用时 = {batch_time:.1f}s, 平均 = {avg_time_per_episode:.2f}s/ep"
#             print(f"阶段1 - 回合 {episode}: 奖励 = {total_reward:.2f}, 步数 = {steps}, "
#                   f"胜 = {batch_wins}, 负 = {batch_loses}, 平 = {batch_draws}, "
#                   f"批次胜率 = {batch_win_rate:.3f}, 平均步长 = {avg_steps:.1f}{param_info}{time_info}")
            
#             # 记录批次统计到数据管理器
#             if data_manager:
#                 data_manager.log_batch_stats(
#                     batch_wins, batch_loses, batch_draws, 
#                     batch_win_rate, avg_steps
#                 )
            
#             # 重置批次统计
#             batch_wins = 0
#             batch_loses = 0
#             batch_draws = 0
#             batch_steps = []
#             batch_start_episode = episode + 1
        
#         # 在阶段1，每100个episode检查一次学习率调整
#         if episode > 0 and episode % 100 == 0:
#             # 计算最近100个episode的胜率用于学习率调整
#             recent_wins = 0
#             recent_episodes = min(100, episode + 1)
#             for i in range(max(0, episode - recent_episodes + 1), episode + 1):
#                 if (data_manager and 
#                     len(data_manager.current_session['training_history']['wins']) > i and 
#                     data_manager.current_session['training_history']['wins'][i] == 1):
#                     recent_wins += 1
#             recent_win_rate = recent_wins / recent_episodes
            
#             dqn_agent.update_learning_rate(recent_win_rate)
        
#         # 记录每个episode的详细数据
#         if data_manager:
#             data_manager.log_episode(
#                 episode=episode,
#                 reward=total_reward,
#                 result=result,
#                 learning_rate=dqn_agent.get_learning_rate(),
#                 epsilon=dqn_agent.epsilon,
#                 loss=dqn_agent.losses[-1] if dqn_agent.losses else None,
#                 phase='phase1_exploration',
#                 steps=steps,
#                 episode_time=episode_time
#             )
    
#     phase1_end_time = time.time()
#     phase_times['phase1'] = phase1_end_time - phase1_start_time
#     print(f"阶段1完成! 当前epsilon: {dqn_agent.epsilon:.3f}, 学习率: {dqn_agent.get_learning_rate():.6f}")
#     print(f"阶段1总耗时: {phase_times['phase1']:.1f}秒, 平均: {phase_times['phase1']/phase1_episodes:.2f}秒/回合")
    
#     # 阶段2: 平衡学习 (50%)
#     phase2_episodes = int(total_episodes * 0.5)
#     print(f"\n阶段2: 平衡学习 ({phase2_episodes} episodes)")
#     print(f"启用缓慢epsilon衰减，当前epsilon: {dqn_agent.epsilon:.3f}")
#     print(f"当前学习率: {dqn_agent.get_learning_rate():.6f}")
    
#     # 启用缓慢epsilon衰减
#     dqn_agent.epsilon_decay = 0.999
#     trainer.target_update_freq = 50
    
#     # 阶段2开始计时
#     phase2_start_time = time.time()
    
#     # 重置批次统计
#     batch_wins = 0
#     batch_loses = 0
#     batch_draws = 0
#     batch_steps = []
#     batch_start_episode = 0
    
#     for episode in range(phase2_episodes):
#         episode_start_time = time.time()
#         total_reward, steps, result = trainer.train_episode()
#         episode_end_time = time.time()
#         episode_time = episode_end_time - episode_start_time
        
#         # 在阶段2调用epsilon衰减
#         dqn_agent.decay_epsilon()
        
#         # 记录批次数据
#         batch_steps.append(steps)
#         if result == 0:  # 智能体胜利
#             batch_wins += 1
#         elif result == 1:  # 对手胜利
#             batch_loses += 1
#         else:  # 平局
#             batch_draws += 1 
        
#         # 定期输出进度
#         if episode % print_interval == 0 or episode == phase2_episodes - 1:
#             # 计算这个批次的统计
#             batch_episodes = episode - batch_start_episode + 1
#             batch_win_rate = (batch_wins + batch_draws / 2) / batch_episodes
#             avg_steps = sum(batch_steps) / len(batch_steps) if batch_steps else 0
            
#             # 计算耗时
#             current_time = time.time()
#             phase2_elapsed = current_time - phase2_start_time
#             if episode == 0:
#                 batch_time = phase2_elapsed
#             else:
#                 avg_time_per_episode = phase2_elapsed / (episode + 1)
#                 batch_time = batch_episodes * avg_time_per_episode
            
#             avg_time_per_episode = batch_time / batch_episodes if batch_episodes > 0 else 0
            
#             param_info = f", ε = {dqn_agent.epsilon:.3f}, lr = {dqn_agent.get_learning_rate():.6f}"
#             time_info = f", 累计用时 = {phase2_elapsed:.1f}s, 平均 = {avg_time_per_episode:.2f}s/ep"
#             print(f"阶段2 - 回合 {episode}: 奖励 = {total_reward:.2f}, 步数 = {steps}, "
#                     f"胜 = {batch_wins}, 负 = {batch_loses}, 平 = {batch_draws}, "
#                   f"批次胜率 = {batch_win_rate:.3f}, 平均步长 = {avg_steps:.1f}{param_info}{time_info}")
            
#             # 记录批次统计到数据管理器
#             if data_manager:
#                 data_manager.log_batch_stats(
#                     batch_wins, batch_loses, batch_draws, 
#                     batch_win_rate, avg_steps
#                 )
            
#             # 重置批次统计
#             batch_wins = 0
#             batch_loses = 0
#             batch_draws = 0
#             batch_steps = []
#             batch_start_episode = episode + 1
        
#         # 阶段2学习率调整频率降低
#         if episode > 0 and episode % 150 == 0:
#             # 计算最近150个episode的胜率用于学习率调整
#             recent_wins = 0
#             recent_episodes = min(150, episode + 1)
#             for i in range(max(0, phase1_episodes + episode - recent_episodes + 1), phase1_episodes + episode + 1):
#                 if (data_manager and 
#                     len(data_manager.current_session['training_history']['wins']) > i and 
#                     data_manager.current_session['training_history']['wins'][i] == 1):
#                     recent_wins += 1
#             recent_win_rate = recent_wins / recent_episodes
            
#             dqn_agent.update_learning_rate(recent_win_rate)
        
#         # 记录每个episode的详细数据
#         if data_manager:
#             data_manager.log_episode(
#                 episode=phase1_episodes + episode,
#                 reward=total_reward,
#                 result=result,
#                 learning_rate=dqn_agent.get_learning_rate(),
#                 epsilon=dqn_agent.epsilon,
#                 loss=dqn_agent.losses[-1] if dqn_agent.losses else None,
#                 phase='phase2_balance',
#                 steps=steps,
#                 episode_time=episode_time
#             )
    
#     phase2_end_time = time.time()
#     phase_times['phase2'] = phase2_end_time - phase2_start_time
#     print(f"阶段2完成! 当前epsilon: {dqn_agent.epsilon:.3f}, 学习率: {dqn_agent.get_learning_rate():.6f}")
#     print(f"阶段2总耗时: {phase_times['phase2']:.1f}秒, 平均: {phase_times['phase2']/phase2_episodes:.2f}秒/回合")
    
#     # 阶段3: 策略精炼 (20%)
#     phase3_episodes = total_episodes - phase1_episodes - phase2_episodes
#     print(f"\n阶段3: 策略精炼 ({phase3_episodes} episodes)")
#     print(f"恢复正常epsilon衰减，当前epsilon: {dqn_agent.epsilon:.3f}")
#     print(f"当前学习率: {dqn_agent.get_learning_rate():.6f}")
    
#     # 恢复正常epsilon衰减
#     dqn_agent.epsilon_decay = original_epsilon_decay
#     trainer.target_update_freq = 100
    
#     # 阶段3开始计时
#     phase3_start_time = time.time()
    
#     # 重置批次统计
#     batch_wins = 0
#     batch_loses = 0
#     batch_draws = 0
#     batch_steps = []
#     batch_start_episode = 0
    
#     for episode in range(phase3_episodes):
#         episode_start_time = time.time()
#         total_reward, steps, result = trainer.train_episode()
#         episode_end_time = time.time()
#         episode_time = episode_end_time - episode_start_time
        
#         # 在阶段3调用epsilon衰减
#         dqn_agent.decay_epsilon()
        
#         # 记录批次数据
#         batch_steps.append(steps)
#         if result == 0:  # 智能体胜利
#             batch_wins += 1
#         elif result == 1:  # 对手胜利
#             batch_loses += 1
#         else:  # 平局
#             batch_draws += 1 
        
#         # 定期输出进度
#         if episode % print_interval == 0 or episode == phase3_episodes - 1:
#             # 计算这个批次的统计
#             batch_episodes = episode - batch_start_episode + 1
#             batch_win_rate = (batch_wins + batch_draws / 2) / batch_episodes
#             avg_steps = sum(batch_steps) / len(batch_steps) if batch_steps else 0
            
#             # 计算耗时
#             current_time = time.time()
#             phase3_elapsed = current_time - phase3_start_time
#             if episode == 0:
#                 batch_time = phase3_elapsed
#             else:
#                 avg_time_per_episode = phase3_elapsed / (episode + 1)
#                 batch_time = batch_episodes * avg_time_per_episode
            
#             avg_time_per_episode = batch_time / batch_episodes if batch_episodes > 0 else 0
            
#             param_info = f", ε = {dqn_agent.epsilon:.3f}, lr = {dqn_agent.get_learning_rate():.6f}"
#             time_info = f", 累计用时 = {phase3_elapsed:.1f}s, 平均 = {avg_time_per_episode:.2f}s/ep"
#             print(f"阶段3 - 回合 {episode}: 奖励 = {total_reward:.2f}, 步数 = {steps}, "
#                   f"胜 = {batch_wins}, 负 = {batch_loses}, 平 = {batch_draws}, "
#                   f"批次胜率 = {batch_win_rate:.3f}, 平均步长 = {avg_steps:.1f}{param_info}{time_info}")
            
#             # 记录批次统计到数据管理器
#             if data_manager:
#                 data_manager.log_batch_stats(
#                     batch_wins, batch_loses, batch_draws, 
#                     batch_win_rate, avg_steps
#                 )
            
#             # 重置批次统计
#             batch_wins = 0
#             batch_loses = 0
#             batch_draws = 0
#             batch_steps = []
#             batch_start_episode = episode + 1
        
#         # 阶段3减少学习率调整，更关注稳定性
#         if episode > 0 and episode % 200 == 0:
#             # 计算最近200个episode的胜率用于学习率调整
#             recent_wins = 0
#             recent_episodes = min(200, episode + 1)
#             for i in range(max(0, phase1_episodes + phase2_episodes + episode - recent_episodes + 1), 
#                           phase1_episodes + phase2_episodes + episode + 1):
#                 if (data_manager and 
#                     len(data_manager.current_session['training_history']['wins']) > i and 
#                     data_manager.current_session['training_history']['wins'][i] == 1):
#                     recent_wins += 1
#             recent_win_rate = recent_wins / recent_episodes
            
#             dqn_agent.update_learning_rate(recent_win_rate)
        
#         # 记录每个episode的详细数据
#         if data_manager:
#             data_manager.log_episode(
#                 episode=phase1_episodes + phase2_episodes + episode,
#                 reward=total_reward,
#                 result=result,
#                 learning_rate=dqn_agent.get_learning_rate(),
#                 epsilon=dqn_agent.epsilon,
#                 loss=dqn_agent.losses[-1] if dqn_agent.losses else None,
#                 phase='phase3_refinement',
#                 steps=steps,
#                 episode_time=episode_time
#             )
    
#     phase3_end_time = time.time()
#     phase_times['phase3'] = phase3_end_time - phase3_start_time
    
#     # 总结训练时间
#     total_training_time = phase3_end_time - training_start_time
    
#     print(f"\n课程训练完成!")
#     print(f"最终epsilon: {dqn_agent.epsilon:.3f}")
#     print(f"最终学习率: {dqn_agent.get_learning_rate():.6f}")
#     print(f"最终胜率: {dqn_agent.get_stats()['win_rate']:.3f}")
    
#     print(f"\n=== 训练耗时统计 ===")
#     print(f"阶段1 ({phase1_episodes} episodes): {phase_times['phase1']:.1f}秒 (平均 {phase_times['phase1']/phase1_episodes:.2f}秒/回合)")
#     print(f"阶段2 ({phase2_episodes} episodes): {phase_times['phase2']:.1f}秒 (平均 {phase_times['phase2']/phase2_episodes:.2f}秒/回合)")
#     print(f"阶段3 ({phase3_episodes} episodes): {phase_times['phase3']:.1f}秒 (平均 {phase_times['phase3']/phase3_episodes:.2f}秒/回合)")
#     print(f"总训练时间: {total_training_time:.1f}秒 ({total_training_time/60:.1f}分钟)")
#     print(f"总体平均: {total_training_time/total_episodes:.2f}秒/回合")
    
#     # 在训练结束后打印详细统计
#     if data_manager:
#         summary = data_manager.get_summary_stats()
#         print(f"\n=== 详细训练统计 ===")
#         if 'average_steps' in summary:
#             print(f"平均步数: {summary['average_steps']:.1f} ± {summary['steps_std']:.1f}")
#             print(f"步数范围: {summary['min_steps']} - {summary['max_steps']}")
#         if 'average_episode_time' in summary:
#             print(f"平均每回合时间: {summary['average_episode_time']:.3f} ± {summary['episode_time_std']:.3f} 秒")
#         print(f"总胜率: {summary['final_win_rate']:.3f}")
#         print(f"平均奖励: {summary['average_reward']:.3f} ± {summary['reward_std']:.3f}")
    
#     return data_manager.current_session['training_history'] if data_manager else {}

# # 使用示例
# if __name__ == "__main__":
#     import argparse
#     from base import Game
    
#     parser = argparse.ArgumentParser(description='DQN Agent 训练和测试')
#     parser.add_argument('--retrain', action='store_true', help='强制重新训练模型')
#     parser.add_argument('--episodes', type=int, default=1000, help='训练回合数')
#     parser.add_argument('--test-games', type=int, default=1, help='测试游戏数量')
#     parser.add_argument('--no-display', action='store_true', help='不显示游戏界面')
#     parser.add_argument('--lr-strategy', choices=['adaptive', 'fixed', 'hybrid'], 
#                        default='hybrid', help='学习率调整策略')
#     parser.add_argument('--test-only', action='store_true', help='仅测试，不训练')
#     parser.add_argument('--print-interval', type=int, default=50, help='训练进度输出间隔')
    
#     args = parser.parse_args()
    
#     # 训练或加载模型
#     if not args.test_only:
#         dqn_agent, random_opponent = train_or_load_model(
#             force_retrain=args.retrain, 
#             episodes=args.episodes,
#             lr_strategy=args.lr_strategy,
#             print_interval=args.print_interval
#         )
#     else:
#         # 仅测试模式：直接加载模型
#         print("仅测试模式，加载已训练模型...")
#         dqn_agent = DQNAgent(player_id=0)
#         if not dqn_agent.load_model("final_D3QNAgent"):
#             print("错误：找不到训练好的模型！")
#             exit(1)
#         from AgentFight import RandomPlayer
#         random_opponent = RandomPlayer(player_id=1)
    
#     # 设置为测试模式
#     dqn_agent.set_training_mode(False)
    
#     # 测试
#     print(f"\n开始测试 {args.test_games} 场游戏...")
#     print(f"测试模式 - epsilon: {dqn_agent.epsilon}")
    
#     wins = 0
#     draws = 0
    
#     for i in range(args.test_games):
#         game = Game(dqn_agent, random_opponent, display=not args.no_display, delay=0.5)
#         result = game.run()
        
#         if result == 0:  # DQN agent wins
#             wins += 1
#         elif result == 2:  # Draw
#             draws += 1
            
#         if args.test_games <= 10:  # 只在测试游戏数较少时显示详细结果
#             result_text = '胜利' if result == 0 else '失败' if result == 1 else '平局'
#             print(f"游戏 {i+1}: {result_text}")
    
#     win_rate = wins / args.test_games
#     draw_rate = draws / args.test_games
    
#     print(f"\n测试结果:")
#     print(f"胜利: {wins}/{args.test_games} ({win_rate:.3f})")
#     print(f"平局: {draws}/{args.test_games} ({draw_rate:.3f})")
#     print(f"失败: {args.test_games - wins - draws}/{args.test_games}")
#     print(f"有效胜率 (胜+0.5*平): {win_rate + 0.5 * draw_rate:.3f}")
    
# # python DQN.py --retrain --episodes 5000 --lr-strategy hybrid --test-games 200 --no-display --print-interval 100