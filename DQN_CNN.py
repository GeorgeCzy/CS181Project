import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
import pickle
import os
from datetime import datetime
from collections import deque, namedtuple
from typing import Tuple, List, Optional, Dict, Any
from base import Board, Player
from utils import save_model_data, load_model_data, AggressiveReward, PrioritizedReplayBuffer, Experience, device
from cnn_features import CNNEnhancedFeatureExtractor
from DQN import DQNAgent, DQNTrainer
import time

# # 检查GPU是否可用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")

class CNNEnhancedDQN(nn.Module):
    """CNN增强的DQN网络"""
    
    def __init__(self, cnn_feature_size: int = 64, traditional_feature_size: int = 32, 
                 action_size: int = 280, hidden_size: int = 256):
        super(CNNEnhancedDQN, self).__init__()
        
        # 特征融合层
        total_feature_size = cnn_feature_size + traditional_feature_size
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Dueling DQN架构
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
    
    def forward(self, cnn_features: torch.Tensor, traditional_features: torch.Tensor):
        # 融合CNN特征和传统特征
        combined_features = torch.cat([cnn_features, traditional_features], dim=1)
        features = self.feature_fusion(combined_features)
        
        # Dueling DQN
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class CNNEnhancedDQNAgent(DQNAgent):
    """CNN增强的DQN智能体 - 统一的分阶段超参数控制"""
    
    def __init__(self, player_id: int, action_size: int = 280,
                 learning_rate: float = 1e-2, epsilon: float = 0.95,
                 epsilon_decay: float = 0.9995, epsilon_min: float = 0.02,
                 batch_size: int = 32, memory_size: int = 30000,
                 exploration_strategy: str = "guided",
                 cnn_model_path: str = None):
        
        # 调用Player的初始化，跳过DQNAgent的初始化
        Player.__init__(self, player_id)
        
        # 重新设置所有必要的属性
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.lr_decay_rate = 0.9995
        self.lr_min = 1e-3
        self.lr_max = learning_rate * 5
        self.adaptive_lr = True
        
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.exploration_strategy = exploration_strategy
        self.reward_buffer = deque(maxlen=1000)
        
        self.training_mode = True
        self.test_mode_epsilon = 0.0
        
        # CNN专用的统一分阶段控制 - 移除向后兼容性
        self.phase_configs = {
            'phase1': {
                # Epsilon 控制
                'epsilon_force_until': 500,      # CNN需要更长的基础学习期
                'epsilon_min': 0.85,             # 保持更高的探索度
                'epsilon_decay_rate': 0.9999,    # 极慢衰减
                
                # Learning Rate 控制
                'lr_force_until': 400,           # CNN需要更长的学习率稳定期
                'lr_update_frequency': 50,      # 更频繁的更新
                'lr_stable_range': (0.9, 1.5),   # CNN训练需要更稳定的学习率
                'lr_adaptive_range': (0.7, 1.5), # 适度的自适应范围
                
                # CNN特有
                'cnn_weight': 0.1,               # 早期更依赖传统评估
                'description': 'CNN基础学习阶段'
            },
            'phase2': {
                # Epsilon 控制
                'epsilon_force_until': 400,      # 较长的进阶期
                'epsilon_min': 0.4,              # 中等探索度
                'epsilon_decay_rate': 0.9996,    # 慢衰减
                
                # Learning Rate 控制
                'lr_force_until': 250,           # 保持学习率稳定期
                'lr_update_frequency': 80,      # 降低更新频率
                'lr_stable_range': (0.9, 1.4), # 更稳定
                'lr_adaptive_range': (0.8, 1.5), # 保守的自适应
                
                # CNN特有
                'cnn_weight': 0.4,               # 中期平衡CNN和传统
                'description': 'CNN进阶学习阶段'
            },
            'phase3': {
                # Epsilon 控制
                'epsilon_force_until': 150,      # 精炼期保持一定探索
                'epsilon_min': self.epsilon_min, # 使用原始最小epsilon
                'epsilon_decay_rate': self.epsilon_decay,
                
                # Learning Rate 控制
                'lr_force_until': 80,            # 短的稳定期
                'lr_update_frequency': 80,      # 进一步降低频率
                'lr_stable_range': (0.9, 1.2), # 非常稳定
                'lr_adaptive_range': (0.9, 1.2), # 保守的自适应
                
                # CNN特有
                'cnn_weight': 0.6,               # 后期更依赖CNN
                'description': 'CNN策略精炼阶段'
            }
        }
        
        self.current_phase = 'phase1'
        # 添加阶段内episode计数器
        self.episode_in_phase = 0
        
        # CNN特征提取器
        self.cnn_extractor = CNNEnhancedFeatureExtractor(cnn_model_path)
        
        # 网络架构 - 使用CNN增强网络
        self.q_network = CNNEnhancedDQN().to(device)
        self.target_network = CNNEnhancedDQN().to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-6)
        
        # 经验回放 - 调整参数
        self.memory = PrioritizedReplayBuffer(memory_size, alpha=0.3, beta=0.3)
        
        # 奖励函数
        self.reward_function = AggressiveReward()
        
        # 更新目标网络
        self.update_target_network()
        
        # 训练统计
        self.losses = []
        self.episode_count = 0
        
        # 学习率调度器 - 更温和的衰减
        self.base_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1200, gamma=0.95
        )
        
        self.ai_type = f"CNN-DQN (ε={epsilon:.2f})" if epsilon > 0 else "CNN-DQN (Trained)"
        
        print("初始化 CNN增强 DQN - 统一分阶段超参数控制")
        for phase, config in self.phase_configs.items():
            print(f"  {phase} ({config['description']}):")
            print(f"    Epsilon: 强制{config['epsilon_force_until']}回合, 最小ε={config['epsilon_min']:.2f}")
            print(f"    LR: 强制{config['lr_force_until']}回合, 频率={config['lr_update_frequency']}")
            print(f"    CNN权重: {config['cnn_weight']:.1f}")

    
    def _extract_features(self, board: Board) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取CNN特征和传统特征"""
        # CNN特征
        cnn_features = self.cnn_extractor.extract_cnn_features(board, self.player_id)
        cnn_tensor = torch.FloatTensor(cnn_features).unsqueeze(0).to(device)
        
        # 传统特征（简化版，32维）
        traditional_features = []
        
        # 基础统计特征
        my_pieces = board.get_player_pieces(self.player_id)
        enemy_pieces = board.get_player_pieces(1 - self.player_id)
        
        traditional_features.extend([
            len(my_pieces) / 16.0,
            len(enemy_pieces) / 16.0,
        ])
        
        # 已翻开比例
        my_revealed = sum(1 for r, c in my_pieces if board.get_piece(r, c).revealed)
        enemy_revealed = sum(1 for r, c in enemy_pieces if board.get_piece(r, c).revealed)
        
        traditional_features.extend([
            my_revealed / max(len(my_pieces), 1),
            enemy_revealed / max(len(enemy_pieces), 1),
        ])
        
        # 价值统计
        piece_values = {1: 1.8, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5, 6: 3.0, 7: 3.5, 8: 4.0}
        my_total_value = sum(
            piece_values[board.get_piece(r, c).strength]
            for r, c in my_pieces if board.get_piece(r, c).revealed
        )
        enemy_total_value = sum(
            piece_values[board.get_piece(r, c).strength]
            for r, c in enemy_pieces if board.get_piece(r, c).revealed
        )
        
        total_value = my_total_value + enemy_total_value + 1e-6
        traditional_features.extend([
            my_total_value / total_value,
            enemy_total_value / total_value,
        ])
        
        # 威胁机会统计
        threat_count = 0
        opportunity_count = 0
        for r, c in my_pieces:
            piece = board.get_piece(r, c)
            if piece and piece.revealed:
                # 检查周围威胁和机会
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 7 and 0 <= nc < 8:
                        neighbor = board.get_piece(nr, nc)
                        if neighbor and neighbor.revealed and neighbor.player != self.player_id:
                            compare = piece.compare_strength(neighbor)
                            if compare == -1:  # 受威胁
                                threat_count += 1
                            elif compare == 1:  # 有机会
                                opportunity_count += 1
        
        traditional_features.extend([
            threat_count / max(len(my_pieces), 1),
            opportunity_count / max(len(my_pieces), 1),
        ])
        
        # 位置分布特征
        center_pieces = sum(1 for r, c in my_pieces if 2 <= r <= 4 and 2 <= c <= 5)
        edge_pieces = sum(1 for r, c in my_pieces if r in [0, 6] or c in [0, 7])
        
        traditional_features.extend([
            center_pieces / max(len(my_pieces), 1),
            edge_pieces / max(len(my_pieces), 1),
        ])
        
        # 填充到固定长度32维
        while len(traditional_features) < 32:
            traditional_features.append(0.0)
        
        traditional_tensor = torch.FloatTensor(traditional_features[:32]).unsqueeze(0).to(device)
        
        return cnn_tensor, traditional_tensor
    
    def _guided_exploration_with_cnn(self, board: Board, valid_actions: List[Tuple]) -> Tuple:
        """CNN引导探索 - 根据当前阶段动态调整权重"""
        if not valid_actions:
            return ("reveal", (0, 0), None)
        
        # 获取当前阶段配置
        config = self.phase_configs.get(self.current_phase, self.phase_configs['phase1'])
        cnn_weight = config['cnn_weight']
        traditional_weight = 1.0 - cnn_weight
        
        action_scores = []
        
        for action in valid_actions:
            # 传统评估
            try:
                traditional_score = self._evaluate_action_like_greedy(board, action)
            except:
                traditional_score = 0.0
            
            # CNN评估
            try:
                temp_board = copy.deepcopy(board)
                self._simulate_action(temp_board, action)
                cnn_score = self.cnn_extractor.evaluate_position(temp_board, self.player_id)
            except:
                cnn_score = 0.0
            
            # 使用阶段配置的权重
            combined_score = (traditional_weight * traditional_score + 
                            cnn_weight * cnn_score * 8)  # CNN分数适度放大
            
            # 根据阶段调整随机扰动
            force_until = config['epsilon_force_until']
            if self.episode_in_phase < force_until:
                # 强制探索期
                noise_level = 0.6
                selection_range = 0.7
            elif self.current_phase == 'phase1':
                noise_level = 0.5  # 高随机性
                selection_range = 0.6  # 从前60%中选择
            elif self.current_phase == 'phase2':
                noise_level = 0.3  # 中等随机性
                selection_range = 0.4  # 从前40%中选择
            else:  # phase3
                noise_level = 0.1  # 低随机性
                selection_range = 0.2  # 从前20%中选择
            
            combined_score += random.uniform(-noise_level, noise_level)
            action_scores.append((action, combined_score))
        
        # 按分数排序并选择顶部动作
        action_scores.sort(key=lambda x: x[1], reverse=True)
        top_count = max(1, int(len(action_scores) * selection_range))
        best_actions = action_scores[:top_count]
        
        # 加权随机选择
        actions, scores = zip(*best_actions)
        min_score = min(scores)
        adjusted_scores = [s - min_score + 1.0 for s in scores]
        total_score = sum(adjusted_scores)
        
        if total_score > 0:
            rand_val = random.random() * total_score
            cumulative = 0.0
            for i, weight in enumerate(adjusted_scores):
                cumulative += weight
                if rand_val <= cumulative:
                    return actions[i]
        
        return random.choice(valid_actions)
    
    def _simulate_action(self, board: Board, action: Tuple) -> bool:
        """在临时棋盘上模拟执行动作"""
        action_type, pos1, pos2 = action
        
        if action_type == "reveal":
            r, c = pos1
            piece = board.get_piece(r, c)
            if piece and piece.player == self.player_id and not piece.revealed:
                piece.revealed = True
                return True
        elif action_type == "move":
            return board.try_move(pos1, pos2)
        
        return False
    
    def choose_action(self, board: Board, valid_actions: List[Tuple]) -> Tuple:
        """改进的动作选择"""
        if not valid_actions:
            all_actions = board.get_all_possible_moves(self.player_id)
            if not all_actions:
                return ("reveal", (0, 0), None)
            valid_actions = all_actions
        
        # 标准化动作
        normalized_actions = [self._normalize_action(action) for action in valid_actions]
        validated_actions = [action for action in normalized_actions if self._validate_action(action)]
        
        if not validated_actions:
            return ("reveal", (0, 0), None)
        
        # epsilon-greedy with CNN-enhanced exploration
        if random.random() < self.epsilon:
            if self.exploration_strategy == "guided":
                return self._guided_exploration_with_cnn(board, validated_actions)
            else:
                return random.choice(validated_actions)
        
        # 利用策略
        cnn_features, traditional_features = self._extract_features(board)
        valid_indices = [self._action_to_index(action) for action in validated_actions]
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(cnn_features, traditional_features)
            
            # 创建掩码
            masked_q_values = torch.full((self.action_size,), float('-inf')).to(device)
            for idx in valid_indices:
                if 0 <= idx < self.action_size:
                    masked_q_values[idx] = q_values[0][idx]
            
            best_index = masked_q_values.argmax().item()
            best_action = self._index_to_action(best_index)
            
            if best_action and self._validate_action(best_action) and best_action in validated_actions:
                return best_action
            else:
                return random.choice(validated_actions)
        
        if self.training_mode:
            self.q_network.train()
    
    def store_experience(self, board_before: Board, action: Tuple, reward: float, 
                        board_after: Board, result: int):
        """存储经验"""
        normalized_reward = np.tanh(reward / 5.0)
        
        cnn_features_before, trad_features_before = self._extract_features(board_before)
        cnn_features_after, trad_features_after = self._extract_features(board_after)
        
        # 合并特征
        state_before = torch.cat([cnn_features_before, trad_features_before], dim=1)
        state_after = torch.cat([cnn_features_after, trad_features_after], dim=1)
        
        action_index = self._action_to_index(action)
        
        self.memory.push(state_before.squeeze(0), action_index, normalized_reward, 
                        state_after.squeeze(0), result)
    
    def replay(self):
        """经验回放学习"""
        if len(self.memory) < self.batch_size:
            return
        
        self.q_network.train()
        self.target_network.eval()
        
        experiences, indices, weights = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # 分离CNN和传统特征
        state_batch = torch.stack(batch.state)
        cnn_features = state_batch[:, :64]  # 前64维是CNN特征
        trad_features = state_batch[:, 64:]  # 后32维是传统特征
        
        action_batch = torch.LongTensor(batch.action).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        
        next_state_batch = torch.stack(batch.next_state)
        next_cnn_features = next_state_batch[:, :64]
        next_trad_features = next_state_batch[:, 64:]
        
        result_batch = torch.LongTensor(batch.result).to(device)
        
        # 限制奖励范围
        reward_batch = torch.clamp(reward_batch, -5.0, 5.0)
        
        # 计算当前Q值
        current_q_values = self.q_network(cnn_features, trad_features).gather(1, action_batch.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            # Double DQN
            self.q_network.eval()
            next_actions = self.q_network(next_cnn_features, next_trad_features).argmax(dim=1)
            self.q_network.train()
            
            next_q_values = self.target_network(next_cnn_features, next_trad_features).gather(1, next_actions.unsqueeze(1)).squeeze()
            
            gamma = 0.99
            target_q_values = reward_batch + (gamma * next_q_values * (result_batch == -1).float())
            target_q_values = torch.clamp(target_q_values, -20.0, 20.0)
        
        # 计算TD误差和损失
        td_errors = (target_q_values - current_q_values.squeeze()).detach()
        self.memory.update_priorities(indices, torch.clamp(td_errors.abs(), 0, 10).cpu().numpy())
        
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values, reduction='none')
        weighted_loss = (loss * weights).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(weighted_loss.item())
        
    def save_model(self, filename: str):
        """保存模型"""
        model_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.base_scheduler.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'cnn_features': True,  # 标记使用CNN特征
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'initial_learning_rate': self.initial_learning_rate,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'batch_size': self.batch_size,
                'lr_min': self.lr_min,
                'lr_max': self.lr_max
            },
            'model_info': {
                'action_size': self.action_size,
                'ai_type': self.ai_type
            }
        }
        save_model_data(model_data, f"{filename}.pkl")
        print(f"CNN-DQN模型已保存到: {filename}.pkl")
    
    def load_model(self, filename: str) -> bool:
        """加载模型"""
        data = load_model_data(f"{filename}.pkl")
        if data:
            try:
                self.q_network.load_state_dict(data['q_network_state_dict'])
                self.target_network.load_state_dict(data['target_network_state_dict'])
                self.optimizer.load_state_dict(data['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in data:
                    self.base_scheduler.load_state_dict(data['scheduler_state_dict'])
                
                self.epsilon = data.get('epsilon', self.epsilon)
                self.episode_count = data.get('episode_count', 0)
                
                # 加载超参数
                if 'hyperparameters' in data:
                    hyper = data['hyperparameters']
                    self.learning_rate = hyper.get('learning_rate', self.learning_rate)
                    self.initial_learning_rate = hyper.get('initial_learning_rate', self.initial_learning_rate)
                    self.epsilon_decay = hyper.get('epsilon_decay', self.epsilon_decay)
                    self.epsilon_min = hyper.get('epsilon_min', self.epsilon_min)
                    self.batch_size = hyper.get('batch_size', self.batch_size)
                    self.lr_min = hyper.get('lr_min', self.lr_min)
                    self.lr_max = hyper.get('lr_max', self.lr_max)
                
                print(f"CNN-DQN模型加载成功: {filename}.pkl")
                return True
            except Exception as e:
                print(f"CNN-DQN模型加载失败: {e}")
                return False
        return False

class CNNDQNTrainer(DQNTrainer):
    """CNN-DQN训练器"""
    
    def __init__(self, agent: CNNEnhancedDQNAgent, opponent_agent: Player, **kwargs):
        super().__init__(agent, opponent_agent, **kwargs)
        self.target_update_freq = 100
        print("使用CNN-DQN训练器")