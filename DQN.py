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
from base import Board, Player, BaseTrainer
from utils import (
    SimpleReward,
    save_model_data,
    load_model_data,
    AggressiveReward,
    PrioritizedReplayBuffer,
    Experience,
    device,
)
import time

# # 检查GPU是否可用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")


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
            nn.ReLU(),
        )

        # Dueling DQN: 价值流和优势流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )

    def forward(self, x):
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Dueling DQN 公式: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values


class DuelingDoubleDQN(nn.Module):
    """Dueling Double DQN 网络 - 结合两种技术，修复BatchNorm问题"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super(DuelingDoubleDQN, self).__init__()

        # 更深的共享特征提取层 - 移除BatchNorm避免单样本推理问题
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Dueling DQN: 价值流和优势流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, action_size),
        )
        
        # 改进权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """改进的权重初始化 - 增加输出多样性"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用He初始化增加初始方差
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    # 给偏置更大的随机初始化
                    nn.init.uniform_(module.bias, -0.3, 0.3)
    
        # 特别处理最后的输出层
        if hasattr(self, 'advantage_stream') and hasattr(self, 'value_stream'):
            # Dueling DQN的输出层
            for stream in [self.advantage_stream, self.value_stream]:
                if len(stream) > 0 and isinstance(stream[-1], nn.Linear):
                    # 输出层使用更大的初始化范围
                    nn.init.uniform_(stream[-1].weight, -0.1, 0.1)
                    nn.init.uniform_(stream[-1].bias, -1.0, 1.0)  # 更大的偏置范围

    def forward(self, x):
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Dueling DQN 公式: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values


class DoubleDQN(nn.Module):
    """Double DQN 网络 - 修复BatchNorm问题"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super(DoubleDQN, self).__init__()

        # 更深的网络 - 移除BatchNorm避免单样本推理问题
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent(Player):
    """改进的 DQN 智能体 - 统一的分阶段超参数控制"""

    def __init__(
        self,
        player_id: int,
        state_size: int = 448,
        action_size: int = 280,
        learning_rate: float = 1e-2,
        epsilon: float = 0.9,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 32,
        memory_size: int = 30000,
        use_dueling: bool = True,
        use_double: bool = True,
        exploration_strategy: str = "guided",
    ):

        super().__init__(player_id)
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.lr_decay_rate = 0.999
        self.lr_min = 1e-3
        self.lr_max = learning_rate * 5
        self.adaptive_lr = True

        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.use_dueling = use_dueling
        self.use_double = use_double
        self.exploration_strategy = exploration_strategy
        self.reward_buffer = deque(maxlen=1000)

        self.training_mode = True
        self.test_mode_epsilon = 0.0

        # 统一的分阶段超参数控制配置 - 移除向后兼容性
        self.phase_configs = {
            "phase1": {
                # Epsilon 控制
                "epsilon_force_until": 400,  # 前400回合强制保持高epsilon
                "epsilon_min": 0.7,  # 阶段1最小epsilon
                "epsilon_decay_rate": 0.9999,  # 阶段1极慢衰减
                # Learning Rate 控制
                "lr_force_until": 300,  # 前300回合学习率保持稳定
                "lr_update_frequency": 50,  # 学习率更新频率
                "lr_stable_range": (0.9, 1.2),  # 稳定期学习率调整范围
                "lr_adaptive_range": (0.8, 1.5),  # 自适应期学习率调整范围
                "description": "基础学习阶段",
            },
            "phase2": {
                # Epsilon 控制
                "epsilon_force_until": 300,  # 前300回合保持中等探索
                "epsilon_min": 0.3,  # 阶段2最小epsilon
                "epsilon_decay_rate": 0.9995,  # 阶段2慢衰减
                # Learning Rate 控制
                "lr_force_until": 200,  # 前200回合学习率保持稳定
                "lr_update_frequency": 50,  # 降低更新频率
                "lr_stable_range": (0.9, 1.2),  # 更稳定的调整范围
                "lr_adaptive_range": (0.7, 2.0),  # 更保守的自适应范围
                "description": "进阶学习阶段",
            },
            "phase3": {
                # Epsilon 控制
                "epsilon_force_until": 100,  # 前100回合保持基本探索
                "epsilon_min": self.epsilon_min,  # 使用原始最小epsilon
                "epsilon_decay_rate": self.epsilon_decay,
                # Learning Rate 控制
                "lr_force_until": 50,  # 前50回合学习率保持稳定
                "lr_update_frequency": 200,  # 进一步降低更新频率
                "lr_stable_range": (0.95, 1.05),  # 非常稳定的调整
                "lr_adaptive_range": (0.8, 1.2),  # 保守的自适应范围
                "description": "策略精炼阶段",
            },
        }

        self.current_phase = "phase1"
        # 添加阶段内episode计数器
        self.episode_in_phase = 0

        # 创建网络... (保持原有代码)
        if use_dueling and use_double:
            self.q_network = DuelingDoubleDQN(
                state_size, action_size, hidden_size=256
            ).to(device)
            self.target_network = DuelingDoubleDQN(
                state_size, action_size, hidden_size=256
            ).to(device)
            ai_type = "D3QN"
        elif use_dueling:
            self.q_network = DQN(state_size, action_size, hidden_size=256).to(device)
            self.target_network = DQN(state_size, action_size, hidden_size=256).to(
                device
            )
            ai_type = "DuelDQN"
        elif use_double:
            self.q_network = DoubleDQN(state_size, action_size, hidden_size=256).to(
                device
            )
            self.target_network = DoubleDQN(
                state_size, action_size, hidden_size=256
            ).to(device)
            ai_type = "DoubleDQN"
        else:
            self.q_network = DoubleDQN(state_size, action_size, hidden_size=256).to(
                device
            )
            self.target_network = DoubleDQN(
                state_size, action_size, hidden_size=256
            ).to(device)
            ai_type = "DQN"

        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        # 经验回放缓冲区
        self.memory = PrioritizedReplayBuffer(memory_size, alpha=0.4, beta=0.4)

        # 使用激进的奖励函数
        self.reward_function = AggressiveReward()

        # 更新目标网络
        self.update_target_network()

        # 训练统计
        self.losses = []
        self.episode_count = 0

        # 学习率调度器
        self.base_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=800, gamma=0.92
        )

        self.ai_type = (
            f"{ai_type} (ε={epsilon:.2f})" if epsilon > 0 else f"{ai_type} (Trained)"
        )

        print(f"初始化 {ai_type} - 统一分阶段超参数控制")
        for phase, config in self.phase_configs.items():
            print(f"  {phase} ({config['description']}):")
            print(
                f"    Epsilon: 强制{config['epsilon_force_until']}回合, 最小ε={config['epsilon_min']:.2f}"
            )
            print(
                f"    LR: 强制{config['lr_force_until']}回合, 频率={config['lr_update_frequency']}"
            )
        self._test_initial_network_output()
            
    
    def _test_initial_network_output(self):
        """测试网络初始输出，确保有合理的Q值范围"""
        self.q_network.eval()
        with torch.no_grad():
            # 创建多个随机状态进行测试
            test_states = torch.randn(20, self.state_size).to(device)  # 增加测试样本
            test_outputs = self.q_network(test_states)
            
            print(f"\n=== 网络初始化测试 ===")
            print(f"输出范围: [{test_outputs.min().item():.4f}, {test_outputs.max().item():.4f}]")
            print(f"输出均值: {test_outputs.mean().item():.4f}")
            print(f"输出标准差: {test_outputs.std().item():.4f}")
            print(f"不同状态输出的方差: {test_outputs.var(dim=0).mean().item():.4f}")
            print(f"不同动作输出的方差: {test_outputs.var(dim=1).mean().item():.4f}")
            
            # 检查输出多样性
            output_range = test_outputs.max().item() - test_outputs.min().item()
            print(f"输出值域宽度: {output_range:.4f}")
            
            # 如果输出方差太小或值域太窄，重新初始化
            if (test_outputs.std().item() < 0.5 or 
                output_range < 1.0 or 
                test_outputs.var(dim=1).mean().item() < 0.1):
                
                print("警告：网络输出多样性不足，重新初始化...")
                
                # 重新初始化最后几层
                for module_name, module in self.q_network.named_modules():
                    if isinstance(module, nn.Linear):
                        # 对所有线性层使用更激进的初始化
                        nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.uniform_(module.bias, -0.5, 0.5)
                
                # 特别处理输出层
                if hasattr(self.q_network, 'advantage_stream') and hasattr(self.q_network, 'value_stream'):
                    for stream in [self.q_network.advantage_stream, self.q_network.value_stream]:
                        if len(stream) > 0 and isinstance(stream[-1], nn.Linear):
                            # 输出层使用更大范围
                            nn.init.uniform_(stream[-1].weight, -0.2, 0.2)
                            nn.init.uniform_(stream[-1].bias, -2.0, 2.0)
                
                # 重新测试
                test_outputs = self.q_network(test_states)
                print(f"重新初始化后:")
                print(f"  输出范围: [{test_outputs.min().item():.4f}, {test_outputs.max().item():.4f}]")
                print(f"  输出标准差: {test_outputs.std().item():.4f}")
                print(f"  不同动作方差: {test_outputs.var(dim=1).mean().item():.4f}")
                print(f"  值域宽度: {test_outputs.max().item() - test_outputs.min().item():.4f}")
            
            # 检查是否有异常值
            if torch.isnan(test_outputs).any() or torch.isinf(test_outputs).any():
                print("错误：网络输出包含NaN或Inf")
        
        self.q_network.train()

    def _evaluate_board_like_greedy(self, board: Board) -> float:
        """参照GreedyPlayer的评估函数评估棋盘"""
        score = 0

        # Base piece values (和GreedyPlayer保持一致)
        piece_values = {8: 100, 7: 90, 6: 80, 5: 70, 4: 60, 3: 50, 2: 40, 1: 10}

        # Get pieces for both players
        self_pieces = board.get_player_pieces(self.player_id)
        opponent_pieces = board.get_player_pieces(1 - self.player_id)

        # Check for game-ending states
        if not opponent_pieces:
            return float("inf")  # Win
        if not self_pieces:
            return float("-inf")  # Loss

        # 1. Base Score - piece values
        for r, c in self_pieces:
            piece = board.get_piece(r, c)
            if piece and piece.revealed:
                score += piece_values[piece.strength]
            else:
                score += 30  # Unknown piece average value

        λ_1 = 0.8
        for r, c in opponent_pieces:
            piece = board.get_piece(r, c)
            if piece and piece.revealed:
                score -= λ_1 * piece_values[piece.strength]
            else:
                score -= λ_1 * 30  # Unknown piece average value

        # 2. Positional Rewards - Advancing (和GreedyPlayer保持一致)
        λ_3 = 15  # Advancing bonus weight
        for r, c in self_pieces:
            piece = board.get_piece(r, c)
            if piece and piece.revealed:
                score += λ_3 * (
                    6 - r
                )  # Higher score for pieces closer to opponent's side

        # 3. Mobility - count available moves
        λ_2 = 2  # Mobility weight
        possible_actions = board.get_all_possible_moves(self.player_id)
        score += len(possible_actions) * λ_2

        # 4. Safety Penalty - threatened pieces (和GreedyPlayer保持一致)
        safety_weight = 10
        for sr, sc in self_pieces:
            self_piece = board.get_piece(sr, sc)
            if self_piece and self_piece.revealed:
                # Check adjacent enemy pieces that can capture this piece
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    er, ec = sr + dr, sc + dc
                    if 0 <= er < 7 and 0 <= ec < 8:
                        enemy_piece = board.get_piece(er, ec)
                        if (
                            enemy_piece
                            and enemy_piece.revealed
                            and enemy_piece.player != self.player_id
                        ):
                            if enemy_piece.compare_strength(self_piece) == 1:
                                score -= (
                                    safety_weight
                                    * piece_values[self_piece.strength]
                                    / 100
                                )

        return score

    def _evaluate_action_like_greedy(self, board: Board, action: Tuple) -> float:
        """参照GreedyPlayer的评估函数评估动作"""
        # Get current board score
        current_score = self._evaluate_board_like_greedy(board)

        # Simulate the action on a copy of the board
        temp_board = copy.deepcopy(board)

        action_type = action[0]
        action_successful = False

        if action_type == "reveal":
            r, c = action[1]
            piece = temp_board.get_piece(r, c)
            if piece and piece.player == self.player_id and not piece.revealed:
                piece.revealed = True
                action_successful = True
        elif action_type == "move":
            start_pos, end_pos = action[1], action[2]
            action_successful = temp_board.try_move(start_pos, end_pos)

        if not action_successful:
            return float("-inf")  # Invalid action

        # Get new board score after action
        new_score = self._evaluate_board_like_greedy(temp_board)

        return new_score - current_score

    def _guided_exploration(self, board: Board, valid_actions: List[Tuple]) -> Tuple:
        """优化的引导性探索：使用类似GreedyPlayer的评估来指导探索 - 修复属性引用"""
        if not valid_actions:
            return ("reveal", (0, 0), None)

        action_scores = []

        for action in valid_actions:
            # 传统评估
            try:
                traditional_score = self._evaluate_action_like_greedy(board, action)
            except:
                traditional_score = 0.0

            # 根据当前阶段配置动态调整权重
            config = self.phase_configs.get(
                self.current_phase, self.phase_configs["phase1"]
            )
            force_until = config["epsilon_force_until"]

            # 计算阶段内的进度
            if self.episode_in_phase < force_until:
                # 强制探索期：高随机性
                noise_level = 0.5
                selection_range = 0.7
            else:
                # 自适应期：根据阶段降低随机性
                if self.current_phase == "phase1":
                    noise_level = 0.4
                    selection_range = 0.6
                elif self.current_phase == "phase2":
                    noise_level = 0.3
                    selection_range = 0.4
                else:  # phase3
                    noise_level = 0.1
                    selection_range = 0.2

            # 添加适应性随机扰动
            adjusted_score = traditional_score + random.uniform(
                -noise_level, noise_level
            )

            action_scores.append((action, adjusted_score))

        # 按分数排序
        action_scores.sort(key=lambda x: x[1], reverse=True)

        # 动态选择范围
        top_count = max(1, int(len(action_scores) * selection_range))
        best_actions = action_scores[:top_count]

        # 加权随机选择
        actions, scores = zip(*best_actions)
        min_score = min(scores)
        adjusted_scores = [s - min_score + 1.0 for s in scores]

        # 温度参数：根据阶段调整
        if self.current_phase == "phase1":
            temperature = 2.0
        elif self.current_phase == "phase2":
            temperature = 1.0
        else:  # phase3
            temperature = 0.3

        adjusted_scores = [(s ** (1.0 / temperature)) for s in adjusted_scores]

        total_score = sum(adjusted_scores)

        if total_score > 0:
            rand_val = random.random() * total_score
            cumulative = 0.0
            for i, weight in enumerate(adjusted_scores):
                cumulative += weight
                if rand_val <= cumulative:
                    return actions[i]

        return random.choice(valid_actions)

    def _board_to_state_enhanced(self, board: Board) -> torch.Tensor:
        """增强的状态表示 - 包含历史信息"""
        # 原始状态
        state = np.zeros((7, 8, 8))  # 增加到8个通道

        for r in range(7):
            for c in range(8):
                piece = board.get_piece(r, c)
                if piece:
                    # 通道0: 玩家归属 (0=red, 1=blue, -1=empty)
                    state[r, c, 0] = piece.player
                    # 通道1: 棋子强度 (归一化到0-1)
                    state[r, c, 1] = piece.strength / 8.0
                    # 通道2: 是否翻开
                    state[r, c, 2] = 1 if piece.revealed else 0
                    # 通道3: 是否有棋子
                    state[r, c, 3] = 1
                    # 通道4: 当前玩家的棋子
                    state[r, c, 4] = 1 if piece.player == self.player_id else 0
                    # 通道5: 对手的棋子
                    state[r, c, 5] = 1 if piece.player != self.player_id else 0

        # 通道6: 最近访问过的位置
        if hasattr(self.reward_function, "position_history"):
            for pos in self.reward_function.position_history[-5:]:
                r, c = pos
                if 0 <= r < 7 and 0 <= c < 8:
                    state[r, c, 6] = 1.0

        # 通道7: 当前可移动的棋子
        my_pieces = board.get_player_pieces(self.player_id)
        for r, c in my_pieces:
            piece = board.get_piece(r, c)
            if piece and piece.revealed:
                state[r, c, 7] = 1.0

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
                return ("reveal", (0, 0), None)
        elif len(action) == 3:
            return action
        else:
            return ("reveal", (0, 0), None)

    def choose_action(self, board: Board, valid_actions: List[Tuple]) -> Tuple:
        """改进的动作选择：支持引导性探索"""
        if not valid_actions:
            all_actions = board.get_all_possible_moves(self.player_id)
            if not all_actions:
                return ("reveal", (0, 0), None)
            valid_actions = all_actions

        # 标准化所有动作为3元组格式
        normalized_actions = [
            self._normalize_action(action) for action in valid_actions
        ]
        validated_actions = [
            action for action in normalized_actions if self._validate_action(action)
        ]

        if not validated_actions:
            return ("reveal", (0, 0), None)

        # epsilon-greedy策略，但探索方式可以选择
        if random.random() < self.epsilon:
            if self.exploration_strategy == "guided":
                return self._guided_exploration(board, validated_actions)
            else:
                return random.choice(validated_actions)

        # 利用（贪心）策略
        state = self._board_to_state_enhanced(board)
        valid_indices = [self._action_to_index(action) for action in validated_actions]

        self.q_network.eval()

        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))

            # 创建掩码，只考虑有效动作
            masked_q_values = torch.full((self.action_size,), float("-inf")).to(device)
            for idx in valid_indices:
                if 0 <= idx < self.action_size:
                    masked_q_values[idx] = q_values[0][idx]

            best_index = masked_q_values.argmax().item()
            best_action = self._index_to_action(best_index)

            if (
                best_action
                and self._validate_action(best_action)
                and best_action in validated_actions
            ):
                return best_action
            else:
                return random.choice(validated_actions)

        if self.training_mode:
            self.q_network.train()

    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def normalize_reward(self, reward: float) -> float:
        """修复奖励归一化 - 保持更多信号强度"""
        if abs(reward) < 1e-6:
            return 0.0
        
        # 使用更保守的归一化，保持奖励的差异性
        if abs(reward) <= 2.0:
            return reward  # 小奖励完全不压缩
        elif abs(reward) <= 10.0:
            # 轻度压缩
            return np.sign(reward) * (2.0 + 0.8 * (abs(reward) - 2.0))
        else:
            # 中度压缩
            return np.sign(reward) * (2.0 + 0.8 * 8.0 + 0.5 * (abs(reward) - 10.0))

    def store_experience(
        self,
        board_before: Board,
        action: Tuple,
        reward: float,
        board_after: Board,
        result: int,
    ):
        """存储经验时使用增强状态表示"""
        normalized_reward = self.normalize_reward(reward)

        # 统一使用增强状态表示
        state = self._board_to_state_enhanced(board_before)
        next_state = self._board_to_state_enhanced(board_after)
        action_index = self._action_to_index(action)

        self.memory.push(state, action_index, normalized_reward, next_state, result)

    def replay(self):
        """改进的经验回放学习 - 增强调试和修复损失过小问题"""
        if len(self.memory) < self.batch_size:
            return

        # 确保网络在训练模式
        self.q_network.train()
        self.target_network.eval()

        # 采样经验
        experiences, indices, weights = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # 转换为张量
        state_batch = torch.stack(batch.state)
        action_batch = torch.LongTensor(batch.action).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        next_state_batch = torch.stack(batch.next_state)
        result_batch = torch.LongTensor(batch.result).to(device)

        # 修复：确保weights张量在正确的设备上
        if isinstance(weights, torch.Tensor):
            weights_tensor = weights.to(device)
        else:
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

        # 不要过度限制奖励范围 - 保持奖励的差异性
        # original_reward_range = (reward_batch.min().item(), reward_batch.max().item())
        reward_batch = torch.clamp(reward_batch, -50.0, 50.0)  # 放宽限制

        # 计算当前Q值
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            if self.use_double:
                # Double DQN
                self.q_network.eval()
                next_actions = self.q_network(next_state_batch).argmax(dim=1)
                self.q_network.train()

                next_q_values = (
                    self.target_network(next_state_batch)
                    .gather(1, next_actions.unsqueeze(1))
                    .squeeze()
                )
            else:
                next_q_values = self.target_network(next_state_batch).max(1)[0]

            gamma = 0.99
            target_q_values = reward_batch + (
                gamma * next_q_values * (result_batch == -1).float()
            )
            target_q_values = torch.clamp(target_q_values, -200.0, 200.0)  # 放宽限制

        # 计算TD误差
        td_errors = (target_q_values - current_q_values.squeeze()).detach()

        # 初始化调试计数器
        # if not hasattr(self, '_debug_counter'):
        #     self._debug_counter = 0
        #     self._loss_history = []
        # self._debug_counter += 1

        # 每100次replay输出详细调试信息
        # if self._debug_counter % 100 == 0:
        #     print(f"\n=== 详细调试信息 (第{self._debug_counter}次replay) ===")
        #     print(f"奖励范围: 原始[{original_reward_range[0]:.4f}, {original_reward_range[1]:.4f}] -> 限制后[{reward_batch.min().item():.4f}, {reward_batch.max().item():.4f}]")
        #     print(f"当前Q值: 均值={current_q_values.mean().item():.4f}, 范围=[{current_q_values.min().item():.4f}, {current_q_values.max().item():.4f}], 标准差={current_q_values.std().item():.4f}")
        #     print(f"下一步Q值: 均值={next_q_values.mean().item():.4f}, 范围=[{next_q_values.min().item():.4f}, {next_q_values.max().item():.4f}], 标准差={next_q_values.std().item():.4f}")
        #     print(f"目标Q值: 均值={target_q_values.mean().item():.4f}, 范围=[{target_q_values.min().item():.4f}, {target_q_values.max().item():.4f}], 标准差={target_q_values.std().item():.4f}")
        #     print(f"TD误差: 均值={td_errors.mean().item():.4f}, 范围=[{td_errors.min().item():.4f}, {td_errors.max().item():.4f}], 标准差={td_errors.std().item():.4f}")
            
        #     # 检查action分布
        #     unique_actions, action_counts = torch.unique(action_batch, return_counts=True)
        #     print(f"动作分布: {len(unique_actions)}种不同动作，最频繁动作出现{action_counts.max().item()}次")
            
        #     # 检查result分布
        #     result_dist = [(result_batch == i).sum().item() for i in [-1, 0, 1]]
        #     print(f"结果分布: 进行中={result_dist[0]}, 智能体胜={result_dist[1]}, 对手胜={result_dist[2]}")

        # 更新优先级
        self.memory.update_priorities(
            indices, torch.clamp(td_errors.abs(), 0.001, 20).cpu().numpy()
        )

        # 计算损失 - 如果Q值差异太小，人为增加一些多样性
        q_diff = (target_q_values - current_q_values.squeeze()).abs().mean().item()
        if q_diff < 0.01:  # Q值差异太小
            print(f"Q值差异过小({q_diff:.6f})，添加学习增强...")
            # 添加一些噪声来增加学习信号
            target_noise = torch.randn_like(target_q_values) * 0.05
            target_q_values = target_q_values + target_noise
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values, reduction='none')
        weighted_loss = (loss * weights_tensor).mean()
        
        # 记录损失历史
        # self._loss_history.append(weighted_loss.item())
        # if len(self._loss_history) > 1000:
        #     self._loss_history = self._loss_history[-1000:]  # 只保留最近1000个

        # 检查损失是否异常小
        # if weighted_loss.item() < 0.01:
            # print(f"\n警告：损失过小 ({weighted_loss.item():.6f})！")
            # print(f"  当前Q值均值: {current_q_values.mean().item():.6f} ± {current_q_values.std().item():.6f}")
            # print(f"  目标Q值均值: {target_q_values.mean().item():.6f} ± {target_q_values.std().item():.6f}")
            # print(f"  Q值差异: {(target_q_values - current_q_values.squeeze()).abs().mean().item():.6f}")
            # print(f"  最近100次损失均值: {np.mean(self._loss_history[-100:]):.6f}")
            
            # 检查是否是网络饱和
            # with torch.no_grad():
                # 测试随机输入的网络输出
                # random_state = torch.randn_like(state_batch[:1])
                # random_output = self.q_network(random_state)
                # print(f"  随机输入的网络输出范围: [{random_output.min().item():.4f}, {random_output.max().item():.4f}]")
                # print(f"  随机输入的网络输出标准差: {random_output.std().item():.4f}")
            
            # 如果损失持续过小，自动调整学习率
            # if len(self._loss_history) >= 100 and np.mean(self._loss_history[-100:]) < 0.005:
            #     print("  连续100次损失都过小，可能网络已饱和！")
            #     current_lr = self.get_learning_rate()
            #     if current_lr < 1e-4:
            #         new_lr = min(current_lr * 3, 1e-3)  # 更激进的学习率提升
            #         for param_group in self.optimizer.param_groups:
            #             param_group['lr'] = new_lr
            #         print(f"  自动大幅提升学习率: {current_lr:.6f} -> {new_lr:.6f}")

        # 检查损失是否为NaN或过大
        # if torch.isnan(weighted_loss) or weighted_loss.item() > 100:
        #     print(f"警告：损失异常 ({weighted_loss.item():.6f})，跳过此次更新")
        #     return

        # 反向传播
        self.optimizer.zero_grad()
        weighted_loss.backward()

        # 检查梯度
        # total_grad_norm = 0
        # grad_count = 0
        # max_grad = 0
        # for param in self.q_network.parameters():
        #     if param.grad is not None:
        #         param_grad_norm = param.grad.data.norm(2).item()
        #         total_grad_norm += param_grad_norm ** 2
        #         max_grad = max(max_grad, param_grad_norm)
        #         grad_count += 1
        # total_grad_norm = total_grad_norm ** 0.5

        # if self._debug_counter % 100 == 0:
            # print(f"梯度信息: 总范数={total_grad_norm:.6f}, 最大梯度={max_grad:.6f}, 有梯度参数数={grad_count}")

        # 如果梯度太小，可能需要增加学习率
        # if total_grad_norm < 1e-6:
        #     print(f"梯度过小({total_grad_norm:.8f})，可能需要调整学习率")

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 5.0)

        self.optimizer.step()

        # 记录损失
        self.losses.append(weighted_loss.item())

        # 定期输出损失趋势
        # if self._debug_counter % 500 == 0 and len(self._loss_history) >= 100:
        #     recent_avg = np.mean(self._loss_history[-100:])
        #     older_avg = np.mean(self._loss_history[-200:-100]) if len(self._loss_history) >= 200 else recent_avg
        #     trend = "上升" if recent_avg > older_avg else "下降" if recent_avg < older_avg else "稳定"
        #     print(f"损失趋势 (最近500次): 当前平均={recent_avg:.6f}, 趋势={trend}")

    def disable_adaptive_lr(self):
        """禁用自适应学习率，只使用基础调度器"""
        self.adaptive_lr = False
        print("已禁用自适应学习率调整")

    def enable_adaptive_lr(self):
        """启用自适应学习率"""
        self.adaptive_lr = True
        print("已启用自适应学习率调整")

    def set_phase(self, phase_name: str):
        """设置当前训练阶段"""
        if phase_name in self.phase_configs:
            old_phase = self.current_phase
            self.current_phase = phase_name
            self.episode_in_phase = 0  # 重置阶段内episode计数
            config = self.phase_configs[phase_name]
            print(
                f"切换训练阶段: {old_phase} -> {phase_name} ({config['description']})"
            )
            print(
                f"  Epsilon控制: 强制{config['epsilon_force_until']}回合, 最小ε={config['epsilon_min']:.2f}"
            )
            print(
                f"  学习率控制: 强制{config['lr_force_until']}回合, 频率={config['lr_update_frequency']}"
            )
        else:
            print(f"警告: 未知的阶段名称 {phase_name}")

    def update_learning_rate_by_phase(
        self, phase_name: str, episode_in_phase: int, batch_win_rate: float = None
    ):
        """分阶段的学习率控制 - 修复学习率过低的问题"""
        if phase_name not in self.phase_configs:
            self.update_learning_rate(batch_win_rate)
            return

        # 更新内部计数器
        self.episode_in_phase = episode_in_phase

        config = self.phase_configs[phase_name]
        force_until = config["lr_force_until"]
        update_freq = config["lr_update_frequency"]
        stable_range = config["lr_stable_range"]
        adaptive_range = config["lr_adaptive_range"]

        current_lr = self.get_learning_rate()

        # 阶段内强制稳定期
        if episode_in_phase < force_until:
            # 强制期内只进行微调，保持学习率相对稳定
            if episode_in_phase > 0 and episode_in_phase % (update_freq * 2) == 0:
                if batch_win_rate is not None:
                    if batch_win_rate < 0.2:
                        multiplier = stable_range[1] * 1.5  # 小幅提高
                        print(
                            f"{phase_name}: 强制期，批次胜率过低({batch_win_rate:.3f})，小幅提高学习率"
                        )
                    elif batch_win_rate > 0.8:
                        multiplier = stable_range[0]  # 小幅降低
                        print(
                            f"{phase_name}: 强制期，批次胜率过高({batch_win_rate:.3f})，小幅降低学习率"
                        )
                    else:
                        multiplier = 1.0  # 保持不变

                    new_lr = current_lr * multiplier
                    # 确保学习率不会太低
                    new_lr = max(1e-3, min(new_lr, self.lr_max))  # 提高最小学习率

                    if abs(new_lr - current_lr) > 1e-5:
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = new_lr
                        print(
                            f"{phase_name}: 强制期学习率调整: {current_lr:.6f} -> {new_lr:.6f}"
                        )
            return

        # 阶段内自适应调整期
        if episode_in_phase > 0 and episode_in_phase % update_freq == 0:
            old_lr = current_lr

            if self.adaptive_lr and batch_win_rate is not None:
                # 根据批次胜率调整学习率 - 使用阶段特定的范围
                if batch_win_rate < 0.2:
                    # 批次胜率很低，增加学习率
                    multiplier = adaptive_range[1]
                    print(
                        f"{phase_name}: 批次胜率过低({batch_win_rate:.3f})，提高学习率"
                    )
                elif batch_win_rate < 0.4:
                    # 批次胜率较低，适度增加学习率
                    multiplier = adaptive_range[1] + 1.0
                elif batch_win_rate > 0.8:
                    # 批次胜率很高，但不要降得太低
                    multiplier = max(0.9, adaptive_range[0])  # 限制最大衰减
                    print(
                        f"{phase_name}: 批次胜率较高({batch_win_rate:.3f})，适度降低学习率"
                    )
                elif batch_win_rate > 0.65:
                    # 批次胜率较高，适度降低学习率
                    multiplier = (adaptive_range[0] + 1.0) / 2
                else:
                    # 批次胜率正常，保持当前学习率
                    multiplier = 1.0

                new_lr = current_lr * multiplier

                # 限制学习率范围 - 提高最小值
                new_lr = max(1e-3, min(new_lr, self.lr_max))

                # 更新学习率
                if abs(new_lr - current_lr) > 1e-5:
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = new_lr
                    print(
                        f"{phase_name}: 基于批次胜率调整学习率: {current_lr:.6f} -> {new_lr:.6f}"
                    )

            # 记录学习率变化
            if (
                abs(self.get_learning_rate() - old_lr) > 1e-5
                and episode_in_phase % 100 == 0
            ):
                win_rate_str = (
                    f"{batch_win_rate:.3f}" if batch_win_rate is not None else "N/A"
                )
                print(
                    f"{phase_name}: Episode {episode_in_phase}, "
                    f"学习率: {old_lr:.6f} -> {self.get_learning_rate():.6f}, "
                    f"批次胜率: {win_rate_str}"
                )

    def decay_epsilon_by_phase(
        self, phase_name: str, episode_in_phase: int, batch_win_rate: float = None
    ):
        """分阶段的epsilon衰减控制"""
        if phase_name not in self.phase_configs:
            # 回退到原来的方法
            self.decay_epsilon(batch_win_rate)
            return

        # 更新内部计数器
        self.episode_in_phase = episode_in_phase

        config = self.phase_configs[phase_name]
        force_until = config["epsilon_force_until"]
        min_epsilon = config["epsilon_min"]
        decay_rate = config["epsilon_decay_rate"]

        # 阶段内强制探索期
        if episode_in_phase < force_until:
            # 强制保持高探索
            self.epsilon = max(min_epsilon, self.epsilon * 0.9999)  # 极慢衰减
            if episode_in_phase % 100 == 0:
                print(
                    f"{phase_name}: 强制探索期 ({episode_in_phase}/{force_until}), "
                    f"保持高epsilon={self.epsilon:.3f}"
                )
            return

        # 阶段内自适应衰减期 - 使用批次胜率
        old_epsilon = self.epsilon

        if batch_win_rate is not None:
            # 根据批次胜率调整衰减速度
            if batch_win_rate < 0.25:
                # 批次胜率很低，几乎停止衰减
                actual_decay = 0.9999
                if episode_in_phase % 50 == 0:
                    print(
                        f"{phase_name}: 批次胜率过低({batch_win_rate:.3f})，减缓epsilon衰减"
                    )
            elif batch_win_rate < 0.4:
                # 批次胜率较低，减慢衰减
                actual_decay = max(0.9995, decay_rate)
            elif batch_win_rate > 0.8:
                # 批次胜率很高，可以加快衰减
                actual_decay = min(0.992, decay_rate * 0.9)
                if episode_in_phase % 50 == 0:
                    print(
                        f"{phase_name}: 批次胜率较高({batch_win_rate:.3f})，加快epsilon衰减"
                    )
            elif batch_win_rate > 0.65:
                # 批次胜率较高，正常衰减
                actual_decay = decay_rate
            else:
                # 中等批次胜率，稍慢衰减
                actual_decay = min(0.997, decay_rate * 1.002)
        else:
            # 使用阶段默认衰减率
            actual_decay = decay_rate

        self.epsilon = max(min_epsilon, self.epsilon * actual_decay)

        # 记录显著的epsilon变化 - 修复这里的f-string格式错误
        if abs(old_epsilon - self.epsilon) > 0.05 or episode_in_phase % 100 == 0:
            # 修复：先处理条件表达式，再放入f-string
            win_rate_str = (
                f"{batch_win_rate:.3f}" if batch_win_rate is not None else "N/A"
            )
            print(
                f"{phase_name}: Episode {episode_in_phase}, "
                f"epsilon: {old_epsilon:.3f} -> {self.epsilon:.3f}, "
                f"批次胜率: {win_rate_str}"
            )

    def get_learning_rate(self) -> float:
        """获取当前学习率"""
        # 从优化器中获取实际的学习率
        return self.optimizer.param_groups[0]["lr"]

    def update_learning_rate(self, batch_win_rate: float = None):
        """原有的学习率更新方法（后备）"""
        if not hasattr(self, "training_stats"):
            self.training_stats = {"episodes": 0}

        current_lr = self.get_learning_rate()

        # 自适应调整 - 基于批次胜率
        if self.adaptive_lr and batch_win_rate is not None:
            # 根据批次表现调整学习率
            if batch_win_rate < 0.2:
                multiplier = 1.2
                print(f"批次胜率过低({batch_win_rate:.3f})，提高学习率")
            elif batch_win_rate < 0.4:
                multiplier = 1.1
            elif batch_win_rate > 0.8:
                multiplier = 0.9
                print(f"批次胜率较高({batch_win_rate:.3f})，降低学习率")
            elif batch_win_rate > 0.65:
                multiplier = 0.95
            else:
                multiplier = 1.0

            new_lr = current_lr * multiplier
            new_lr = max(self.lr_min, min(new_lr, self.lr_max))

            if abs(new_lr - current_lr) > 1e-7:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = new_lr
                print(f"基于批次胜率调整学习率: {current_lr:.6f} -> {new_lr:.6f}")

    # 保持原有的decay_epsilon方法作为后备
    def decay_epsilon(self, batch_win_rate: float = None):
        """原有的epsilon衰减方法（后备） - 使用批次胜率"""
        if batch_win_rate is not None:
            if batch_win_rate < 0.25:
                decay = 0.9999
                print(f"批次胜率过低({batch_win_rate:.3f})，减缓epsilon衰减")
            elif batch_win_rate < 0.4:
                decay = 0.9995
            elif batch_win_rate > 0.8:
                decay = 0.992
                print(f"批次胜率较高({batch_win_rate:.3f})，加快epsilon衰减")
            elif batch_win_rate > 0.65:
                decay = 0.995
            else:
                decay = 0.997
        else:
            decay = self.epsilon_decay

        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * decay)

        if abs(old_epsilon - self.epsilon) > 0.05:
            print(f"Epsilon显著变化: {old_epsilon:.3f} -> {self.epsilon:.3f}")

    def set_training_mode(self, training: bool = True):
        """设置训练/测试模式 - 同时控制PyTorch模型模式"""
        self.training_mode = training

        if not training:
            # 测试模式：保存当前epsilon，设置为0，网络设为eval模式
            self._saved_epsilon = self.epsilon
            self.epsilon = self.test_mode_epsilon
            self.q_network.eval()
            self.target_network.eval()
            print(f"切换到测试模式，epsilon: {self.epsilon}")
        else:
            # 训练模式：恢复保存的epsilon，网络设为train模式
            if hasattr(self, "_saved_epsilon"):
                self.epsilon = self._saved_epsilon
            self.q_network.train()
            self.target_network.eval()  # 目标网络始终为eval模式
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
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.base_scheduler.state_dict(),  # 新增
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "hyperparameters": {  # 新增：保存超参数
                "learning_rate": self.learning_rate,
                "initial_learning_rate": self.initial_learning_rate,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min,
                "batch_size": self.batch_size,
                "use_dueling": self.use_dueling,
                "use_double": self.use_double,
                "lr_min": self.lr_min,
                "lr_max": self.lr_max,
            },
            "model_info": {
                "state_size": self.state_size,
                "action_size": self.action_size,
                "ai_type": self.ai_type,
            },
        }
        save_model_data(model_data, f"{filename}.pkl")
        print(f"模型已保存到: {filename}.pkl")

    def load_model(self, filename: str) -> bool:
        """加载模型"""
        data = load_model_data(f"{filename}.pkl")
        if data:
            try:
                self.q_network.load_state_dict(data["q_network_state_dict"])
                self.target_network.load_state_dict(data["target_network_state_dict"])
                self.optimizer.load_state_dict(data["optimizer_state_dict"])

                if "scheduler_state_dict" in data:
                    self.base_scheduler.load_state_dict(data["scheduler_state_dict"])

                self.epsilon = data.get("epsilon", self.epsilon)
                self.episode_count = data.get("episode_count", 0)

                # 加载超参数
                if "hyperparameters" in data:
                    hyper = data["hyperparameters"]
                    self.learning_rate = hyper.get("learning_rate", self.learning_rate)
                    self.initial_learning_rate = hyper.get(
                        "initial_learning_rate", self.initial_learning_rate
                    )
                    self.epsilon_decay = hyper.get("epsilon_decay", self.epsilon_decay)
                    self.epsilon_min = hyper.get("epsilon_min", self.epsilon_min)
                    self.batch_size = hyper.get("batch_size", self.batch_size)
                    self.lr_min = hyper.get("lr_min", self.lr_min)
                    self.lr_max = hyper.get("lr_max", self.lr_max)

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

    def _agent_update(
        self,
        board_before: Board,
        action: Tuple,
        reward: float,
        board_after: Board,
        result: int,
    ):
        """更新智能体"""
        # 存储经验
        self.agent.store_experience(board_before, action, reward, board_after, result)

        # 学习
        self.agent.replay()

        # 更新目标网络
        episodes = self.agent.training_stats["episodes"]
        if episodes % self.target_update_freq == 0:
            self.agent.update_target_network()

    def train_episode(self, opponent=None, **kwargs) -> Tuple[float, int, int]:
        """训练一个回合 - 重置奖励函数历史"""
        if opponent:
            self.opponent = opponent

        # 重置奖励函数的历史记录
        if hasattr(self.agent.reward_function, "reset_history"):
            self.agent.reward_function.reset_history()

        # 调用父类方法
        return super().train_episode(opponent, **kwargs)

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
