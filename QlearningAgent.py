import numpy as np
import random
import copy
from collections import deque
from typing import Tuple, List, Dict
from base import Player, BaseTrainer, GameEnvironment, Board
from utils import save_model_data, load_model_data

class QLearningAgent(Player):
    """Q-learning智能体 - 增强版，支持分阶段控制"""
    
    def __init__(self, player_id: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.9,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.05):
        super().__init__(player_id)
        # 基础参数
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        self.episode_count = 0
        
        # 学习率调整参数
        self.lr_min = learning_rate * 0.1
        self.lr_max = learning_rate * 3
        self.adaptive_lr = True
        
        # 统一的分阶段超参数控制配置
        self.phase_configs = {
            "phase1": {
                "epsilon_force_until": 200,
                "epsilon_min": 0.8,
                "epsilon_decay_rate": 0.998,
                "lr_force_until": 150,
                "lr_update_frequency": 50,
                "lr_stable_range": (0.95, 1.1),
                "lr_adaptive_range": (0.8, 1.5),
                "description": "基础学习阶段",
            },
            "phase2": {
                "epsilon_force_until": 150,
                "epsilon_min": 0.5,
                "epsilon_decay_rate": 0.997,
                "lr_force_until": 100,
                "lr_update_frequency": 50,
                "lr_stable_range": (0.9, 1.1),
                "lr_adaptive_range": (0.7, 1.8),
                "description": "进阶学习阶段",
            },
            "phase3": {
                "epsilon_force_until": 100,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay_rate": self.epsilon_decay,
                "lr_force_until": 50,
                "lr_update_frequency": 100,
                "lr_stable_range": (0.95, 1.05),
                "lr_adaptive_range": (0.8, 1.2),
                "description": "策略精炼阶段",
            },
        }
        
        self.current_phase = "phase1"
        self.episode_in_phase = 0
        
        # 测试模式控制
        self.training_mode = True
        self.test_mode_epsilon = 0.0
        
        self.ai_type = f"QL (ε={epsilon:.2f})" if epsilon > 0 else "QL (Trained)"
    
    def get_learning_rate(self) -> float:
        """获取当前学习率"""
        return self.learning_rate
    
    def _board_to_state(self, board: Board) -> np.ndarray:
        """将Board对象转换为状态数组"""
        # 创建一个7x8的二维数组，表示棋盘
        state = np.zeros((7, 8, 4))  # 4个通道: 己方棋子/敌方棋子/已翻开/强度
        
        for r in range(7):
            for c in range(8):
                piece = board.get_piece(r, c)
                if piece:
                    # 通道1: 己方棋子
                    if piece.player == self.player_id and piece.revealed:
                        state[r, c, 0] = 1
                    # 通道2: 敌方棋子
                    elif piece.player != self.player_id and piece.revealed:
                        state[r, c, 1] = 1
                    # 通道3: 棋子是否已翻开
                    if piece.revealed:
                        state[r, c, 2] = 1
                    # 通道4: 棋子强度
                    if piece.revealed:
                        state[r, c, 3] = piece.strength / 8.0  # 归一化强度
        
        # 添加当前玩家信息
        state_flat = state.flatten()
        player_info = np.array([1.0 if self.player_id == 0 else -1.0])
        return np.concatenate([state_flat, player_info])
        
    def get_state_key(self, board: Board) -> str:
        """将Board转换为可哈希的键"""
        state = self._board_to_state(board)
        # 只保留小数点后两位，减少状态空间
        return str(state.round(2).tolist())
    
    def get_action_key(self, action: Tuple) -> str:
        """将动作转换为可哈希的键"""
        return str(action)
    
    def choose_action(self, board: Board, valid_actions: List[Tuple]) -> Tuple:
        """使用epsilon-greedy策略选择动作 - 直接使用Board对象"""
        if not valid_actions:
            return ("reveal", (0, 0), None)
            
        # 标准化动作为三元组格式
        normalized_actions = []
        for action in valid_actions:
            if len(action) == 2:
                action_type, pos = action
                if action_type == "reveal":
                    normalized_actions.append((action_type, pos, None))
                else:
                    # 跳过
                    continue
            elif len(action) == 3:
                normalized_actions.append(action)
        
        if not normalized_actions:
            return ("reveal", (0, 0), None)
            
        state_key = self.get_state_key(board)
        
        # epsilon-greedy策略
        if random.random() < self.epsilon:
            return random.choice(normalized_actions)
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            best_action = None
            best_q_value = float('-inf')
            
            for action in normalized_actions:
                action_key = self.get_action_key(action)
                q_value = self.q_table[state_key].get(action_key, 0.0)
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            return best_action if best_action else random.choice(normalized_actions)
    
    def update_q_value(self, board_before: Board, action: Tuple, reward: float, 
                      board_after: Board, next_valid_actions: List[Tuple], result: int):
        """更新Q值 - 直接使用Board对象，支持结束状态"""
        state_key = self.get_state_key(board_before)
        action_key = self.get_action_key(action)
        
        # 初始化状态-动作值
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
            
        # 对于结束状态，没有下一个状态
        if result != -1:
            # 游戏结束，只考虑当前奖励
            self.q_table[state_key][action_key] = reward
            return
            
        next_state_key = self.get_state_key(board_after)
        
        # 计算下一状态的最大Q值
        max_next_q = 0.0
        if next_valid_actions:
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = {}
                
            next_q_values = [
                self.q_table[next_state_key].get(self.get_action_key(a), 0.0)
                for a in next_valid_actions
            ]
            if next_q_values:
                max_next_q = max(next_q_values)
        
        # Q-learning更新公式
        current_q = self.q_table[state_key][action_key]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # 更新Q值
        self.q_table[state_key][action_key] = new_q
    
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
    
    def set_phase(self, phase_name: str):
        """设置当前训练阶段"""
        if phase_name in self.phase_configs:
            self.current_phase = phase_name
            self.episode_in_phase = 0
            print(f"QLearningAgent: 切换到 {phase_name}")
        else:
            print(f"警告: 未知阶段 {phase_name}, 保持当前阶段 {self.current_phase}")

    def decay_epsilon(self, batch_win_rate: float = None):
        """原始的epsilon衰减方法（后备）"""
        # 根据胜率调整衰减率
        if batch_win_rate is not None:
            if batch_win_rate < 0.2:  # 表现差
                decay = self.epsilon_decay * 0.95  # 降低衰减速度
            elif batch_win_rate > 0.7:  # 表现好
                decay = self.epsilon_decay * 1.05  # 加快衰减速度
            else:
                decay = self.epsilon_decay
        else:
            decay = self.epsilon_decay
        
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * decay)
        
        self.episode_count += 1
        
        if abs(old_epsilon - self.epsilon) > 0.05:
            print(f"epsilon衰减: {old_epsilon:.3f} -> {self.epsilon:.3f}")

    def decay_epsilon_by_phase(self, phase_name: str, episode_in_phase: int, batch_win_rate: float = None):
        """改进的分阶段epsilon衰减控制 - 对近期胜率更敏感"""
        if phase_name not in self.phase_configs:
            self.decay_epsilon(batch_win_rate)
            return
        
        # 更新内部计数器
        self.episode_in_phase = episode_in_phase
        
        config = self.phase_configs[phase_name]
        force_until = config["epsilon_force_until"]
        min_epsilon = config["epsilon_min"]
        decay_rate = config["epsilon_decay_rate"]
        
        # 计算最近100回合的胜率(如果可用)
        recent_win_rate = None
        if hasattr(self, '_recent_results') and len(self._recent_results) > 0:
            recent_wins = sum(1 for res in self._recent_results[-100:] if res == self.player_id)
            recent_draws = sum(1 for res in self._recent_results[-100:] if res == 2)
            recent_count = min(len(self._recent_results), 100)
            recent_win_rate = (recent_wins + 0.5 * recent_draws) / recent_count if recent_count > 0 else None
        
        # 阶段内强制探索期
        if episode_in_phase < force_until:
            self.epsilon = max(min_epsilon, self.epsilon * 0.9998)
            if episode_in_phase % 100 == 0:
                print(f"{phase_name}: 强制探索期 ({episode_in_phase}/{force_until}), "
                    f"保持高epsilon={self.epsilon:.3f}")
            return
        
        # 阶段内自适应衰减期
        old_epsilon = self.epsilon
        
        # 优先使用最近100回合的胜率，否则使用批次胜率
        win_rate_to_use = recent_win_rate if recent_win_rate is not None else batch_win_rate
        
        if win_rate_to_use is not None:
            # 胜率低于40%时，增加epsilon
            if win_rate_to_use < 0.4:
                # 胜率越低，epsilon增加越多
                actual_decay = 1.01 + (0.4 - win_rate_to_use) * 0.1
                if episode_in_phase % 50 == 0:
                    print(f"{phase_name}: 最近胜率过低({win_rate_to_use:.3f})，提高epsilon以增加探索")
            # 胜率40%-60%之间，缓慢衰减
            elif win_rate_to_use < 0.6:
                actual_decay = 0.999
                if episode_in_phase % 100 == 0:
                    print(f"{phase_name}: 胜率适中({win_rate_to_use:.3f})，缓慢衰减epsilon")
            # 胜率高于60%，可以加速衰减
            else:
                actual_decay = decay_rate
                if episode_in_phase % 50 == 0:
                    print(f"{phase_name}: 胜率良好({win_rate_to_use:.3f})，正常衰减epsilon")
        else:
            actual_decay = decay_rate
        
        # 应用衰减或增长因子
        self.epsilon = min(0.9, max(min_epsilon, self.epsilon * actual_decay))
        
        # 记录epsilon变化
        if abs(old_epsilon - self.epsilon) > 0.01 or episode_in_phase % 100 == 0:
            win_rate_str = f"{win_rate_to_use:.3f}" if win_rate_to_use is not None else "N/A"
            print(f"{phase_name}: Episode {episode_in_phase}, "
                f"epsilon: {old_epsilon:.3f} → {self.epsilon:.3f}, "
                f"最近胜率: {win_rate_str}")
        
        # 维护最近结果队列
        if not hasattr(self, '_recent_results'):
            self._recent_results = deque(maxlen=200)

    def update_learning_rate(self, batch_win_rate: float = None):
        """原始的学习率更新方法（后备）"""
        if not self.adaptive_lr:
            return
            
        current_lr = self.learning_rate
            
        # 根据批次胜率调整学习率
        if batch_win_rate is not None:
            if batch_win_rate < 0.2:  # 表现差
                new_lr = current_lr * 1.2
                print(f"表现不佳 (胜率={batch_win_rate:.3f})，提高学习率")
            elif batch_win_rate > 0.7:  # 表现好
                new_lr = current_lr * 0.9
                print(f"表现优秀 (胜率={batch_win_rate:.3f})，降低学习率")
            else:
                new_lr = current_lr
                
            # 限制学习率范围
            new_lr = max(self.lr_min, min(new_lr, self.lr_max))
            
            if abs(new_lr - current_lr) > 1e-6:
                self.learning_rate = new_lr
                print(f"学习率调整: {current_lr:.6f} -> {new_lr:.6f}")
                
        # 随着训练进度逐渐降低学习率
        if self.episode_count > 500:
            decay_factor = 0.9999
            self.learning_rate *= decay_factor
    
    def update_learning_rate_by_phase(self, phase_name: str, episode_in_phase: int, batch_win_rate: float = None):
        """分阶段的学习率控制"""
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
        
        current_lr = self.learning_rate
        
        # 阶段内强制稳定期
        if episode_in_phase < force_until:
            if episode_in_phase > 0 and episode_in_phase % update_freq == 0:
                # 强制期内更温和的调整
                if batch_win_rate is not None:
                    if batch_win_rate < 0.2:
                        multiplier = stable_range[1]
                        print(f"{phase_name}: 强制期，批次胜率过低({batch_win_rate:.3f})，轻微提高学习率")
                    elif batch_win_rate > 0.7:
                        multiplier = stable_range[0]
                        print(f"{phase_name}: 强制期，批次胜率较高({batch_win_rate:.3f})，轻微降低学习率")
                    else:
                        multiplier = 1.0
                    
                    new_lr = current_lr * multiplier
                    new_lr = max(self.lr_min, min(new_lr, self.lr_max))
                    
                    if abs(new_lr - current_lr) > 1e-6:
                        self.learning_rate = new_lr
                        print(f"{phase_name}: 强制期学习率调整: {current_lr:.6f} -> {new_lr:.6f}")
            return
        
        # 阶段内自适应调整期
        if episode_in_phase > 0 and episode_in_phase % update_freq == 0:
            old_lr = current_lr
            
            if self.adaptive_lr and batch_win_rate is not None:
                if batch_win_rate < 0.2:
                    # 胜率很低，大幅提高学习率
                    multiplier = adaptive_range[1]
                    print(f"{phase_name}: 批次胜率过低({batch_win_rate:.3f})，提高学习率")
                elif batch_win_rate < 0.35:
                    # 胜率较低，适度提高学习率
                    multiplier = (adaptive_range[1] + 1.0) / 2
                elif batch_win_rate > 0.7:
                    # 胜率很高，降低学习率
                    multiplier = adaptive_range[0]
                    print(f"{phase_name}: 批次胜率较高({batch_win_rate:.3f})，降低学习率")
                elif batch_win_rate > 0.55:
                    # 胜率较高，轻微降低学习率
                    multiplier = (adaptive_range[0] + 1.0) / 2
                else:
                    # 正常范围，保持稳定
                    multiplier = 1.0
                
                new_lr = current_lr * multiplier
                new_lr = max(self.lr_min, min(new_lr, self.lr_max))
                
                if abs(new_lr - current_lr) > 1e-6:
                    self.learning_rate = new_lr
                    print(f"{phase_name}: 基于批次胜率调整学习率: {current_lr:.6f} -> {new_lr:.6f}")
            
            # 记录学习率变化
            if abs(self.learning_rate - old_lr) > 1e-6 and episode_in_phase % 100 == 0:
                win_rate_str = f"{batch_win_rate:.3f}" if batch_win_rate is not None else "N/A"
                print(f"{phase_name}: Episode {episode_in_phase}, "
                      f"学习率: {old_lr:.6f} -> {self.learning_rate:.6f}, "
                      f"批次胜率: {win_rate_str}")
    
    def take_turn(self, board: Board) -> bool:
        """为游戏集成实现的take_turn方法"""
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
            if piece and not piece.revealed:
                piece.reveal()
                return True
                
        elif action_type == "move" and pos2 is not None:
            if board.try_move(pos1, pos2):
                return True
        
        return False
    
    def enable_adaptive_lr(self):
        """启用自适应学习率"""
        self.adaptive_lr = True
        print("已启用自适应学习率调整")
    
    def disable_adaptive_lr(self):
        """禁用自适应学习率"""
        self.adaptive_lr = False
        print("已禁用自适应学习率调整")
        
    def save_model(self, filename: str):
        """保存模型"""
        data = {
            'q_table': self.q_table,
            'learning_rate': self.learning_rate,
            'initial_learning_rate': self.initial_learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'stats': self.training_stats,
            'episode_count': self.episode_count
        }
        save_model_data(data, f"{filename}.pkl")
        print(f"QL模型已保存到: {filename}.pkl")
    
    def load_model(self, filename: str) -> bool:
        """加载模型"""
        data = load_model_data(f"{filename}.pkl")
        if data:
            try:
                self.q_table = data.get('q_table', {})
                self.learning_rate = data.get('learning_rate', self.learning_rate)
                self.initial_learning_rate = data.get('initial_learning_rate', self.initial_learning_rate)
                self.discount_factor = data.get('discount_factor', self.discount_factor)
                self.epsilon = data.get('epsilon', self.epsilon)
                self.epsilon_min = data.get('epsilon_min', self.epsilon_min)
                self.epsilon_decay = data.get('epsilon_decay', self.epsilon_decay)
                self.training_stats = data.get('stats', self.training_stats)
                self.episode_count = data.get('episode_count', 0)
                print("QL模型加载成功!")
                return True
            except Exception as e:
                print(f"加载QL模型失败: {e}")
        return False

class QLearningTrainer(BaseTrainer):
    """Q-learning训练器 - 改进版"""
    
    def __init__(self, agent: QLearningAgent, opponent_agent: Player, **kwargs):
        super().__init__(agent, opponent_agent, **kwargs)
    
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
        
        self.agent.update_q_value(board_before, action, reward, board_after, next_valid_actions, result)
    
    def save_model(self, filename: str):
        """保存模型"""
        self.agent.save_model(filename)
