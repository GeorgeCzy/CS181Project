import numpy as np
from sklearn.linear_model import SGDRegressor
import random
import copy
import time
import os
import argparse
from datetime import datetime
from collections import deque
from typing import Tuple, List, Optional, Dict, Any
from base import Player, BaseTrainer, GameEnvironment, Board
from utils import save_model_data, load_model_data, FeatureExtractor


class ApproximateQAgent(Player):
    def __init__(
        self,
        player_id: int,
        learning_rate: float = 0.01,
        discount_factor: float = 0.5,
        discount_max: float = 0.99,
        discount_growth: float = 0.001,
        epsilon: float = 0.9,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        super().__init__(player_id)
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate  # 保存初始学习率
        self.discount_factor = discount_factor
        self.discount_max = discount_max
        self.discount_growth = discount_growth
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episode_count = 0

        # 学习率调整参数
        self.lr_min = 0.001
        self.lr_max = learning_rate * 3
        self.adaptive_lr = True
        self.lr_adjustment_frequency = 50

        # 训练模式控制
        self.training_mode = True
        self.test_mode_epsilon = 0.0

        # 特征提取器
        self.feature_extractor = FeatureExtractor()

        # 使用SGD回归器作为函数逼近器
        self.q_function = SGDRegressor(
            loss="huber",
            learning_rate="adaptive",
            eta0=learning_rate,
            power_t=0.25,
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            warm_start=True,
        )

        # 用于存储训练数据
        self.training_data = []
        self.batch_size = 32
        self.is_fitted = False

        # 训练统计
        self.losses = []
        self.q_values_history = []

        self.phase_configs = {
            "phase1": {
                "epsilon_force_until": 300,
                "epsilon_min": 0.8,
                "epsilon_decay_rate": 0.9996,
                "lr_force_until": 200,
                "lr_update_frequency": 100,
                "lr_stable_range": (0.95, 1.1),
                "lr_adaptive_range": (0.8, 1.5),
                "description": "基础学习阶段",
            },
            "phase2": {
                "epsilon_force_until": 200,
                "epsilon_min": 0.4,
                "epsilon_decay_rate": 0.9995,
                "lr_force_until": 150,
                "lr_update_frequency": 100,
                "lr_stable_range": (0.9, 1.1),
                "lr_adaptive_range": (0.7, 1.8),
                "description": "进阶学习阶段",
            },
            "phase3": {
                "epsilon_force_until": 100,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay_rate": self.epsilon_decay,
                "lr_force_until": 50,
                "lr_update_frequency": 150,
                "lr_stable_range": (0.95, 1.05),
                "lr_adaptive_range": (0.8, 1.2),
                "description": "策略精炼阶段",
            },
        }

        self.current_phase = "phase1"
        self.episode_in_phase = 0

        self.ai_type = f"AQ (ε={epsilon:.2f})" if epsilon > 0 else "AQ (Trained)"

    def set_phase(self, phase_name: str):
        """设置当前训练阶段"""
        if phase_name in self.phase_configs:
            self.current_phase = phase_name
            self.episode_in_phase = 0
            print(f"ApproximateQAgent: 切换到 {phase_name}")
        else:
            print(f"警告: 未知阶段 {phase_name}, 保持当前阶段 {self.current_phase}")

    def decay_epsilon_by_phase(
        self,
        phase_name: str,
        episode_in_phase: int,
        batch_win_rate: float = None,
        recent_win_rate: float = None,
    ):
        """改进的分阶段epsilon衰减控制 - 接收近期胜率参数"""
        if phase_name not in self.phase_configs:
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
            self.epsilon = max(min_epsilon, self.epsilon * 0.9998)
            if episode_in_phase % 100 == 0:
                recent_str = (
                    f"近期={recent_win_rate:.3f}" if recent_win_rate is not None else ""
                )
                batch_str = (
                    f"批次={batch_win_rate:.3f}" if batch_win_rate is not None else ""
                )
                print(
                    f"{phase_name}: 强制探索期 ({episode_in_phase}/{force_until}), "
                    f"保持高epsilon={self.epsilon:.3f}, 胜率({recent_str}, {batch_str})"
                )
            return

        # 阶段内自适应衰减期
        old_epsilon = self.epsilon

        # 优先使用最近100回合的胜率，否则使用批次胜率
        win_rate_to_use = (
            recent_win_rate if recent_win_rate is not None else batch_win_rate
        )

        if win_rate_to_use is not None:
            # 胜率低于40%时，增加epsilon
            if win_rate_to_use < 0.4:
                # 胜率越低，epsilon增加越多
                actual_decay = 1.01 + (0.4 - win_rate_to_use) * 0.1
                if episode_in_phase % 50 == 0:
                    print(
                        f"{phase_name}: 最近胜率过低({win_rate_to_use:.3f})，提高epsilon以增加探索"
                    )
            # 胜率40%-60%之间，缓慢衰减
            elif win_rate_to_use < 0.6:
                actual_decay = 0.999
                if episode_in_phase % 100 == 0:
                    print(
                        f"{phase_name}: 胜率适中({win_rate_to_use:.3f})，缓慢衰减epsilon"
                    )
            # 胜率高于60%，可以加速衰减
            else:
                actual_decay = decay_rate
                if episode_in_phase % 50 == 0:
                    print(
                        f"{phase_name}: 胜率良好({win_rate_to_use:.3f})，正常衰减epsilon"
                    )
        else:
            actual_decay = decay_rate

        # 应用衰减或增长因子
        self.epsilon = min(0.9, max(min_epsilon, self.epsilon * actual_decay))

        # 记录epsilon变化
        if abs(old_epsilon - self.epsilon) > 0.01 or episode_in_phase % 100 == 0:
            win_rate_str = (
                f"{win_rate_to_use:.3f}" if win_rate_to_use is not None else "N/A"
            )
            print(
                f"{phase_name}: Episode {episode_in_phase}, "
                f"epsilon: {old_epsilon:.3f} → {self.epsilon:.3f}, "
                f"最近胜率: {win_rate_str}"
            )

    def update_learning_rate_by_phase(
        self,
        phase_name: str,
        episode_in_phase: int,
        batch_win_rate: float = None,
        recent_win_rate: float = None,
    ):
        """分阶段的学习率控制 - 修复ApproximateQ的学习率更新"""
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

        win_rate_to_use = (
            recent_win_rate if recent_win_rate is not None else batch_win_rate
        )

        # 阶段内强制稳定期
        if episode_in_phase < force_until:
            if episode_in_phase > 0 and episode_in_phase % update_freq == 0:
                # 强制期内更温和的调整 - 优先使用近期胜率
                win_rate_to_use = (
                    recent_win_rate if recent_win_rate is not None else batch_win_rate
                )
                if win_rate_to_use is not None:
                    if win_rate_to_use < 0.3:
                        multiplier = stable_range[1]  # 胜率低，提高学习率
                        print(
                            f"{phase_name}: 强制期，最近胜率过低({win_rate_to_use:.3f})，轻微提高学习率"
                        )
                    elif win_rate_to_use > 0.7:
                        multiplier = stable_range[0]  # 胜率高，降低学习率
                        print(
                            f"{phase_name}: 强制期，最近胜率较高({win_rate_to_use:.3f})，轻微降低学习率"
                        )
                    else:
                        multiplier = 1.0  # 保持稳定

                    new_lr = current_lr * multiplier
                    new_lr = max(self.lr_min, min(new_lr, self.lr_max))

                    if abs(new_lr - current_lr) > 1e-6:
                        # ApproximateQ使用SGDRegressor，更新eta0属性
                        self.q_function.eta0 = new_lr
                        recent_str = (
                            f"近期={recent_win_rate:.3f}"
                            if recent_win_rate is not None
                            else ""
                        )
                        batch_str = (
                            f"批次={batch_win_rate:.3f}"
                            if batch_win_rate is not None
                            else ""
                        )
                        win_info = f"胜率({recent_str}, {batch_str})"
                        print(
                            f"{phase_name}: 强制期学习率调整: {current_lr:.6f} → {new_lr:.6f}, {win_info}"
                        )
            return

        # 阶段内自适应调整期
        if episode_in_phase > 0 and episode_in_phase % update_freq == 0:
            old_lr = current_lr

            if self.adaptive_lr:
                # 优先使用近期胜率，其次使用批次胜率
                win_rate_to_use = (
                    recent_win_rate if recent_win_rate is not None else batch_win_rate
                )

                if win_rate_to_use is not None:
                    if win_rate_to_use < 0.25:
                        # 胜率很低，大幅提高学习率
                        multiplier = adaptive_range[1]
                        print(
                            f"{phase_name}: 最近胜率过低({win_rate_to_use:.3f})，提高学习率"
                        )
                    elif win_rate_to_use < 0.4:
                        # 胜率较低，适度提高学习率
                        multiplier = (adaptive_range[1] + 1.0) / 2
                    elif win_rate_to_use > 0.75:
                        # 胜率很高，降低学习率
                        multiplier = adaptive_range[0]
                        print(
                            f"{phase_name}: 最近胜率较高({win_rate_to_use:.3f})，降低学习率"
                        )
                    elif win_rate_to_use > 0.6:
                        # 胜率较高，轻微降低学习率
                        multiplier = (adaptive_range[0] + 1.0) / 2
                    else:
                        # 正常范围，保持稳定
                        multiplier = 1.0

                    new_lr = current_lr * multiplier
                    new_lr = max(self.lr_min, min(new_lr, self.lr_max))

                    if abs(new_lr - current_lr) > 1e-6:
                        # ApproximateQ使用SGDRegressor，更新eta0属性
                        self.q_function.eta0 = new_lr
                        recent_str = (
                            f"近期={recent_win_rate:.3f}"
                            if recent_win_rate is not None
                            else ""
                        )
                        batch_str = (
                            f"批次={batch_win_rate:.3f}"
                            if batch_win_rate is not None
                            else ""
                        )
                        win_info = f"胜率({recent_str}, {batch_str})"
                        print(
                            f"{phase_name}: 基于胜率调整学习率: {current_lr:.6f} → {new_lr:.6f}, {win_info}"
                        )

            # 记录学习率变化
            if (
                abs(self.get_learning_rate() - old_lr) > 1e-6
                and episode_in_phase % 100 == 0
            ):
                recent_str = (
                    f"近期={recent_win_rate:.3f}"
                    if recent_win_rate is not None
                    else "N/A"
                )
                batch_str = (
                    f"批次={batch_win_rate:.3f}"
                    if batch_win_rate is not None
                    else "N/A"
                )
                print(
                    f"{phase_name}: Episode {episode_in_phase}, "
                    f"学习率: {old_lr:.6f} → {self.get_learning_rate():.6f}, "
                    f"胜率: 近期={recent_str}, 批次={batch_str}"
                )

    def get_learning_rate(self) -> float:
        """获取当前学习率"""
        return self.q_function.eta0

    def update_learning_rate(self, win_rate: float = None):
        """自适应学习率调整策略 - 修复optimizer引用错误"""
        if not hasattr(self, "training_stats"):
            return

        episodes = self.training_stats["episodes"]
        current_lr = self.get_learning_rate()

        # 每隔一定episode数进行自适应调整
        if (
            self.adaptive_lr
            and episodes % self.lr_adjustment_frequency == 0
            and win_rate is not None
        ):

            # 基于表现的自适应调整
            if win_rate < 0.25:
                # 表现很差，提高学习率
                multiplier = 1.3
                print(f"表现较差 (胜率={win_rate:.3f})，提高学习率")

            elif win_rate < 0.4:
                # 表现不佳，适度提高学习率
                multiplier = 1.1
                print(f"表现一般 (胜率={win_rate:.3f})，轻微提高学习率")

            elif win_rate > 0.8:
                # 表现很好，降低学习率以稳定策略
                multiplier = 0.85
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

            # 更新学习率 - 使用SGDRegressor的eta0属性
            if abs(new_lr - current_lr) > 1e-6:
                self.q_function.eta0 = new_lr
                print(f"学习率调整: {current_lr:.6f} -> {new_lr:.6f}")

    def enable_adaptive_lr(self):
        """启用自适应学习率"""
        self.adaptive_lr = True
        print("已启用自适应学习率调整")

    def disable_adaptive_lr(self):
        """禁用自适应学习率"""
        self.adaptive_lr = False
        print("已禁用自适应学习率调整")

    def decay_epsilon(self, win_rate: float = None):
        """改进的epsilon衰减策略"""
        if win_rate is not None:
            if win_rate < 0.3:
                # 表现差，保持较高探索
                decay = 0.9995
            elif win_rate > 0.6 and self.episode_count > 50:
                # 表现好，加快收敛
                decay = 0.992
            else:
                decay = self.epsilon_decay
        else:
            # 基于episode数的自适应衰减
            if self.episode_count < 100:
                decay = 0.999  # 早期慢衰减
            elif self.episode_count < 500:
                decay = self.epsilon_decay
            else:
                decay = 0.998  # 后期更慢衰减

        self.epsilon = max(self.epsilon_min, self.epsilon * decay)
        self.episode_count += 1

    def update_discount_factor(self, win_rate: float = None):
        """动态更新折扣因子"""
        if win_rate is not None:
            if win_rate < 0.3:
                growth = self.discount_growth * 0.5
            elif win_rate > 0.6 and self.episode_count > 50:
                growth = self.discount_growth * 2.0
            else:
                growth = self.discount_growth

            new_discount = self.discount_factor + growth
            self.discount_factor = min(self.discount_max, new_discount)
        else:
            progress = min(1.0, self.episode_count / 200)
            self.discount_factor = (
                self.discount_factor
                + (self.discount_max - self.discount_factor) * progress
            )

    def get_q_value(self, board: Board, action: Tuple) -> float:
        """获取状态-动作的Q值 - 修复动作解包错误"""
        try:
            # 标准化动作格式为3元组
            if len(action) == 2:
                action_type, pos1 = action
                if action_type == "reveal":
                    action = (action_type, pos1, None)
                else:
                    # 对于移动动作，如果缺少目标位置，返回默认值
                    return 0.0
            elif len(action) == 3:
                # 已经是3元组格式，直接使用
                pass
            else:
                # 未知格式，返回默认值
                return 0.0

            features = self.feature_extractor.extract_features(
                board, self.player_id, action
            )

            if len(features) == 0 or not self.is_fitted:
                return 0.0

            features_2d = features.reshape(1, -1)
            q_value = self.q_function.predict(features_2d)[0]

            # 记录Q值历史用于分析
            self.q_values_history.append(q_value)
            if len(self.q_values_history) > 1000:
                self.q_values_history.pop(0)

            return q_value

        except Exception as e:
            print(f"计算Q值时出错: {e}, 动作: {action}")
            return 0.0
    
    def choose_action(self, board: Board, valid_actions: List[Tuple]) -> Tuple:
        """使用epsilon-greedy策略选择动作 - 修复动作格式问题"""
        if not valid_actions:
            all_actions = board.get_all_possible_moves(self.player_id)
            if not all_actions:
                return ("reveal", (0, 0), None)
            valid_actions = all_actions

        # 标准化动作格式
        normalized_actions = []
        for action in valid_actions:
            if len(action) == 2:
                action_type, pos1 = action
                if action_type == "reveal":
                    normalized_actions.append((action_type, pos1, None))
                else:
                    # 移动动作缺少目标位置，跳过
                    print(f"警告: 移动动作缺少目标位置: {action}")
                    continue
            elif len(action) == 3:
                normalized_actions.append(action)
            else:
                print(f"警告: 未知动作格式: {action}")

        if not normalized_actions:
            return ("reveal", (0, 0), None)

        # 在测试模式下强制贪心策略
        if not self.training_mode:
            epsilon_to_use = 0.0
        else:
            epsilon_to_use = self.epsilon

        if random.random() < epsilon_to_use:
            return random.choice(normalized_actions)
        else:
            best_action = None
            best_q_value = float("-inf")

            for action in normalized_actions:
                q_value = self.get_q_value(board, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action

            return best_action if best_action else random.choice(normalized_actions)

    def update_q_value(
        self,
        board_before: Board,
        action: Tuple,
        reward: float,
        board_after: Board,
        next_valid_actions: List[Tuple],
        result: int,
    ):
        """更新Q值函数 - 修复动作格式问题"""
        # 标准化动作格式
        if len(action) == 2:
            action_type, pos1 = action
            if action_type == "reveal":
                action = (action_type, pos1, None)
            else:
                print(f"警告: 更新Q值时动作格式错误: {action}")
                return
        
        features = self.feature_extractor.extract_features(
            board_before, self.player_id, action
        )

        if result != -1 or not next_valid_actions:
            target_q = reward
        else:
            # 标准化下一步动作格式
            normalized_next_actions = []
            for next_action in next_valid_actions:
                if len(next_action) == 2:
                    action_type, pos1 = next_action
                    if action_type == "reveal":
                        normalized_next_actions.append((action_type, pos1, None))
                elif len(next_action) == 3:
                    normalized_next_actions.append(next_action)
            
            if normalized_next_actions:
                max_next_q = max(
                    self.get_q_value(board_after, next_action)
                    for next_action in normalized_next_actions
                )
            else:
                max_next_q = 0.0
                
            target_q = reward + self.discount_factor * max_next_q

        self.training_data.append((features, target_q))

        if len(self.training_data) >= self.batch_size:
            self._update_model()

    def _update_model(self):
        """更新函数逼近模型"""
        if not self.training_data:
            return

        try:
            feature_lengths = [len(data[0]) for data in self.training_data]
            if len(set(feature_lengths)) > 1:
                print(f"警告：特征向量长度不一致: {set(feature_lengths)}")
                expected_length = max(set(feature_lengths), key=feature_lengths.count)
                filtered_data = [
                    (features, target)
                    for features, target in self.training_data
                    if len(features) == expected_length
                ]
                if not filtered_data:
                    self.training_data = []
                    return
                self.training_data = filtered_data

            X = np.array([data[0] for data in self.training_data])
            y = np.array([data[1] for data in self.training_data])

            if not self.is_fitted:
                self.q_function.fit(X, y)
                self.is_fitted = True
            else:
                self.q_function.partial_fit(X, y)

            # 计算并记录损失
            if len(y) > 0:
                y_pred = self.q_function.predict(X)
                loss = np.mean((y - y_pred) ** 2)
                self.losses.append(loss)
                if len(self.losses) > 1000:
                    self.losses.pop(0)

            self.training_data = []

        except Exception as e:
            print(f"模型更新出错: {e}")
            self.training_data = []

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
            if hasattr(self, "_saved_epsilon"):
                self.epsilon = self._saved_epsilon
                print(f"切换到训练模式，epsilon: {self.epsilon}")

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

    def save_model(self, filename: str):
        """保存模型"""
        data = {
            "q_function": self.q_function,
            "is_fitted": self.is_fitted,
            "feature_extractor": self.feature_extractor,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "initial_learning_rate": self.initial_learning_rate,
                "discount_factor": self.discount_factor,
                "discount_max": self.discount_max,
                "discount_growth": self.discount_growth,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "batch_size": self.batch_size,
            },
            "training_stats": self.training_stats,
            "episode_count": self.episode_count,
        }
        save_model_data(data, f"{filename}.pkl")
        print(f"AQ模型已保存到: {filename}.pkl")

    def load_model(self, filename: str) -> bool:
        """加载模型"""
        data = load_model_data(f"{filename}.pkl")
        if data:
            try:
                self.q_function = data.get("q_function", self.q_function)
                self.is_fitted = data.get("is_fitted", False)
                self.feature_extractor = data.get(
                    "feature_extractor", self.feature_extractor
                )

                # 加载超参数
                hyperparams = data.get("hyperparameters", {})
                self.learning_rate = hyperparams.get(
                    "learning_rate", self.learning_rate
                )
                self.initial_learning_rate = hyperparams.get(
                    "initial_learning_rate", self.initial_learning_rate
                )
                self.discount_factor = hyperparams.get(
                    "discount_factor", self.discount_factor
                )
                self.discount_max = hyperparams.get("discount_max", self.discount_max)
                self.discount_growth = hyperparams.get(
                    "discount_growth", self.discount_growth
                )
                self.epsilon = hyperparams.get("epsilon", self.epsilon)
                self.epsilon_min = hyperparams.get("epsilon_min", self.epsilon_min)
                self.epsilon_decay = hyperparams.get(
                    "epsilon_decay", self.epsilon_decay
                )
                self.batch_size = hyperparams.get("batch_size", self.batch_size)

                # 加载训练统计
                self.training_stats = data.get("training_stats", self.training_stats)
                self.episode_count = data.get("episode_count", 0)

                print("AQ模型加载成功!")
                return True
            except Exception as e:
                print(f"加载AQ模型失败: {e}")
        return False


class ApproximateQTrainer(BaseTrainer):
    """基于函数逼近的Q-learning训练器 - 改进版"""

    def __init__(self, agent: ApproximateQAgent, opponent_agent: Player, **kwargs):
        super().__init__(agent, opponent_agent, **kwargs)

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
        if result == -1:
            next_valid_actions = board_after.get_all_possible_moves(
                self.agent.player_id
            )
        else:
            next_valid_actions = []

        self.agent.update_q_value(
            board_before, action, reward, board_after, next_valid_actions, result
        )

    def save_model(self, filename: str):
        """保存模型"""
        self.agent.save_model(filename)
