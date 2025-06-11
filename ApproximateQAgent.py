import numpy as np
from sklearn.linear_model import SGDRegressor
import random
import copy
import time
import os
import argparse
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any
from base import Player, BaseTrainer, GameEnvironment, Board
from utils import save_model_data, load_model_data, FeatureExtractor

class ApproximateQAgent(Player):
    def __init__(self, player_id: int, learning_rate: float = 0.01, 
                 discount_factor: float = 0.5,
                 discount_max: float = 0.99,
                 discount_growth: float = 0.001,
                 epsilon: float = 0.9,
                 epsilon_min: float = 0.01, 
                 epsilon_decay: float = 0.995):
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
            loss='huber',
            learning_rate='adaptive',
            eta0=learning_rate,
            power_t=0.25,
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            warm_start=True
        )
        
        # 用于存储训练数据
        self.training_data = []
        self.batch_size = 32
        self.is_fitted = False
        
        # 训练统计
        self.losses = []
        self.q_values_history = []
        
        self.ai_type = f"AQ (ε={epsilon:.2f})" if epsilon > 0 else "AQ (Trained)"

    def get_learning_rate(self) -> float:
        """获取当前学习率"""
        return self.q_function.eta0

    def update_learning_rate(self, win_rate: float = None):
        """自适应学习率调整策略"""
        if not hasattr(self, 'training_stats'):
            return
            
        episodes = self.training_stats['episodes']
        current_lr = self.get_learning_rate()
        
        # 每隔一定episode数进行自适应调整
        if self.adaptive_lr and episodes % self.lr_adjustment_frequency == 0 and win_rate is not None:
            
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
            
            # 更新学习率
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
            self.discount_factor = self.discount_factor + \
                (self.discount_max - self.discount_factor) * progress
    
    def get_q_value(self, board: Board, action: Tuple) -> float:
        """获取状态-动作的Q值"""
        try:
            features = self.feature_extractor.extract_features(board, self.player_id, action)
            
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
            print(f"计算Q值时出错: {e}")
            return 0.0
    
    def choose_action(self, board: Board, valid_actions: List[Tuple]) -> Tuple:
        """使用epsilon-greedy策略选择动作"""
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
                    continue
            elif len(action) == 3:
                normalized_actions.append(action)
        
        if not normalized_actions:
            return ("reveal", (0, 0), None)
        
        if random.random() < self.epsilon:
            return random.choice(normalized_actions)
        else:
            best_action = None
            best_q_value = float('-inf')
            
            for action in normalized_actions:
                q_value = self.get_q_value(board, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            return best_action if best_action else random.choice(normalized_actions)
    
    def update_q_value(self, board_before: Board, action: Tuple, reward: float, 
                      board_after: Board, next_valid_actions: List[Tuple], result: int):
        """更新Q值函数"""
        features = self.feature_extractor.extract_features(board_before, self.player_id, action)
        
        if result != -1 or not next_valid_actions:
            target_q = reward
        else:
            max_next_q = max([self.get_q_value(board_after, next_action) 
                             for next_action in next_valid_actions] or [0.0])
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
                filtered_data = [(features, target) for features, target in self.training_data 
                               if len(features) == expected_length]
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
            if hasattr(self, '_saved_epsilon'):
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
            if piece and piece.player == self.player_id and not piece.revealed:
                piece.revealed = True
                return True
                
        elif action_type == "move" and pos2 is not None:
            if board.try_move(pos1, pos2):
                return True
        
        return False
    
    def save_model(self, filename: str):
        """保存模型"""
        data = {
            'q_function': self.q_function,
            'is_fitted': self.is_fitted,
            'feature_extractor': self.feature_extractor,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'initial_learning_rate': self.initial_learning_rate,
                'discount_factor': self.discount_factor,
                'discount_max': self.discount_max,
                'discount_growth': self.discount_growth,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size
            },
            'training_stats': self.training_stats,
            'episode_count': self.episode_count
        }
        save_model_data(data, f"{filename}.pkl")
        print(f"AQ模型已保存到: {filename}.pkl")
    
    def load_model(self, filename: str) -> bool:
        """加载模型"""
        data = load_model_data(f"{filename}.pkl")
        if data:
            try:
                self.q_function = data.get('q_function', self.q_function)
                self.is_fitted = data.get('is_fitted', False)
                self.feature_extractor = data.get('feature_extractor', self.feature_extractor)
                
                # 加载超参数
                hyperparams = data.get('hyperparameters', {})
                self.learning_rate = hyperparams.get('learning_rate', self.learning_rate)
                self.initial_learning_rate = hyperparams.get('initial_learning_rate', self.initial_learning_rate)
                self.discount_factor = hyperparams.get('discount_factor', self.discount_factor)
                self.discount_max = hyperparams.get('discount_max', self.discount_max)
                self.discount_growth = hyperparams.get('discount_growth', self.discount_growth)
                self.epsilon = hyperparams.get('epsilon', self.epsilon)
                self.epsilon_min = hyperparams.get('epsilon_min', self.epsilon_min)
                self.epsilon_decay = hyperparams.get('epsilon_decay', self.epsilon_decay)
                self.batch_size = hyperparams.get('batch_size', self.batch_size)
                
                # 加载训练统计
                self.training_stats = data.get('training_stats', self.training_stats)
                self.episode_count = data.get('episode_count', 0)
                
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



# def train_with_curriculum(aq_agent, opponent, episodes=2000, data_manager=None, print_interval=50):
#     """课程式训练 - AQ版本"""
#     trainer = ApproximateQTrainer(aq_agent, opponent)
    
#     total_episodes = episodes
    
#     # 添加计时记录
#     training_start_time = time.time()
#     phase_times = {}
    
#     # 阶段1: 快速探索 (30%)
#     phase1_episodes = int(total_episodes * 0.3)
#     print(f"阶段1: 快速探索学习 ({phase1_episodes} episodes) - 每{print_interval}回合输出进度")
#     print(f"epsilon固定在: {aq_agent.epsilon:.3f}")
#     print(f"初始学习率: {aq_agent.get_learning_rate():.6f}")
#     print(f"初始折扣因子: {aq_agent.discount_factor:.3f}")
    
#     # 保存原始参数
#     original_epsilon_decay = aq_agent.epsilon_decay
#     original_discount_growth = aq_agent.discount_growth
    
#     # 阶段1: 禁用epsilon衰减，减缓折扣因子增长
#     aq_agent.epsilon_decay = 1.0
#     aq_agent.discount_growth = aq_agent.discount_growth * 0.5
    
#     # 阶段1训练
#     phase1_start_time = time.time()
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
#         if result == 0:
#             batch_wins += 1
#         elif result == 1:
#             batch_loses += 1
#         else:
#             batch_draws += 1
        
#         # 定期输出进度
#         if episode % print_interval == 0 or episode == phase1_episodes - 1:
#             batch_episodes = episode - batch_start_episode + 1
#             batch_win_rate = (batch_wins + batch_draws / 2) / batch_episodes
#             avg_steps = sum(batch_steps) / len(batch_steps) if batch_steps else 0
            
#             current_time = time.time()
#             if episode == 0:
#                 batch_time = current_time - phase1_start_time
#             else:
#                 episodes_since_start = episode + 1
#                 total_elapsed = current_time - phase1_start_time
#                 if episode >= print_interval:
#                     avg_time_per_episode = total_elapsed / episodes_since_start
#                     batch_time = print_interval * avg_time_per_episode
#                 else:
#                     batch_time = total_elapsed
            
#             avg_time_per_episode = batch_time / batch_episodes if batch_episodes > 0 else 0
            
#             param_info = f", ε = {aq_agent.epsilon:.3f}, lr = {aq_agent.get_learning_rate():.6f}, γ = {aq_agent.discount_factor:.3f}"
#             time_info = f", 用时 = {batch_time:.1f}s, 平均 = {avg_time_per_episode:.2f}s/ep"
            
#             # 显示Q值和损失信息
#             q_info = ""
#             if aq_agent.q_values_history:
#                 recent_q = np.mean(aq_agent.q_values_history[-20:])
#                 q_info = f", Q值 = {recent_q:.3f}"
            
#             loss_info = ""
#             if aq_agent.losses:
#                 recent_loss = np.mean(aq_agent.losses[-10:])
#                 loss_info = f", 损失 = {recent_loss:.4f}"
            
#             print(f"阶段1 - 回合 {episode}: 奖励 = {total_reward:.2f}, 步数 = {steps}, "
#                   f"胜 = {batch_wins}, 负 = {batch_loses}, 平 = {batch_draws}, "
#                   f"批次胜率 = {batch_win_rate:.3f}, 平均步长 = {avg_steps:.1f}{param_info}{q_info}{loss_info}{time_info}")
            
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
        
#         # 学习率调整
#         if episode > 0 and episode % 100 == 0:
#             recent_wins = 0
#             recent_episodes = min(100, episode + 1)
#             for i in range(max(0, episode - recent_episodes + 1), episode + 1):
#                 if (data_manager and 
#                     len(data_manager.current_session['training_history']['wins']) > i and 
#                     data_manager.current_session['training_history']['wins'][i] == 1):
#                     recent_wins += 1
#             recent_win_rate = recent_wins / recent_episodes
            
#             aq_agent.update_learning_rate(recent_win_rate)
        
#         if data_manager:
#             data_manager.log_episode(
#                 episode=episode,
#                 reward=total_reward,
#                 result=result,
#                 learning_rate=aq_agent.get_learning_rate(),
#                 epsilon=aq_agent.epsilon,
#                 loss=aq_agent.losses[-1] if aq_agent.losses else None,
#                 phase='phase1_exploration',
#                 steps=steps,
#                 episode_time=episode_time
#             )
    
#     phase1_end_time = time.time()
#     phase_times['phase1'] = phase1_end_time - phase1_start_time
#     print(f"阶段1完成! 当前epsilon: {aq_agent.epsilon:.3f}, 学习率: {aq_agent.get_learning_rate():.6f}, 折扣因子: {aq_agent.discount_factor:.3f}")
#     print(f"阶段1总耗时: {phase_times['phase1']:.1f}秒, 平均: {phase_times['phase1']/phase1_episodes:.2f}秒/回合")
    
#     # 阶段2: 平衡学习 (50%)
#     phase2_episodes = int(total_episodes * 0.5)
#     print(f"\n阶段2: 平衡学习 ({phase2_episodes} episodes)")
#     print(f"启用缓慢epsilon衰减，当前epsilon: {aq_agent.epsilon:.3f}")
    
#     # 启用缓慢epsilon衰减，恢复正常折扣因子增长
#     aq_agent.epsilon_decay = 0.9995
#     aq_agent.discount_growth = original_discount_growth
    
#     phase2_start_time = time.time()
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
        
#         # 在阶段2调用epsilon衰减和折扣因子更新
#         aq_agent.decay_epsilon()
#         aq_agent.update_discount_factor()
        
#         # 记录批次数据
#         batch_steps.append(steps)
#         if result == 0:
#             batch_wins += 1
#         elif result == 1:
#             batch_loses += 1
#         else:
#             batch_draws += 1
        
#         # 定期输出进度
#         if episode % print_interval == 0 or episode == phase2_episodes - 1:
#             batch_episodes = episode - batch_start_episode + 1
#             batch_win_rate = (batch_wins + batch_draws / 2) / batch_episodes
#             avg_steps = sum(batch_steps) / len(batch_steps) if batch_steps else 0
            
#             current_time = time.time()
#             phase2_elapsed = current_time - phase2_start_time
#             if episode == 0:
#                 batch_time = phase2_elapsed
#             else:
#                 avg_time_per_episode = phase2_elapsed / (episode + 1)
#                 batch_time = batch_episodes * avg_time_per_episode
            
#             avg_time_per_episode = batch_time / batch_episodes if batch_episodes > 0 else 0
            
#             param_info = f", ε = {aq_agent.epsilon:.3f}, lr = {aq_agent.get_learning_rate():.6f}, γ = {aq_agent.discount_factor:.3f}"
#             time_info = f", 累计用时 = {phase2_elapsed:.1f}s, 平均 = {avg_time_per_episode:.2f}s/ep"
            
#             q_info = ""
#             if aq_agent.q_values_history:
#                 recent_q = np.mean(aq_agent.q_values_history[-20:])
#                 q_info = f", Q值 = {recent_q:.3f}"
            
#             loss_info = ""
#             if aq_agent.losses:
#                 recent_loss = np.mean(aq_agent.losses[-10:])
#                 loss_info = f", 损失 = {recent_loss:.4f}"
            
#             print(f"阶段2 - 回合 {episode}: 奖励 = {total_reward:.2f}, 步数 = {steps}, "
#                   f"胜 = {batch_wins}, 负 = {batch_loses}, 平 = {batch_draws}, "
#                   f"批次胜率 = {batch_win_rate:.3f}, 平均步长 = {avg_steps:.1f}{param_info}{q_info}{loss_info}{time_info}")
            
#             if data_manager:
#                 data_manager.log_batch_stats(
#                     batch_wins, batch_loses, batch_draws, 
#                     batch_win_rate, avg_steps
#                 )
            
#             batch_wins = 0
#             batch_loses = 0
#             batch_draws = 0
#             batch_steps = []
#             batch_start_episode = episode + 1
        
#         # 学习率调整
#         if episode > 0 and episode % 150 == 0:
#             recent_wins = 0
#             recent_episodes = min(150, episode + 1)
#             for i in range(max(0, phase1_episodes + episode - recent_episodes + 1), phase1_episodes + episode + 1):
#                 if (data_manager and 
#                     len(data_manager.current_session['training_history']['wins']) > i and 
#                     data_manager.current_session['training_history']['wins'][i] == 1):
#                     recent_wins += 1
#             recent_win_rate = recent_wins / recent_episodes
            
#             aq_agent.update_learning_rate(recent_win_rate)
        
#         if data_manager:
#             data_manager.log_episode(
#                 episode=phase1_episodes + episode,
#                 reward=total_reward,
#                 result=result,
#                 learning_rate=aq_agent.get_learning_rate(),
#                 epsilon=aq_agent.epsilon,
#                 loss=aq_agent.losses[-1] if aq_agent.losses else None,
#                 phase='phase2_balance',
#                 steps=steps,
#                 episode_time=episode_time
#             )
    
#     phase2_end_time = time.time()
#     phase_times['phase2'] = phase2_end_time - phase2_start_time
#     print(f"阶段2完成! 当前epsilon: {aq_agent.epsilon:.3f}, 学习率: {aq_agent.get_learning_rate():.6f}, 折扣因子: {aq_agent.discount_factor:.3f}")
#     print(f"阶段2总耗时: {phase_times['phase2']:.1f}秒, 平均: {phase_times['phase2']/phase2_episodes:.2f}秒/回合")
    
#     # 阶段3: 策略精炼 (20%)
#     phase3_episodes = total_episodes - phase1_episodes - phase2_episodes
#     print(f"\n阶段3: 策略精炼 ({phase3_episodes} episodes)")
#     print(f"恢复正常epsilon衰减，当前epsilon: {aq_agent.epsilon:.3f}")
    
#     # 恢复正常epsilon衰减
#     aq_agent.epsilon_decay = original_epsilon_decay
    
#     phase3_start_time = time.time()
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
        
#         # 在阶段3调用epsilon衰减和折扣因子更新
#         aq_agent.decay_epsilon()
#         aq_agent.update_discount_factor()
        
#         # 记录批次数据
#         batch_steps.append(steps)
#         if result == 0:
#             batch_wins += 1
#         elif result == 1:
#             batch_loses += 1
#         else:
#             batch_draws += 1
        
#         # 定期输出进度
#         if episode % print_interval == 0 or episode == phase3_episodes - 1:
#             batch_episodes = episode - batch_start_episode + 1
#             batch_win_rate = (batch_wins + batch_draws / 2) / batch_episodes
#             avg_steps = sum(batch_steps) / len(batch_steps) if batch_steps else 0
            
#             current_time = time.time()
#             phase3_elapsed = current_time - phase3_start_time
#             if episode == 0:
#                 batch_time = phase3_elapsed
#             else:
#                 avg_time_per_episode = phase3_elapsed / (episode + 1)
#                 batch_time = batch_episodes * avg_time_per_episode
            
#             avg_time_per_episode = batch_time / batch_episodes if batch_episodes > 0 else 0
            
#             param_info = f", ε = {aq_agent.epsilon:.3f}, lr = {aq_agent.get_learning_rate():.6f}, γ = {aq_agent.discount_factor:.3f}"
#             time_info = f", 累计用时 = {phase3_elapsed:.1f}s, 平均 = {avg_time_per_episode:.2f}s/ep"
            
#             q_info = ""
#             if aq_agent.q_values_history:
#                 recent_q = np.mean(aq_agent.q_values_history[-20:])
#                 q_info = f", Q值 = {recent_q:.3f}"
            
#             loss_info = ""
#             if aq_agent.losses:
#                 recent_loss = np.mean(aq_agent.losses[-10:])
#                 loss_info = f", 损失 = {recent_loss:.4f}"
            
#             print(f"阶段3 - 回合 {episode}: 奖励 = {total_reward:.2f}, 步数 = {steps}, "
#                   f"胜 = {batch_wins}, 负 = {batch_loses}, 平 = {batch_draws}, "
#                   f"批次胜率 = {batch_win_rate:.3f}, 平均步长 = {avg_steps:.1f}{param_info}{q_info}{loss_info}{time_info}")
            
#             if data_manager:
#                 data_manager.log_batch_stats(
#                     batch_wins, batch_loses, batch_draws, 
#                     batch_win_rate, avg_steps
#                 )
            
#             batch_wins = 0
#             batch_loses = 0
#             batch_draws = 0
#             batch_steps = []
#             batch_start_episode = episode + 1
        
#         # 学习率调整
#         if episode > 0 and episode % 200 == 0:
#             recent_wins = 0
#             recent_episodes = min(200, episode + 1)
#             for i in range(max(0, phase1_episodes + phase2_episodes + episode - recent_episodes + 1), 
#                           phase1_episodes + phase2_episodes + episode + 1):
#                 if (data_manager and 
#                     len(data_manager.current_session['training_history']['wins']) > i and 
#                     data_manager.current_session['training_history']['wins'][i] == 1):
#                     recent_wins += 1
#             recent_win_rate = recent_wins / recent_episodes
            
#             aq_agent.update_learning_rate(recent_win_rate)
        
#         if data_manager:
#             data_manager.log_episode(
#                 episode=phase1_episodes + phase2_episodes + episode,
#                 reward=total_reward,
#                 result=result,
#                 learning_rate=aq_agent.get_learning_rate(),
#                 epsilon=aq_agent.epsilon,
#                 loss=aq_agent.losses[-1] if aq_agent.losses else None,
#                 phase='phase3_refinement',
#                 steps=steps,
#                 episode_time=episode_time
#             )
    
#     phase3_end_time = time.time()
#     phase_times['phase3'] = phase3_end_time - phase3_start_time
    
#     # 总结训练时间
#     total_training_time = phase3_end_time - training_start_time
    
#     print(f"\n课程训练完成!")
#     print(f"最终epsilon: {aq_agent.epsilon:.3f}")
#     print(f"最终学习率: {aq_agent.get_learning_rate():.6f}")
#     print(f"最终折扣因子: {aq_agent.discount_factor:.3f}")
#     print(f"最终胜率: {aq_agent.get_stats()['win_rate']:.3f}")
    
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
        
#         if aq_agent.q_values_history:
#             print(f"平均Q值: {np.mean(aq_agent.q_values_history):.3f} ± {np.std(aq_agent.q_values_history):.3f}")
#         if aq_agent.losses:
#             print(f"平均损失: {np.mean(aq_agent.losses):.4f} ± {np.std(aq_agent.losses):.4f}")
    
#     return data_manager.current_session['training_history'] if data_manager else {}

# def train_or_load_model(force_retrain=False, episodes=2000, lr_strategy="adaptive", print_interval=50):
#     """训练或加载改进的Approximate Q模型 - 使用独立数据管理"""
#     from AgentFight import RandomPlayer
#     from training_data_manager import TrainingDataManager
    
#     model_name = "final_ApproximateQAgent"
    
#     # 创建改进的智能体
#     aq_agent = ApproximateQAgent(
#         player_id=0, 
#         learning_rate=0.02,
#         epsilon=0.9,
#         epsilon_min=0.05,
#         epsilon_decay=0.995,
#         discount_factor=0.3,
#         discount_max=0.95,
#         discount_growth=0.002
#     )
#     random_opponent = RandomPlayer(player_id=1)
    
#     # 设置学习率策略
#     if lr_strategy == "adaptive":
#         aq_agent.enable_adaptive_lr()
#         print("使用自适应学习率策略")
#     elif lr_strategy == "fixed":
#         aq_agent.disable_adaptive_lr()
#         print("使用固定学习率策略")
#     else:  # adaptive (默认)
#         aq_agent.enable_adaptive_lr()
#         print("使用自适应学习率策略")
    
#     # 检查是否存在已训练的模型
#     model_path = os.path.join("model_data", f"{model_name}.pkl")
#     model_exists = os.path.exists(model_path)
    
#     if model_exists and not force_retrain:
#         print(f"发现已训练的模型: {model_path}")
#         if aq_agent.load_model(model_name):
#             print("模型加载成功!")
#             aq_agent.epsilon = 0.0
#             aq_agent.ai_type = "AQ (Trained)"
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
#         session_name = f"AQ_{lr_strategy}_{episodes}eps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         data_manager.start_session(aq_agent, session_name)
        
#         # 使用课程式训练
#         combined_history = train_with_curriculum(aq_agent, random_opponent, episodes, data_manager, print_interval)
        
#         # 结束数据记录会话
#         final_stats = {
#             'training_episodes': episodes,
#             'lr_strategy': lr_strategy,
#             'final_epsilon': aq_agent.epsilon,
#             'final_learning_rate': aq_agent.get_learning_rate(),
#             'final_discount_factor': aq_agent.discount_factor
#         }
#         data_manager.end_session(aq_agent, final_stats)
        
#         # 绘制训练历史
#         data_manager.plot_training_history()
        
#         # 保存模型
#         aq_agent.save_model(model_name)
        
#         print(f"训练完成! 最终epsilon: {aq_agent.epsilon:.3f}")
#         print(f"最终胜率: {aq_agent.get_stats()['win_rate']:.3f}")
        
#         # 设置为测试模式
#         aq_agent.epsilon = 0.0
#         aq_agent.ai_type = "AQ (Trained)"
    
#     return aq_agent, random_opponent

# # 在主程序中添加使用示例
# if __name__ == "__main__":
#     from base import Game
    
#     parser = argparse.ArgumentParser(description='Approximate Q Agent 训练和测试 - 改进版')
#     parser.add_argument('--retrain', action='store_true', help='强制重新训练模型')
#     parser.add_argument('--episodes', type=int, default=2000, help='训练回合数')
#     parser.add_argument('--test-games', type=int, default=100, help='测试游戏数量')
#     parser.add_argument('--no-display', action='store_true', help='不显示游戏界面')
#     parser.add_argument('--lr-strategy', choices=['adaptive', 'fixed'], 
#                        default='adaptive', help='学习率调整策略')
#     parser.add_argument('--test-only', action='store_true', help='仅测试，不训练')
#     parser.add_argument('--print-interval', type=int, default=50, help='训练进度输出间隔')
    
#     args = parser.parse_args()
    
#     # 训练或加载模型
#     if not args.test_only:
#         aq_agent, random_opponent = train_or_load_model(
#             force_retrain=args.retrain, 
#             episodes=args.episodes,
#             lr_strategy=args.lr_strategy,
#             print_interval=args.print_interval
#         )
#     else:
#         # 仅测试模式：直接加载模型
#         print("仅测试模式，加载已训练模型...")
#         aq_agent = ApproximateQAgent(player_id=0)
#         if not aq_agent.load_model("final_ApproximateQAgent"):
#             print("无法加载模型，请先训练!")
#             exit(1)
#         from AgentFight import RandomPlayer
#         random_opponent = RandomPlayer(player_id=1)
    
#     # 设置为测试模式
#     aq_agent.set_training_mode(False)
    
#     # 测试
#     print(f"\n开始测试 {args.test_games} 场游戏...")
#     print(f"测试模式 - epsilon: {aq_agent.epsilon}")
    
#     wins = 0
#     draws = 0
    
#     for i in range(args.test_games):
#         game = Game(aq_agent, random_opponent, display=not args.no_display, delay=0.1)
#         result = game.run()
        
#         if result == 0:  # AQ agent wins
#             wins += 1
#         elif result == 2:  # Draw
#             draws += 1
            
#         if args.test_games <= 10:
#             print(f"游戏 {i+1}: {'胜利' if result == 0 else '失败' if result == 1 else '平局'}")
    
#     win_rate = wins / args.test_games
#     draw_rate = draws / args.test_games
    
#     print(f"\n测试结果:")
#     print(f"胜利: {wins}/{args.test_games} ({win_rate:.3f})")
#     print(f"平局: {draws}/{args.test_games} ({draw_rate:.3f})")
#     print(f"失败: {args.test_games - wins - draws}/{args.test_games}")
#     print(f"有效胜率 (胜+0.5*平): {win_rate + 0.5 * draw_rate:.3f}")

# # 运行示例:
# # python ApproximateQAgent.py --retrain --episodes 3000 --lr-strategy adaptive --test-games 200 --no-display --print-interval 50