import numpy as np
from sklearn.linear_model import SGDRegressor
from collections import defaultdict
import random
import pickle
import copy
from typing import Tuple, List, Optional, Dict, Any
from base import Board, Player
from utils import RewardFunction, Environment

class FeatureExtractor:
    """特征提取器, 根据RewardFunction的评估逻辑提取特征"""
    
    def __init__(self):
        self.reward_function = RewardFunction()
    
    def extract_features(self, board: Board, player_id: int, action: Optional[Tuple] = None) -> np.ndarray:
        """提取状态-动作特征"""
        features = []
        
        # 1. 基础棋盘特征
        features.extend(self._extract_board_features(board, player_id))
        
        # 2. 威胁和机会特征 (基于RewardFunction的评估)
        features.extend(self._extract_threat_opportunity_features(board, player_id))
        
        # 3. 动作特征
        if action:
            features.extend(self._extract_action_features(board, player_id, action))
        
        return np.array(features, dtype=np.float32)
    
    def _extract_board_features(self, board: Board, player_id: int) -> List[float]:
        """提取基础棋盘特征"""
        features = []
        
        my_pieces = board.get_player_pieces(player_id)
        opponent_pieces = board.get_player_pieces(1 - player_id)
        
        # 棋子数量和价值
        my_total_value = 0
        opponent_total_value = 0
        my_revealed_count = 0
        opponent_revealed_count = 0
        
        for r, c in my_pieces:
            piece = board.get_piece(r, c)
            if piece.revealed:
                my_revealed_count += 1
                my_total_value += self.reward_function.get_piece_value(piece.strength)
        
        for r, c in opponent_pieces:
            piece = board.get_piece(r, c)
            if piece.revealed:
                opponent_revealed_count += 1
                opponent_total_value += self.reward_function.get_piece_value(piece.strength)
        
        # 添加特征
        features.extend([
            len(my_pieces) / 16.0,  # 己方棋子数量比例
            len(opponent_pieces) / 16.0,  # 对方棋子数量比例
            my_revealed_count / max(len(my_pieces), 1),  # 己方已翻开比例
            opponent_revealed_count / max(len(opponent_pieces), 1),  # 对方已翻开比例
            my_total_value / (my_total_value + opponent_total_value + 1e-6),  # 己方价值比例
            opponent_total_value / (my_total_value + opponent_total_value + 1e-6)  # 对方价值比例
        ])
        
        return features
    
    def _extract_threat_opportunity_features(self, board: Board, player_id: int) -> List[float]:
        """提取威胁和机会特征，基于最近敌人的曼哈顿距离评估"""
        features = []
        
        max_threat = 0
        max_opportunity = 0
        total_threat = 0
        total_opportunity = 0
        piece_count = 0
        min_distance = float('inf')
        
        # 对每个己方棋子评估威胁和机会
        for r in range(7):
            for c in range(8):
                piece = board.get_piece(r, c)
                if piece and piece.player == player_id and piece.revealed:
                    # 使用RewardFunction的方法找到最近的敌人
                    enemy_pos, distance = self.reward_function.find_closest_enemy(board, (r, c), player_id)
                    
                    if enemy_pos and distance:
                        enemy = board.get_piece(enemy_pos[0], enemy_pos[1])
                        
                        # 更新最小距离
                        min_distance = min(min_distance, distance)
                        
                        # 计算威胁值
                        if self.reward_function.can_capture(enemy.strength, piece.strength):
                            # 威胁随距离增加而减小
                            threat = 4.0 / (distance + 1)
                            if self.reward_function.get_piece_value(piece.strength) >= 3.0:
                                threat *= 1.5
                            threat = -threat
                            total_threat += abs(threat)
                            max_threat = max(max_threat, abs(threat))
                        
                        # 计算机会值
                        if self.reward_function.can_capture(piece.strength, enemy.strength):
                            # 机会随距离增加而减小
                            opportunity = 3.0 / (distance + 1)
                            if self.reward_function.get_piece_value(enemy.strength) >= 3.0:
                                opportunity *= 1.5
                            total_opportunity += opportunity
                            max_opportunity = max(max_opportunity, opportunity)
                    
                    piece_count += 1
        
        # 计算平均值
        avg_threat = total_threat / max(piece_count, 1)
        avg_opportunity = total_opportunity / max(piece_count, 1)
        
        # 添加特征
        features.extend([
            max_threat,          # 最大威胁
            max_opportunity,     # 最大机会
            avg_threat,          # 平均威胁
            avg_opportunity,     # 平均机会
            1.0 / (min_distance + 1) if min_distance != float('inf') else 0,  # 最近敌人距离的倒数
            piece_count / 8.0    # 己方棋子数量比例
        ])
    
        return features
    
    def _extract_action_features(self, board: Board, player_id: int, action: Tuple) -> List[float]:
        """提取动作特征"""
        features = []
        action_type, pos1, pos2 = action
        
        if action_type == "reveal":
            # 翻开动作的特征
            r, c = pos1
            features.extend([
                1.0,  # 是翻开动作
                0.0,  # 不是移动动作
                r / 6.0,  # 行位置（归一化）
                c / 7.0   # 列位置（归一化）
            ])
            
        else:  # move
            # 移动动作的特征
            sr, sc = pos1
            er, ec = pos2
            
            # 计算移动前后的威胁和机会变化
            before_threat, before_opportunity = self.reward_function.evaluate_position(board, pos1, player_id)
            
            # 模拟移动后的状态
            temp_board = copy.deepcopy(board)
            temp_board.try_move(pos1, pos2)
            after_threat, after_opportunity = self.reward_function.evaluate_position(temp_board, pos2, player_id)
            
            features.extend([
                0.0,  # 不是翻开动作
                1.0,  # 是移动动作
                after_threat - before_threat,  # 威胁变化
                after_opportunity - before_opportunity  # 机会变化
            ])
        
        return features

class ApproximateQAgent(Player):
    def __init__(self, player_id: int, learning_rate: float = 0.01, 
                 discount_factor: float = 0.5,          # 初始折扣因子更小
                 discount_max: float = 0.99,           # 最大折扣因子
                 discount_growth: float = 0.001,       # 增长率
                 epsilon: float = 0.9,
                 epsilon_min: float = 0.01, 
                 epsilon_decay: float = 0.995):
        super().__init__(player_id)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.discount_max = discount_max
        self.discount_growth = discount_growth
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.episode_count = 0
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor()
        
        # 使用SGD回归器作为函数逼近器
        self.q_function = SGDRegressor(
            loss='huber',           # 使用更稳定的Huber损失
            learning_rate='adaptive',# 自适应学习率
            eta0=learning_rate,
            power_t=0.25,           # 学习率衰减指数
            alpha=0.0001,           # L2正则化
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            warm_start=True
        )
        
        # 用于存储训练数据
        self.training_data = []
        self.is_fitted = False
        
        if epsilon > 0:
            self.ai_type = f"AQ (ε={epsilon:.2f})"
        else:
            self.ai_type = "AQ (Trained)"

    def decay_epsilon(self, win_rate: float = None):
        """智能衰减探索率"""
        if win_rate is not None:
            # 基于胜率的自适应衰减
            if win_rate < 0.3:  # 胜率太低，减缓衰减
                decay = 0.999
            elif win_rate > 0.6 and self.episode_count > 20:  # 胜率较高，加快衰减
                decay = 0.95
            else:  # 正常衰减
                decay = self.epsilon_decay
                
            self.epsilon = max(self.epsilon_min, self.epsilon * decay)
        else:
            # 使用episode count进行衰减
            self.episode_count += 1
            # 指数衰减和周期衰减的结合
            periodic_factor = 0.5 * (1 + np.cos(self.episode_count * np.pi / 1000))
            exp_decay = np.exp(-0.003 * self.episode_count)
            
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon_min + (self.epsilon - self.epsilon_min) * exp_decay * periodic_factor
            )

    def update_discount_factor(self, win_rate: float = None):
        """动态更新折扣因子"""
        if win_rate is not None:
            # 基于胜率的自适应增长
            if win_rate < 0.3:  # 胜率低，减缓增长
                growth = self.discount_growth * 0.5
            elif win_rate > 0.6 and self.episode_count > 20:  # 胜率高，加快增长
                growth = self.discount_growth * 2.0
            else:  # 正常增长
                growth = self.discount_growth
            
            # 更新折扣因子
            new_discount = self.discount_factor + growth
            self.discount_factor = min(self.discount_max, new_discount)
        else:
            # 基于回合数的渐进增长
            progress = min(1.0, self.episode_count / 100)  # 在100回合内逐渐增长
            self.discount_factor = self.discount_factor + \
                (self.discount_max - self.discount_factor) * progress
    
    def get_q_value(self, board: Board, action: Tuple) -> float:
        """获取状态-动作的Q值"""
        try:
            features = self.feature_extractor.extract_features(board, self.player_id, action)
            
            # 确保特征向量长度正确
            if len(features) == 0:
                return 0.0
                
            if not self.is_fitted:
                return 0.0
            
            # 重塑为2D数组
            features_2d = features.reshape(1, -1)
            return self.q_function.predict(features_2d)[0]
            
        except Exception as e:
            print(f"计算Q值时出错: {e}")
            return 0.0
    
    def choose_action(self, board: Board, valid_actions: List[Tuple]) -> Tuple:
        """使用epsilon-greedy策略选择动作"""
        if random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.choice(valid_actions)
        else:
            # 利用：选择Q值最高的动作
            best_action = None
            best_q_value = float('-inf')
            
            for action in valid_actions:
                q_value = self.get_q_value(board, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            return best_action if best_action else random.choice(valid_actions)
    
    def update_q_value(self, board_before: Board, action: Tuple, reward: float, 
                      board_after: Board, next_valid_actions: List[Tuple], result: int):
        """更新Q值函数"""
        # 提取当前状态-动作特征
        features = self.feature_extractor.extract_features(board_before, self.player_id, action)
        
        # 计算目标Q值
        if result != -1 or not next_valid_actions:
            target_q = reward
        else:
            # 找到下一状态的最大Q值
            max_next_q = max([self.get_q_value(board_after, next_action) 
                             for next_action in next_valid_actions])
            target_q = reward + self.discount_factor * max_next_q
        
        # 添加训练数据
        self.training_data.append((features, target_q))
        
        # 定期更新模型
        if len(self.training_data) >= 32:  # 批量更新
            self._update_model()
    
    def _update_model(self):
        """更新函数逼近模型"""
        if not self.training_data:
            return
        
        try:
            # 检查特征向量长度是否一致
            feature_lengths = [len(data[0]) for data in self.training_data]
            if len(set(feature_lengths)) > 1:
                print(f"警告：特征向量长度不一致: {set(feature_lengths)}")
                # 过滤掉长度不正确的特征
                expected_length = max(set(feature_lengths), key=feature_lengths.count)
                filtered_data = [(features, target) for features, target in self.training_data 
                               if len(features) == expected_length]
                if not filtered_data:
                    # print("没有有效的训练数据，跳过更新")
                    self.training_data = []
                    return
                self.training_data = filtered_data
            
            X = np.array([data[0] for data in self.training_data])
            y = np.array([data[1] for data in self.training_data])
            
            # 检查数组形状
            # print(f"训练数据形状: X={X.shape}, y={y.shape}")
            
            if not self.is_fitted:
                self.q_function.fit(X, y)
                self.is_fitted = True
            else:
                # 增量学习
                self.q_function.partial_fit(X, y)
            
            # 清空训练数据
            self.training_data = []
            
        except Exception as e:
            print(f"模型更新出错: {e}")
            print(f"训练数据数量: {len(self.training_data)}")
            if self.training_data:
                print(f"第一个特征向量长度: {len(self.training_data[0][0])}")
                print(f"第一个特征向量: {self.training_data[0][0]}")
            # 清空有问题的训练数据
            self.training_data = []
            
    def debug_feature_extraction(self, board: Board, actions: List[Tuple]) -> None:
        """调试特征提取，检查特征向量长度的一致性"""
        print("=== 特征提取调试 ===")
        for i, action in enumerate(actions[:5]):  # 只检查前5个动作
            features = self.feature_extractor.extract_features(board, self.player_id, action)
            print(f"动作 {i}: {action}")
            print(f"特征长度: {len(features)}")
            print(f"特征向量: {features}")
            print("---")
    
    def take_turn(self, board) -> bool:
        """为游戏集成实现的take_turn方法"""
        env = Environment()
        env.board = copy.deepcopy(board)
        env.current_player = self.player_id
        
        valid_actions = env.get_valid_actions(self.player_id)
        
        if not valid_actions:
            return False
        
        action = self.choose_action(board, valid_actions)
        
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
        """保存模型"""
        import pickle
        model_data = {
            'q_function': self.q_function,
            'is_fitted': self.is_fitted,
            'feature_extractor': self.feature_extractor
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filename: str):
        """加载模型"""
        import pickle
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
                self.q_function = model_data['q_function']
                self.is_fitted = model_data['is_fitted']
                self.feature_extractor = model_data['feature_extractor']
        except FileNotFoundError:
            print(f"模型文件 {filename} 不存在，从头开始训练")

class ApproximateQTrainer:
    """基于函数逼近的Q-learning训练器"""
    
    def __init__(self, agent: ApproximateQAgent, opponent_agent: Player):
        self.agent = agent
        self.opponent_agent = opponent_agent
        self.env = Environment()
        self.wins = 0
        self.losses = 0
        self.win_history = []  # 用于跟踪胜率
        self.window_size = 100  # 胜率计算窗口大小
        
    def train_episode(self, opponent_agent = None) -> Tuple[float, int]:
        """训练一个回合"""
        state_board = self.env.reset()
        total_reward = 0
        steps = 0
        if opponent_agent is not None:
            self.opponent_agent = opponent_agent

        replay_buffer = []
        buffer_size = 1000
        batch_size = 32
        
        while True:
            if self.env.current_player == self.agent.player_id:
                valid_actions = self.env.get_valid_actions(self.agent.player_id)
                
                if not valid_actions:
                    break
                
                board_before = copy.deepcopy(self.env.board)
                # 减少早期探索
                if random.random() < max(0.1, 1.0 - steps / 500):
                    action = random.choice(valid_actions)
                else:
                    action = self.agent.choose_action(self.env.board, valid_actions)
                    
                next_state, reward, result, _ = self.env.step(action)
                
                # 存储经验
                experience = (board_before, action, reward, copy.deepcopy(self.env.board), result)
                replay_buffer.append(experience)
                if len(replay_buffer) > buffer_size:
                    replay_buffer.pop(0)
                
                # 批量学习
                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    for exp in batch:
                        b_before, a, r, b_after, res = exp
                        next_actions = self.env.get_valid_actions(self.agent.player_id) if res == -1 else []
                        self.agent.update_q_value(b_before, a, r, b_after, next_actions, res)
                
                total_reward += reward
                steps += 1
                
                if result != -1:
                    break
            else:
                # 对手回合
                if isinstance(self.opponent_agent, ApproximateQAgent):
                    valid_actions = self.env.get_valid_actions(self.opponent_agent.player_id)
                    if valid_actions:
                        action = self.opponent_agent.choose_action(self.env.board, valid_actions)
                        _, _, result, _ = self.env.step(action)
                else:
                    # 随机对手或其他类型
                    if self.opponent_agent.take_turn(self.env.board):
                        self.env.current_player = 1 - self.env.current_player
                
                result = self.env._check_game_over()
                
                if result != -1:
                    break

            if steps >= 1000:
                result = 2
                break
        
        return total_reward, steps, result
    
    def train(self, opponent_agent = None, episodes: int = 10000, save_interval: int = 1000):
        """训练指定回合数"""
        print(f"开始函数逼近Q-learning训练 {episodes} 回合...")
        
        # 在训练前进行特征一致性检查
        print("进行特征一致性检查...")
        test_board = self.env.reset()
        valid_actions = self.env.get_valid_actions(self.agent.player_id)
        if valid_actions:
            self.agent.debug_feature_extraction(self.env.board, valid_actions)
        
        save_path = r"model_data/"
        
        for episode in range(episodes):
            try:
                total_reward, steps, result = self.train_episode()

                # 记录胜负
                if result == self.agent.player_id:
                    self.wins += 1
                    self.win_history.append(1)
                elif result == 1 - self.agent.player_id:
                    self.losses += 1
                    self.win_history.append(0)
                else:
                    self.win_history.append(0.5)  # 平局
                
                # 保持历史记录在窗口大小范围内
                if len(self.win_history) > self.window_size:
                    self.win_history.pop(0)
                
                # 计算当前胜率
                current_win_rate = sum(self.win_history) / len(self.win_history)
                
                # 更新epsilon
                self.agent.decay_epsilon(current_win_rate)
                self.agent.update_discount_factor(current_win_rate)
                
                if episode % 10 == 0:
                    print(f"回合 {episode}: 奖励 = {total_reward:.2f}, "
                          f"步数 = {steps}, 胜 = {self.wins}, 负 = {self.losses}, "
                          f"胜率 = {current_win_rate:.2f}, epsilon = {self.agent.epsilon:.3f}")
                    self.wins = 0
                    self.losses = 0
                
                if episode % save_interval == 0 and episode > 0:
                    self.agent.save_model(save_path + f"aq_model_episode_{episode}.pkl")
                    
            except Exception as e:
                print(f"回合 {episode} 训练出错: {e}")
                continue
        
        # 最终更新模型
        self.agent._update_model()
        print("训练完成！")
        self.agent.save_model(save_path + "final_aq_model.pkl")

# 在主程序中添加使用示例
if __name__ == "__main__":
    from new_sim import RandomPlayer
    
    # 创建智能体
    aq_agent = ApproximateQAgent(player_id=0, learning_rate=0.01, epsilon=0.1)
    random_opponent = RandomPlayer(player_id=1)
    
    # 创建训练器
    trainer = ApproximateQTrainer(aq_agent, random_opponent)
    
    # 开始训练
    trainer.train(episodes=5000)
    
    print("函数逼近Q-learning训练完成!")