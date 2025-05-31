import numpy as np
from sklearn.linear_model import SGDRegressor
from collections import defaultdict
import random
import pickle
import copy
from typing import Tuple, List, Optional, Dict, Any
from new_sim import Board, Player
from QlearningAgent import RewardFunction, Environment

class FeatureExtractor:
    """特征提取器，将棋盘状态转换为特征向量"""
    
    def __init__(self):
        self.reward_function = RewardFunction()
    
    def extract_features(self, board: Board, player_id: int, action: Optional[Tuple] = None) -> np.ndarray:
        """
        提取状态-动作特征
        
        Args:
            board: 棋盘状态
            player_id: 当前玩家ID
            action: 动作（可选，用于提取状态-动作特征）
        
        Returns:
            特征向量
        """
        features = []
        
        # 1. 基础棋盘特征
        features.extend(self._extract_board_features(board, player_id))
        
        # 2. 棋子数量和价值特征
        features.extend(self._extract_piece_features(board, player_id))
        
        # 3. 战术特征
        features.extend(self._extract_tactical_features(board, player_id))
        
        # 4. 位置控制特征
        features.extend(self._extract_position_features(board, player_id))
        
        # 5. 动作相关特征（如果提供了动作）
        if action is not None:
            features.extend(self._extract_action_features(board, player_id, action))
        else:
            # 如果没有动作，用零填充
            features.extend([0.0] * 10)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_board_features(self, board: Board, player_id: int) -> List[float]:
        """提取基础棋盘特征"""
        features = []
        
        # 己方和对方的棋子总数
        my_pieces = board.get_player_pieces(player_id)
        opponent_pieces = board.get_player_pieces(1 - player_id)
        
        features.append(len(my_pieces) / 16.0)  # 己方棋子数量比例
        features.append(len(opponent_pieces) / 16.0)  # 对方棋子数量比例
        
        # 已翻开的棋子数量
        my_revealed = sum(1 for r, c in my_pieces if board.get_piece(r, c).revealed)
        opponent_revealed = sum(1 for r, c in opponent_pieces if board.get_piece(r, c).revealed)
        
        features.append(my_revealed / max(len(my_pieces), 1))  # 己方翻开比例
        features.append(opponent_revealed / max(len(opponent_pieces), 1))  # 对方翻开比例
        
        return features
    
    def _extract_piece_features(self, board: Board, player_id: int) -> List[float]:
        """提取棋子价值特征"""
        features = []
        
        my_pieces = board.get_player_pieces(player_id)
        opponent_pieces = board.get_player_pieces(1 - player_id)
        
        # 计算各种强度棋子的数量
        my_strengths = [0] * 9  # 索引0不用，1-8对应强度
        opponent_strengths = [0] * 9
        
        for r, c in my_pieces:
            piece = board.get_piece(r, c)
            if piece.revealed:
                my_strengths[piece.strength] += 1
        
        for r, c in opponent_pieces:
            piece = board.get_piece(r, c)
            if piece.revealed:
                opponent_strengths[piece.strength] += 1
        
        # 添加关键棋子的数量特征
        features.append(my_strengths[1] / 2.0)  # 己方鼠的数量
        features.append(my_strengths[8] / 2.0)  # 己方象的数量
        features.append(opponent_strengths[1] / 2.0)  # 对方鼠的数量
        features.append(opponent_strengths[8] / 2.0)  # 对方象的数量
        
        # 计算总价值
        my_total_value = sum(my_strengths[i] * self.reward_function.get_piece_value(i) for i in range(1, 9))
        opponent_total_value = sum(opponent_strengths[i] * self.reward_function.get_piece_value(i) for i in range(1, 9))
        
        max_value = max(my_total_value + opponent_total_value, 1)
        features.append(my_total_value / max_value)  # 己方价值比例
        features.append(opponent_total_value / max_value)  # 对方价值比例
        
        return features
    
    def _extract_tactical_features(self, board: Board, player_id: int) -> List[float]:
        """提取战术特征"""
        features = []
        
        my_pieces = board.get_player_pieces(player_id)
        opponent_pieces = board.get_player_pieces(1 - player_id)
        
        # 威胁和机会统计
        total_threats = 0
        total_opportunities = 0
        total_mobility = 0
        
        for r, c in my_pieces:
            piece = board.get_piece(r, c)
            if piece.revealed:
                threats = self.reward_function.count_adjacent_threats(board, (r, c), player_id)
                opportunities = self.reward_function.count_capture_opportunities(board, (r, c), player_id)
                
                total_threats += threats
                total_opportunities += opportunities
                
                # 计算移动性（可移动的方向数）
                mobility = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 7 and 0 <= nc < 8:
                        temp_board = copy.deepcopy(board)
                        if temp_board.try_move((r, c), (nr, nc)):
                            mobility += 1
                total_mobility += mobility
        
        max_pieces = max(len(my_pieces), 1)
        features.append(total_threats / max_pieces)  # 平均威胁数
        features.append(total_opportunities / max_pieces)  # 平均机会数
        features.append(total_mobility / max_pieces)  # 平均移动性
        
        return features
    
    def _extract_position_features(self, board: Board, player_id: int) -> List[float]:
        """提取位置控制特征"""
        features = []
        
        my_pieces = board.get_player_pieces(player_id)
        
        # 中央控制
        center_control = 0
        for r, c in my_pieces:
            piece = board.get_piece(r, c)
            if piece.revealed and self.reward_function.is_center_position(r, c):
                center_control += self.reward_function.get_piece_value(piece.strength)
        
        features.append(center_control / 10.0)  # 中央控制强度
        
        # 边缘棋子数量
        edge_pieces = 0
        for r, c in my_pieces:
            if r == 0 or r == 6 or c == 0 or c == 7:
                edge_pieces += 1
        
        features.append(edge_pieces / max(len(my_pieces), 1))  # 边缘棋子比例
        
        return features
    
    def _extract_action_features(self, board: Board, player_id: int, action: Tuple) -> List[float]:
        """提取动作相关特征"""
        features = []
        action_type, pos1, pos2 = action
        
        if action_type == "reveal":
            r, c = pos1
            # 翻开棋子的位置特征
            features.append(1.0)  # 是翻开动作
            features.append(0.0)  # 不是移动动作
            features.append(1.0 if self.reward_function.is_center_position(r, c) else 0.0)  # 是否中央位置
            features.extend([0.0] * 7)  # 其他特征填0
            
        elif action_type == "move":
            start_pos, end_pos = pos1, pos2
            features.append(0.0)  # 不是翻开动作
            features.append(1.0)  # 是移动动作
            
            # 移动方向特征
            dr = end_pos[0] - start_pos[0]
            dc = end_pos[1] - start_pos[1]
            features.extend([
                1.0 if dr == -1 else 0.0,  # 向上
                1.0 if dr == 1 else 0.0,   # 向下
                1.0 if dc == -1 else 0.0,  # 向左
                1.0 if dc == 1 else 0.0,   # 向右
            ])
            
            # 移动到中央区域
            features.append(1.0 if self.reward_function.is_center_position(end_pos[0], end_pos[1]) else 0.0)
            
            # 是否是攻击性移动（目标位置有敌方棋子）
            target_piece = board.get_piece(end_pos[0], end_pos[1])
            features.append(1.0 if target_piece and target_piece.player != player_id else 0.0)
            
            # 移动棋子的价值
            moving_piece = board.get_piece(start_pos[0], start_pos[1])
            if moving_piece and moving_piece.revealed:
                features.append(self.reward_function.get_piece_value(moving_piece.strength) / 4.0)
            else:
                features.append(0.0)
        else:
            # 未知动作类型，全部填0
            features.extend([0.0] * 10)
        
        # 确保特征向量长度总是10
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]  # 截断到固定长度

class ApproximateQAgent(Player):
    """基于函数逼近的Q-learning智能体"""
    
    def __init__(self, player_id: int, learning_rate: float = 0.01, 
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        super().__init__(player_id)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor()
        
        # 使用SGD回归器作为函数逼近器
        self.q_function = SGDRegressor(
            learning_rate='constant',
            eta0=learning_rate,
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
                      board_after: Board, next_valid_actions: List[Tuple], done: bool):
        """更新Q值函数"""
        # 提取当前状态-动作特征
        features = self.feature_extractor.extract_features(board_before, self.player_id, action)
        
        # 计算目标Q值
        if done or not next_valid_actions:
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
        
    def train_episode(self) -> Tuple[float, int]:
        """训练一个回合"""
        state_board = self.env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            if self.env.current_player == self.agent.player_id:
                # RL智能体回合
                valid_actions = self.env.get_valid_actions(self.agent.player_id)
                
                if not valid_actions:
                    break
                
                board_before = copy.deepcopy(self.env.board)
                action = self.agent.choose_action(self.env.board, valid_actions)
                next_state, reward, done, _ = self.env.step(action)
                
                # 更新Q值
                if not done:
                    next_valid_actions = self.env.get_valid_actions(self.agent.player_id)
                else:
                    next_valid_actions = []
                
                self.agent.update_q_value(
                    board_before, action, reward, self.env.board, next_valid_actions, done
                )
                
                total_reward += reward
                steps += 1
                
                if done:
                    break
            else:
                # 对手回合
                if isinstance(self.opponent_agent, ApproximateQAgent):
                    valid_actions = self.env.get_valid_actions(self.opponent_agent.player_id)
                    if valid_actions:
                        action = self.opponent_agent.choose_action(self.env.board, valid_actions)
                        _, _, done, _ = self.env.step(action)
                else:
                    # 随机对手或其他类型
                    if self.opponent_agent.take_turn(self.env.board):
                        self.env.current_player = 1 - self.env.current_player
                
                # 检查游戏是否结束
                red_pieces = self.env.board.get_player_pieces(0)
                blue_pieces = self.env.board.get_player_pieces(1)
                if not red_pieces or not blue_pieces:
                    done = True
                elif len(red_pieces) == 1 and len(blue_pieces) == 1:
                    rr, rc = red_pieces[0]
                    br, bc = blue_pieces[0]
                    if not self.env.board.is_adjacent((rr, rc), (br, bc)):
                        done = True
                
                if done:
                    break
        
        return total_reward, steps
    
    def train(self, episodes: int = 10000, save_interval: int = 1000):
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
                total_reward, steps = self.train_episode()
                
                if episode % 100 == 0:
                    print(f"回合 {episode}: 奖励 = {total_reward:.2f}, 步数 = {steps}")
                
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