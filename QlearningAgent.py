import numpy as np
import random
import pickle
import copy
from typing import Tuple, List, Optional, Dict, Any
from new_sim import Board, Player


class RewardFunction:
    """斗兽棋智能奖励函数"""
    
    def __init__(self):
        # 奖励权重配置
        self.weights = {
            'win_game': 100.0,          # 获胜
            'lose_game': -100.0,        # 失败
            'draw_game': 0.0,           # 平局
            'capture_piece': 5.0,       # 吃掉对方棋子基础奖励
            'be_captured': -3.0,        # 被吃掉基础惩罚
            'mutual_destruction': -1.0,  # 同归于尽
            'reveal_piece': 0.2,        # 翻开棋子
            'move_to_capture': 2.0,     # 移动到可以吃掉对方的位置
            'move_to_danger': -1.5,     # 移动到危险位置（会被吃）
            'move_to_safety': 0.5,      # 移动到安全位置
            'control_center': 0.3,      # 控制中央区域
            'piece_mobility': 0.1,      # 棋子活动性
            'invalid_action': -2.0,     # 无效动作
            'defensive_move': 0.8,      # 防守性移动
            'aggressive_move': 1.2,     # 攻击性移动
        }
    
    def can_capture(self, attacker_strength: int, defender_strength: int) -> bool:
        """判断攻击方是否能吃掉防守方"""
        if attacker_strength == 8 and defender_strength == 1:  # 象吃鼠
            return False
        if attacker_strength > defender_strength:
            return True
        if attacker_strength == 1 and defender_strength == 8:  # 鼠吃象
            return True
        return False
    
    def get_piece_value(self, strength: int) -> float:
        """根据棋子强度返回价值权重"""
        # 象和鼠有特殊价值，中等强度棋子也很重要
        value_map = {1: 3.0, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5, 6: 3.0, 7: 3.5, 8: 4.0}
        return value_map.get(strength, 1.0)
    
    def is_center_position(self, row: int, col: int) -> bool:
        """判断是否为中央区域"""
        return 2 <= row <= 4 and 2 <= col <= 5
    
    def count_adjacent_threats(self, board: Board, pos: Tuple[int, int], player_id: int) -> int:
        """计算位置周围的威胁数量"""
        r, c = pos
        threats = 0
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 7 and 0 <= nc < 8:
                piece = board.get_piece(nr, nc)
                if piece and piece.player != player_id and piece.revealed:
                    # 检查这个敌方棋子是否能威胁到当前位置的己方棋子
                    current_piece = board.get_piece(r, c)
                    if current_piece and self.can_capture(piece.strength, current_piece.strength):
                        threats += 1
        
        return threats
    
    def count_capture_opportunities(self, board: Board, pos: Tuple[int, int], player_id: int) -> int:
        """计算位置周围可以吃掉的敌方棋子数量"""
        r, c = pos
        opportunities = 0
        
        current_piece = board.get_piece(r, c)
        if not current_piece or current_piece.player != player_id:
            return 0
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 7 and 0 <= nc < 8:
                target_piece = board.get_piece(nr, nc)
                if target_piece and target_piece.player != player_id:
                    if self.can_capture(current_piece.strength, target_piece.strength):
                        opportunities += 1
        
        return opportunities
    
    def calculate_reward(self, 
                        board_before: Board,
                        board_after: Board, 
                        action: Tuple[str, Tuple[int, int], Optional[Tuple[int, int]]],
                        player_id: int,
                        done: bool) -> float:
        """
        计算奖励值
        
        Args:
            board_before: 执行动作前的棋盘状态
            board_after: 执行动作后的棋盘状态
            action: 执行的动作
            player_id: 当前玩家ID
            done: 游戏是否结束
        
        Returns:
            float: 奖励值
        """
        total_reward = 0.0
        action_type, pos1, pos2 = action
        
        # 1. 游戏结束奖励
        if done:
            red_pieces = board_after.get_player_pieces(0)
            blue_pieces = board_after.get_player_pieces(1)
            
            if not red_pieces:
                total_reward += self.weights['win_game'] if player_id == 1 else self.weights['lose_game']
            elif not blue_pieces:
                total_reward += self.weights['win_game'] if player_id == 0 else self.weights['lose_game']
            else:
                total_reward += self.weights['draw_game']
        
        # 2. 翻开棋子奖励
        if action_type == "reveal":
            r, c = pos1
            piece = board_after.get_piece(r, c)
            if piece and piece.revealed:
                # 根据翻开棋子的强度和位置给奖励
                piece_value = self.get_piece_value(piece.strength)
                total_reward += self.weights['reveal_piece'] * piece_value
        
        # 3. 移动棋子的复杂奖励计算
        elif action_type == "move":
            start_pos, end_pos = pos1, pos2
            
            # 获取移动的棋子信息
            moving_piece_before = board_before.get_piece(start_pos[0], start_pos[1])
            target_piece_before = board_before.get_piece(end_pos[0], end_pos[1])
            piece_after = board_after.get_piece(end_pos[0], end_pos[1])
            
            if not moving_piece_before:
                total_reward += self.weights['invalid_action']
                return total_reward
            
            # 3a. 吃棋子的奖励/惩罚
            if target_piece_before and target_piece_before.player != player_id:
                if piece_after and piece_after.player == player_id:
                    # 成功吃掉对方棋子
                    capture_value = self.get_piece_value(target_piece_before.strength)
                    total_reward += self.weights['capture_piece'] * capture_value
                    total_reward += self.weights['aggressive_move']
                elif not piece_after:
                    # 同归于尽
                    total_reward += self.weights['mutual_destruction']
                else:
                    # 被对方吃掉
                    lost_value = self.get_piece_value(moving_piece_before.strength)
                    total_reward += self.weights['be_captured'] * lost_value
            
            # 3b. 战术位置奖励
            if piece_after and piece_after.player == player_id:
                # 移动后的位置分析
                
                # 检查移动后是否能吃掉更多棋子
                capture_opportunities_after = self.count_capture_opportunities(board_after, end_pos, player_id)
                capture_opportunities_before = self.count_capture_opportunities(board_before, start_pos, player_id)
                
                if capture_opportunities_after > capture_opportunities_before:
                    total_reward += self.weights['move_to_capture'] * (capture_opportunities_after - capture_opportunities_before)
                
                # 检查移动后的危险程度
                threats_after = self.count_adjacent_threats(board_after, end_pos, player_id)
                threats_before = self.count_adjacent_threats(board_before, start_pos, player_id)
                
                if threats_after > threats_before:
                    # 移动到更危险的位置
                    threat_penalty = self.weights['move_to_danger'] * (threats_after - threats_before)
                    total_reward += threat_penalty
                elif threats_after < threats_before:
                    # 移动到更安全的位置
                    safety_bonus = self.weights['move_to_safety'] * (threats_before - threats_after)
                    total_reward += safety_bonus
                
                # 控制中央区域奖励
                if self.is_center_position(end_pos[0], end_pos[1]):
                    total_reward += self.weights['control_center'] * self.get_piece_value(piece_after.strength)
                
                # 防守性移动奖励（保护己方重要棋子）
                if self._is_defensive_move(board_before, board_after, start_pos, end_pos, player_id):
                    total_reward += self.weights['defensive_move']
        
        # 4. 无效动作惩罚（这应该在环境的step函数中处理）
        return total_reward
    
    def _is_defensive_move(self, board_before: Board, board_after: Board, 
                          start_pos: Tuple[int, int], end_pos: Tuple[int, int], 
                          player_id: int) -> bool:
        """判断是否为防守性移动"""
        # 检查移动是否保护了己方的高价值棋子
        for r in range(7):
            for c in range(8):
                piece = board_after.get_piece(r, c)
                if piece and piece.player == player_id and piece.revealed:
                    # 检查这个己方棋子在移动前后的威胁变化
                    threats_before = self.count_adjacent_threats(board_before, (r, c), player_id)
                    threats_after = self.count_adjacent_threats(board_after, (r, c), player_id)
                    
                    if threats_after < threats_before and self.get_piece_value(piece.strength) >= 3.0:
                        # 高价值棋子的威胁减少了，可能是防守性移动
                        return True
        
        return False
    
    def get_position_value(self, board: Board, pos: Tuple[int, int], player_id: int) -> float:
        """评估某个位置对玩家的价值"""
        r, c = pos
        value = 0.0
        
        # 中央位置更有价值
        if self.is_center_position(r, c):
            value += 1.0
        
        # 计算周围的机会和威胁
        opportunities = self.count_capture_opportunities(board, pos, player_id)
        threats = self.count_adjacent_threats(board, pos, player_id)
        
        value += opportunities * 2.0 - threats * 1.5
        
        return value


class Environment:
    """斗兽棋强化学习环境"""
    
    def __init__(self):
        self.board = None
        self.current_player = 0  # 0: RL agent, 1: opponent
        self.reward_function = RewardFunction()
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置环境，返回初始状态"""
        self.board = Board()
        self.current_player = 0
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        将棋盘状态转换为数值特征向量
        状态包括：
        - 棋盘上每个位置的棋子信息 (玩家, 强度, 是否翻开)
        - 当前玩家
        - 游戏阶段信息
        """
        state = np.zeros((7, 8, 4))  # ROWS x COLS x features
        
        for r in range(7):
            for c in range(8):
                piece = self.board.get_piece(r, c)
                if piece:
                    state[r, c, 0] = piece.player  # 玩家 (0 或 1)
                    state[r, c, 1] = piece.strength / 8.0  # 强度标准化
                    state[r, c, 2] = 1 if piece.revealed else 0  # 是否翻开
                    state[r, c, 3] = 1  # 有棋子
                else:
                    state[r, c, :] = 0  # 空位置
        
        # 展平状态向量并添加当前玩家信息
        flat_state = state.flatten()
        current_player_feature = np.array([self.current_player])
        
        return np.concatenate([flat_state, current_player_feature])
    
    def get_valid_actions(self, player_id: int) -> List[Tuple[str, Tuple[int, int], Optional[Tuple[int, int]]]]:
        """获取指定玩家的所有有效动作"""
        actions = []
        
        # 翻开棋子动作
        for r in range(7):
            for c in range(8):
                piece = self.board.get_piece(r, c)
                if piece and piece.player == player_id and not piece.revealed:
                    actions.append(("reveal", (r, c), None))
        
        # 移动棋子动作
        for r in range(7):
            for c in range(8):
                piece = self.board.get_piece(r, c)
                if piece and piece.player == player_id and piece.revealed:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 7 and 0 <= nc < 8:
                            # 使用临时棋盘测试移动是否有效
                            temp_board = copy.deepcopy(self.board)
                            if temp_board.try_move((r, c), (nr, nc)):
                                actions.append(("move", (r, c), (nr, nc)))
        
        return actions
    
    def step(self, action: Tuple[str, Tuple[int, int], Optional[Tuple[int, int]]]) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作，使用智能奖励函数"""
        import copy
        
        action_type, pos1, pos2 = action
        board_before = copy.deepcopy(self.board)
        done = False
        info = {}
        
        # 验证动作有效性
        valid_actions = self.get_valid_actions(self.current_player)
        if action not in valid_actions:
            # 无效动作
            reward = self.reward_function.weights['invalid_action']
            return self.get_state(), reward, False, {"invalid": True}
        
        # 执行动作
        success = False
        if action_type == "reveal":
            r, c = pos1
            piece = self.board.get_piece(r, c)
            if piece and piece.player == self.current_player and not piece.revealed:
                piece.revealed = True
                success = True
                
        elif action_type == "move":
            success = self.board.try_move(pos1, pos2)
        
        if not success:
            reward = self.reward_function.weights['invalid_action']
            return self.get_state(), reward, False, {"invalid": True}
        
        # 检查游戏结束
        red_pieces = self.board.get_player_pieces(0)
        blue_pieces = self.board.get_player_pieces(1)
        
        if not red_pieces or not blue_pieces:
            done = True
        elif len(red_pieces) == 1 and len(blue_pieces) == 1:
            rr, rc = red_pieces[0]
            br, bc = blue_pieces[0]
            if not self.board.is_adjacent((rr, rc), (br, bc)):
                done = True
        
        # 使用智能奖励函数计算奖励
        reward = self.reward_function.calculate_reward(
            board_before, self.board, action, self.current_player, done
        )
        
        # 切换玩家
        self.current_player = 1 - self.current_player
        
        return self.get_state(), reward, done, info

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
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
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
        
    def train_episode(self) -> Tuple[float, int]:
        """训练一个回合，返回总奖励和步数"""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            if self.env.current_player == self.agent.player_id:
                # RL智能体回合
                valid_actions = self.env.get_valid_actions(self.agent.player_id)
                
                if not valid_actions:
                    break
                
                action = self.agent.choose_action(state, valid_actions)
                next_state, reward, done, _ = self.env.step(action)
                
                # 更新Q值
                if not done:
                    next_valid_actions = self.env.get_valid_actions(self.agent.player_id)
                else:
                    next_valid_actions = []
                
                self.agent.update_q_value(state, action, reward, next_state, next_valid_actions)
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            else:
                # 对手回合
                if isinstance(self.opponent_agent, QLearningAgent):
                    valid_actions = self.env.get_valid_actions(self.opponent_agent.player_id)
                    if valid_actions:
                        action = self.opponent_agent.choose_action(state, valid_actions)
                        state, _, done, _ = self.env.step(action)
                else:
                    # 随机对手或其他类型
                    if self.opponent_agent.take_turn(self.env.board):
                        state = self.env.get_state()
                        self.env.current_player = 1 - self.env.current_player
                
                if done:
                    break
        
        return total_reward, steps
    
    def train(self, episodes: int = 10000, save_interval: int = 1000):
        """训练指定回合数"""
        print(f"开始训练 {episodes} 回合...")
        save_path = r"model_data/"
        
        for episode in range(episodes):
            total_reward, steps = self.train_episode()
            
            if episode % 100 == 0:
                print(f"回合 {episode}: 奖励 = {total_reward:.2f}, 步数 = {steps}")
            
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