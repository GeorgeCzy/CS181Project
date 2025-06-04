import pygame
import random
import sys
import time
import copy # 导入 copy 模块用于深拷贝
from MCTS import MCTSAgent
from base import Player, Board, Piece, screen, font, ROWS, COLS, TILE_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, STATUS_BAR_HEIGHT, WHITE, BLACK, RED, BLUE, GREY, YELLOW, LIGHT_GREY, DARK_GREY


class HumanPlayer(Player):
    """Controls a player via human input."""
    def __init__(self, player_id):
        super().__init__(player_id)
        self.selected_piece_pos = None

    def handle_event(self, event, board):
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            row, col = y // TILE_SIZE, x // TILE_SIZE

            if not (0 <= row < ROWS and 0 <= col < COLS):
                return False # Event not handled (clicked outside board)

            piece = board.get_piece(row, col)

            if self.selected_piece_pos:
                # A piece is already selected, try to move or deselect
                sr, sc = self.selected_piece_pos
                piece_moving = board.get_piece(sr, sc)

                # 确保选择的棋子是自己的
                if piece_moving and piece_moving.player == self.player_id:
                    # 如果点击了相邻位置，尝试移动
                    if board.is_adjacent(self.selected_piece_pos, (row, col)):
                        moved = board.try_move(self.selected_piece_pos, (row, col))
                        if moved:
                            self.selected_piece_pos = None
                            return True # Move successful, turn ends
                        else:
                            # 移动失败（例如，试图移动到自己已翻开的棋子），取消选择
                            self.selected_piece_pos = None
                            # 尝试选择新点击的棋子（如果它属于当前玩家）
                            if piece and piece.player == self.player_id and piece.revealed:
                                self.selected_piece_pos = (row, col)
                    else:
                        # 目标位置不相邻，取消选择当前棋子，并尝试选择新棋子
                        self.selected_piece_pos = None
                        if piece and piece.player == self.player_id:
                            if not piece.revealed:
                                piece.revealed = True
                                return True # Revealed piece, turn ends
                            else:
                                self.selected_piece_pos = (row, col) # Select new piece
                else: # 如果 selected_piece_pos 指向的不是自己的棋子了（可能被AI吃掉），或者已经清空了
                    self.selected_piece_pos = None
                    if piece and piece.player == self.player_id:
                        if not piece.revealed:
                            piece.revealed = True
                            return True # Revealed piece, turn ends
                        else:
                            self.selected_piece_pos = (row, col) # Select new piece
            elif piece:
                # No piece selected, try to select one
                if not piece.revealed:
                    piece.revealed = True
                    return True # Revealed piece, turn ends
                elif piece.player == self.player_id: # 只能选择自己的棋子
                    self.selected_piece_pos = (row, col) # Select piece
        return False # Event not handled or turn not ended

    def get_selected_pos(self):
        """Returns the position of the currently selected piece."""
        return self.selected_piece_pos

class RandomPlayer(Player):
    """An AI player that makes random valid moves."""
    def __init__(self, player_id):
        super().__init__(player_id)

    def take_turn(self, board):
        # RandomPlayer 现在使用 Board.get_all_possible_moves
        possible_actions = board.get_all_possible_moves(self.player_id)
        random.shuffle(possible_actions)

        for action in possible_actions:
            action_type = action[0]
            if action_type == "reveal":
                r, c = action[1]
                piece = board.get_piece(r, c)
                if piece and piece.player == self.player_id and not piece.revealed:
                    piece.revealed = True
                    return True # Turn ended
            elif action_type == "move":
                start_pos, end_pos = action[1], action[2]
                piece_to_move = board.get_piece(start_pos[0], start_pos[1])
                if piece_to_move and piece_to_move.player == self.player_id:
                    if board.try_move(start_pos, end_pos):
                        return True # Turn ended
        return False # No valid moves found
class GreedyPlayer(Player):
    """An AI player that uses heuristic evaluation to make greedy moves."""
    def __init__(self, player_id):
        super().__init__(player_id)
        self.opponent_id = 1 - player_id

    def _evaluate_board(self, board):
        """
        Evaluates the current board state using heuristics.
        Based on the presentation's comprehensive formula:
        Score(s) = Σ(my pieces) - λ₁Σ(opp pieces) + λ₂·Mobility + λ₃·Advancing
        """
        score = 0
        
        # Base piece values (Elephant=100, Lion=90, Tiger=80, ..., Rat=10)
        piece_values = {8: 100, 7: 90, 6: 80, 5: 70, 4: 60, 3: 50, 2: 40, 1: 10}
        
        # Get pieces for both players
        self_pieces = board.get_player_pieces(self.player_id)
        opponent_pieces = board.get_player_pieces(self.opponent_id)
        
        # Check for game-ending states
        if not opponent_pieces:
            return float('inf')  # Win
        if not self_pieces:
            return float('-inf')  # Loss
            
        # 1. Base Score - piece values
        for r, c in self_pieces:
            piece = board.get_piece(r, c)
            if piece and piece.revealed:
                score += piece_values[piece.strength]
            else:
                score += 30  # Average value for unrevealed pieces
        λ_1 = 0.8        
        for r, c in opponent_pieces:
            piece = board.get_piece(r, c)
            if piece and piece.revealed:
                score -= piece_values[piece.strength] * λ_1
            else:
                score -= 30 * λ_1
        
        # 2. Positional Rewards - Advancing (closer to opponent's backline)
        λ_3 = 15  # Advancing bonus weight
        for r, c in self_pieces:
            piece = board.get_piece(r, c)
            if piece and piece.revealed:
                if self.player_id == 0:  # Red player (advancing towards bottom)
                    advance_score = r * 5  # More points for being further down
                else:  # Blue player (advancing towards top)
                    advance_score = (ROWS - 1 - r) * 5  # More points for being further up
                score += advance_score * λ_3 / 10
                
                # Central control bonus
                center_cols = [COLS//2 - 1, COLS//2]
                if c in center_cols:
                    score += 5
        
        # 3. Mobility - count available moves
        λ_2 = 2  # Mobility weight
        possible_actions = board.get_all_possible_moves(self.player_id)
        score += len(possible_actions) * λ_2
        
        # 4. Safety Penalty - threatened pieces
        safety_weight = 10
        for sr, sc in self_pieces:
            self_piece = board.get_piece(sr, sc)
            if self_piece and self_piece.revealed:
                for er, ec in opponent_pieces:
                    opponent_piece = board.get_piece(er, ec)
                    if opponent_piece and opponent_piece.revealed:
                        if board.is_adjacent((sr, sc), (er, ec)):
                            # Check if we're threatened
                            if (opponent_piece.strength > self_piece.strength and 
                                not (opponent_piece.strength == 8 and self_piece.strength == 1)) or \
                               (opponent_piece.strength == 1 and self_piece.strength == 8):
                                score -= safety_weight
                            # Check if we can threaten
                            elif (self_piece.strength > opponent_piece.strength and 
                                  not (self_piece.strength == 8 and opponent_piece.strength == 1)) or \
                                 (self_piece.strength == 1 and opponent_piece.strength == 8):
                                score += safety_weight * 1.5  # Bonus for threatening
        
        return score

    def _evaluate_action(self, board, action):
        """
        Evaluates a specific action by simulating it and scoring the resulting board.
        Returns the score difference (new_score - current_score).
        """
        # Get current board score
        current_score = self._evaluate_board(board)
        
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
            return float('-inf')  # Invalid action
            
        # Get new board score after action
        new_score = self._evaluate_board(temp_board)
        
        return new_score - current_score

    def take_turn(self, board):
        """
        Chooses the best action based on heuristic evaluation.
        This is a greedy approach - evaluates all possible moves and picks the best one.
        """
        print(f"Greedy Player {self.player_id} thinking...")
        start_time = time.time()
        
        # Get all possible actions
        possible_actions = board.get_all_possible_moves(self.player_id)
        
        if not possible_actions:
            return False  # No valid moves
        
        # Evaluate each action and find the best one
        best_action = None
        best_score = float('-inf')
        
        for action in possible_actions:
            action_score = self._evaluate_action(board, action)
            
            # Add small random factor to break ties
            action_score += random.uniform(-0.1, 0.1)
            
            if action_score > best_score:
                best_score = action_score
                best_action = action
        
        end_time = time.time()
        print(f"Greedy found best action: {best_action} with score improvement {best_score:.2f} in {end_time - start_time:.3f} seconds")
        
        # Execute the best action
        if best_action:
            action_type = best_action[0]
            if action_type == "reveal":
                r, c = best_action[1]
                piece = board.get_piece(r, c)
                if piece and piece.player == self.player_id and not piece.revealed:
                    piece.revealed = True
                    return True
            elif action_type == "move":
                start_pos, end_pos = best_action[1], best_action[2]
                piece_to_move = board.get_piece(start_pos[0], start_pos[1])
                if piece_to_move and piece_to_move.player == self.player_id:
                    return board.try_move(start_pos, end_pos)
        
        return False
class MinimaxPlayer(Player):
    def __init__(self, player_id, max_depth=3): # Added max_depth for search
        super().__init__(player_id)
        self.max_depth = max_depth
        self.opponent_id = 1 - player_id

    def _evaluate(self, board):
        """
        Evaluates the current board state for the AI player.
        Positive values are good for self.player_id, negative for opponent.
        """
        score = 0
        
        # 1. Piece count difference
        self_pieces = board.get_player_pieces(self.player_id)
        opponent_pieces = board.get_player_pieces(self.opponent_id)
        
        # 优先判断游戏是否结束，避免分数计算偏差
        if not opponent_pieces: # Opponent has no pieces left
            return float('inf') # Win state is highly favorable
        if not self_pieces: # Self has no pieces left
            return float('-inf') # Loss state is highly unfavorable

        score += (len(self_pieces) - len(opponent_pieces)) * 100 # Each piece is worth 100 points

        # 2. Strength sum difference (prioritize stronger pieces)
        # 只计算已翻开棋子的强度
        self_strength_sum = sum(board.get_piece(r,c).strength for r,c in self_pieces if board.get_piece(r,c).revealed)
        opponent_strength_sum = sum(board.get_piece(r,c).strength for r,c in opponent_pieces if board.get_piece(r,c).revealed)
        score += (self_strength_sum - opponent_strength_sum) * 10 # Each strength point worth 10

        # 3. Revealed vs Unrevealed pieces
        # Incentivize revealing own pieces
        for r, c in self_pieces:
            if not board.get_piece(r, c).revealed:
                score += 5 # Small bonus for having unrevealed pieces to reveal (potential future power)
        
        # Penalize opponent for having unrevealed pieces (unknown threat)
        for r, c in opponent_pieces:
            if not board.get_piece(r, c).revealed:
                score -= 10 # Small penalty for opponent having hidden pieces

        # 4. Proximity to opponent's pieces for revealed pieces (threats/opportunities)
        for sr, sc in self_pieces:
            self_piece = board.get_piece(sr, sc)
            if self_piece and self_piece.revealed:
                for er, ec in opponent_pieces:
                    opponent_piece = board.get_piece(er, ec)
                    if opponent_piece and opponent_piece.revealed:
                        if board.is_adjacent((sr, sc), (er, ec)):
                            if self_piece.strength > opponent_piece.strength or \
                               (self_piece.strength == 1 and opponent_piece.strength == 8):
                                score += 20 # Strong capture opportunity
                            elif self_piece.strength < opponent_piece.strength and \
                                 not (self_piece.strength == 8 and opponent_piece.strength == 1):
                                score -= 15 # Danger of being captured

        return score

    def _minimax(self, board, depth, maximizing_player_id):
        # Base case: max_depth reached or game over
        # 优化：在每次评估时，判断游戏是否结束，避免不必要的递归
        self_pieces_count = len(board.get_player_pieces(self.player_id))
        opponent_pieces_count = len(board.get_player_pieces(self.opponent_id))

        if depth == 0 or self_pieces_count == 0 or opponent_pieces_count == 0:
            return self._evaluate(board), None # Return score and no move

        current_player = maximizing_player_id
        is_maximizing_player = (current_player == self.player_id)

        min_strength_value = min((piece.strength for row in board.board for piece in row if piece and piece.revealed), default=0)
        max_strength_value = max((piece.strength for row in board.board for piece in row if piece and piece.revealed), default=0)
        
        best_move = None
        # 如果是最大化玩家（AI自己）
        if is_maximizing_player:
            max_eval = float('-inf')
            # 获取当前玩家所有可能的动作
            possible_actions = board.get_all_possible_moves(current_player)
            random.shuffle(possible_actions) # 打乱顺序，增加AI行为多样性（当多个动作分数相同）

            for action in possible_actions:
                # 对棋盘进行深拷贝以模拟动作
                temp_board = copy.deepcopy(board)
                
                action_performed = False
                action_type = action[0]
                
                if action_type == "reveal":
                    r, c = action[1]
                    piece = temp_board.get_piece(r, c)
                    if piece and piece.player == current_player and not piece.revealed:
                        # 假设翻开的棋子可能是最强或最弱的情况，进行两次递归评估
                        piece.revealed = True
                        original_strength = piece.strength
                        # 假设为最强
                        piece.strength = max_strength_value
                        eval_strong, _ = self._minimax(temp_board, depth - 1, 1 - current_player)
                        # 假设为最弱
                        piece.strength = min_strength_value
                        eval_weak, _ = self._minimax(temp_board, depth - 1, 1 - current_player)
                        # 恢复原始强度
                        piece.strength = original_strength
                        # 取两者的平均值作为评估
                        eval = (eval_strong + eval_weak) / 2
                        action_performed = True
                elif action_type == "move":
                    start_pos, end_pos = action[1], action[2]
                    # 在临时棋盘上尝试移动
                    action_performed = temp_board.try_move(start_pos, end_pos)

                if action_performed: # 只有当动作成功执行后，才进行下一步递归
                    # 递归调用 Minimax，切换到对手玩家
                    eval, _ = self._minimax(temp_board, depth - 1, 1 - current_player)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = action
            return max_eval, best_move
        # 如果是最小化玩家（对手）
        else:
            min_eval = float('inf')
            # 获取对手所有可能的动作
            possible_actions = board.get_all_possible_moves(current_player)
            random.shuffle(possible_actions) # 打乱顺序

            for action in possible_actions:
                temp_board = copy.deepcopy(board)

                action_performed = False
                action_type = action[0]

                if action_type == "reveal":
                    r, c = action[1]
                    piece = temp_board.get_piece(r, c)
                    if piece and piece.player == current_player and not piece.revealed:
                        piece.revealed = True
                        action_performed = True
                elif action_type == "move":
                    start_pos, end_pos = action[1], action[2]
                    action_performed = temp_board.try_move(start_pos, end_pos)

                if action_performed:
                    # 递归调用 Minimax，切换回 AI 玩家
                    eval, _ = self._minimax(temp_board, depth - 1, 1 - current_player)
                    if eval < min_eval:
                        min_eval = eval
                        best_move = action
            return min_eval, best_move

    def take_turn(self, board):
        print(f"Minimax Player {self.player_id} thinking...")
        start_time = time.time()
        
        # 运行 Minimax 搜索
        value, best_action = self._minimax(board, self.max_depth, self.player_id)
        
        end_time = time.time()
        print(f"Minimax found best action: {best_action} with value {value} in {end_time - start_time:.2f} seconds")

        if best_action:
            action_type = best_action[0]
            if action_type == "reveal":
                r, c = best_action[1]
                piece = board.get_piece(r, c)
                # 再次检查以确保棋子仍然存在且未被翻开（尽管在 Minimax 模拟中应该是这样）
                if piece and piece.player == self.player_id and not piece.revealed:
                    piece.revealed = True
                    return True
            elif action_type == "move":
                start_pos, end_pos = best_action[1], best_action[2]
                piece_to_move = board.get_piece(start_pos[0], start_pos[1])
                # 再次检查以确保棋子仍然存在且属于当前玩家
                if piece_to_move and piece_to_move.player == self.player_id:
                    return board.try_move(start_pos, end_pos)
        return False # 如果没有找到最佳动作或动作失败，不应该发生

MAX_STEPS = 1000  # 最大允许步数

class Game:

    def __init__(self, agent=None, base_agent=None):
        self.board = Board()
        if agent is None:
            self.players = {
                # 修改这里，将人类玩家替换为AI玩家
                0: MinimaxPlayer(0, max_depth=1), # 红色AI
                # 0: QLearningPlayer(0), # 红色AI
                # 0: RandomPlayer(0), # 红色AI
                1: GreedyPlayer(1)  # 蓝色AI
                # 或者可以是一个Minimax vs Random
                # 0: MinimaxPlayer(0, max_depth=3),
                # 1: RandomPlayer(1)
            }
        else:
            base_agent = base_agent if base_agent else RandomPlayer
            self.players = {
                0: agent if isinstance(agent, Player) else agent(0),  # 红色AI
                1: base_agent if isinstance(base_agent, Player) else agent(1)   # 蓝色AI
            }
        self.current_player_id = 0
        self.running = True
        self.AI_DELAY_SECONDS = 0.0 # AI行动之间的延迟，以便观察

    def _draw_board(self):
        """Draws the game board and pieces."""
        screen.fill(WHITE)
        for i in range(ROWS):
            for j in range(COLS):
                rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                # 交替颜色
                if (i + j) % 2 == 0:
                    pygame.draw.rect(screen, LIGHT_GREY, rect) # 明色格子
                else:
                    pygame.draw.rect(screen, DARK_GREY, rect) # 暗色格子
                pygame.draw.rect(screen, BLACK, rect, 1) # Cell border

                piece = self.board.get_piece(i, j)
                if piece:
                    if piece.revealed:
                        color = RED if piece.player == 0 else BLUE
                        pygame.draw.circle(screen, color, rect.center, TILE_SIZE // 3)
                        text_color = WHITE # Always white text on colored circle for better contrast
                        text = font.render(str(piece.strength), True, text_color)
                        text_rect = text.get_rect(center=rect.center)
                        screen.blit(text, text_rect)
                    else:
                        # 在棋盘上绘制未揭示的棋子
                        pygame.draw.circle(screen, BLACK, rect.center, TILE_SIZE // 3) # 使用 rect.center 代替 (x, y)
                        # pygame.draw.rect(screen, GREY, rect.inflate(-10, -10)) # Unrevealed piece block

        # Highlight selected piece for human player (不再需要，但保留以防将来修改为人类玩家)
        # human_player = self.players[0] 
        # if isinstance(human_player, HumanPlayer) and human_player.get_selected_pos():
        #     i, j = human_player.get_selected_pos()
        #     rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        #     pygame.draw.rect(screen, YELLOW, rect, 3)

        # Draw status bar
        status_bar_rect = pygame.Rect(0, SCREEN_HEIGHT - STATUS_BAR_HEIGHT, SCREEN_WIDTH, STATUS_BAR_HEIGHT)
        pygame.draw.rect(screen, BLACK, status_bar_rect) # Background for status bar

        player_name = "Red AI" if self.current_player_id == 0 else "Blue AI"
        text_color = RED if self.current_player_id == 0 else BLUE
        status_text = f"Current Turn: {player_name} (AI thinking...)"

        text_surface = font.render(status_text, True, text_color)
        screen.blit(text_surface, (10, SCREEN_HEIGHT - STATUS_BAR_HEIGHT + 5))

    def _check_game_over(self):
        """Checks for win/loss conditions and terminates the game if met."""
        red_pieces = self.board.get_player_pieces(0)
        blue_pieces = self.board.get_player_pieces(1)

        if not red_pieces:
            self._game_over("Blue AI wins!")
            return True
        elif not blue_pieces:
            self._game_over("Red AI wins!")
            return True
        elif len(red_pieces) == 1 and len(blue_pieces) == 1:
            rr, rc = red_pieces[0]
            br, bc = blue_pieces[0]
            red_piece = self.board.get_piece(rr, rc)
            blue_piece = self.board.get_piece(br, bc)

            # 只有当两个棋子都已翻开时，才能判断是否能互相捕获
            if red_piece.revealed and blue_piece.revealed:
                can_red_attack = (red_piece.strength > blue_piece.strength) or \
                                (red_piece.strength == 1 and blue_piece.strength == 8)
                can_blue_attack = (blue_piece.strength > red_piece.strength) or \
                                (blue_piece.strength == 1 and red_piece.strength == 8)

                # 如果它们相邻，且双方都无法捕获对方，则为和棋
                if self.board.is_adjacent((rr, rc), (br, bc)):
                    if not can_red_attack and not can_blue_attack:
                        self._game_over("Draw! (Stalemate - last pieces cannot capture each other)")
                        return True
                else: # 如果不相邻，也无法捕获，则为和棋
                    if not can_red_attack and not can_blue_attack:
                        self._game_over("Draw! (Last pieces not adjacent or cannot capture)")
                        return True
            # 如果有一方或双方未翻开，游戏继续 (因为信息不完全，未来可能仍有变化)
            
        return False # 游戏未结束

    def _game_over(self, message):
        """Displays game over message and exits."""
        print(message)
        self.running = False # Stop the main game loop
        # Optional: Keep the window open for a few seconds before closing
        pygame.time.wait(3000)

    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        
        step_count = 0
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                # 移除了人类玩家事件处理逻辑

            current_player_obj = self.players[self.current_player_id]
            
            # AI 玩家的回合逻辑
            # AI 不依赖事件，直接在循环中执行
            time.sleep(self.AI_DELAY_SECONDS) # 增加延迟，方便观察

            if current_player_obj.take_turn(self.board):
                # 只在AI成功执行动作后更新计数器
                
                # 在AI成功执行动作后更新计数器
                step_count += 1
                if self._check_game_over():
                    self.running = False
                elif step_count >= MAX_STEPS:
                    self._game_over(f"draw, exceeded {MAX_STEPS} steps")
                    self.running = False
                elif not self.board.get_player_pieces(self.current_player_id):
                    player_name = "Red AI" if self.current_player_id == 0 else "Blue AI"
                    self._game_over(f"{player_name} no movements avaliabe")
                    self.running = False
                else:
                    self.current_player_id = 1 - self.current_player_id # 切换回合
            else:
                # 如果AI没有找到任何合法动作（极少发生，通常意味着游戏结束了）
                # 可以在这里添加一个平局判断或者其他处理
                print(f"Player {self.current_player_id} could not make a valid move. Game might be stuck or draw.")
                self._game_over("Draw! (No valid moves left for current player or stuck state)")
                self.running = False # 结束游戏

            self._draw_board()
            pygame.display.flip()
            # pygame.time.wait(5000) # 暂停5秒
            clock.tick(30) # 控制帧率

        pygame.quit()
        sys.exit()

# --- Run the Game ---
if __name__ == "__main__":
    game = Game()
    game.run()