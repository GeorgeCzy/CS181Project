import pygame
import random
import sys
import time
import copy # 导入 copy 模块用于深拷贝
from MCTS import MCTSAgent
from base import Player, Board, Piece, screen, font, small_font, ROWS, COLS, TILE_SIZE, SCREEN_WIDTH, SCREEN_HEIGHT, STATUS_BAR_HEIGHT, WHITE, BLACK, RED, BLUE, GREY, YELLOW, GREEN


class HumanPlayer(Player):
    """Controls a player via human input."""
    def __init__(self, player_id):
        super().__init__(player_id)
        self.selected_piece_pos = None
        self.ai_type = "Human Player"

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
                    if board.is_adjacent(self.selected_piece_pos, (row, col)):
                        # 尝试移动
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
        self.ai_type = "Random AI"

    def take_turn(self, board):
        # RandomPlayer 现在使用 Board.get_all_possible_moves
        possible_actions = board.get_all_possible_moves(self.player_id)
        random.shuffle(possible_actions)

        for action in possible_actions:
            # RandomPlayer 在这里不需要深拷贝，因为它是直接在实际 board 上尝试动作
            # MinimaxPlayer 才需要深拷贝进行模拟
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

### NEW CODE FOR MINIMAX PLAYER ###
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
                        piece.revealed = True
                        action_performed = True
                elif action_type == "move":
                    start_pos, end_pos = action[1], action[2]
                    # 在临时棋盘上尝试移动
                    # 这里不需要再检查 piece_to_move.revealed，因为 get_all_possible_moves 已经过滤了
                    action_performed = temp_board.try_move(start_pos, end_pos)

                if action_performed: # 只有当动作成功执行后，才进行下一步递归
                    # 递归调用 Minimax，切换到对手玩家
                    eval, _ = self._minimax(temp_board, depth - 1, self.opponent_id)
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
                    eval, _ = self._minimax(temp_board, depth - 1, self.player_id)
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


### END NEW CODE FOR MINIMAX PLAYER ###


class Game:
    """Main class to manage the game flow."""
    def __init__(self, agent=None, agent_base=None, test_mode=0):
        self.board = Board()
        if not test_mode:
            self.mode = 0 # not test mode, human vs AI
            if agent is None:
                self.players = {
                    0: HumanPlayer(0),
                    1: RandomPlayer(1)
                }
            elif isinstance(agent, Player):
                self.players = {
                    0: HumanPlayer(0),
                    1: agent
                }
            else:
                self.players = {
                    0: HumanPlayer(0),
                    1: agent(1)
                }
        else:
            self.mode = 1 # test mode, AI vs AI
            if agent_base is None:
                self.players = {
                    0: agent if isinstance(agent, Player) else agent(0), # AI Player 1
                    1: RandomPlayer(1)   # AI Player 2
                }
            elif isinstance(agent_base, Player):
                self.players = {
                    0: agent if isinstance(agent, Player) else agent(0),  # AI Player 1
                    1: agent_base   # AI Player 2
                }
            else:
                self.players = {
                    0: agent if isinstance(agent, Player) else agent(0),  # AI Player 1
                    1: agent_base(1)   # AI Player 2
                }
        self.current_player_id = 0
        self.running = True
        self._last_human_move_time = 0 # Track when human player last made a move
        self.AI_DELAY_SECONDS = 1.5 # The delay for AI moves
        
    def _get_player_type_name(self, player):
        """获取玩家类型的显示名称"""
        if hasattr(player, 'ai_type'):
            return player.ai_type
        
        # 如果没有ai_type属性，根据类名推断
        class_name = player.__class__.__name__
        type_map = {
            'HumanPlayer': 'Human Player',
            'RandomPlayer': 'Random',
            'QLearningAgent': 'QL AI',
            'DQNAgent': 'DQN AI',
            'MinimaxPlayer': 'Minimax AI',
            'AlphaBetaPlayer': 'Alpha-Beta AI',
            'MCTSPlayer': 'MCTS AI',
        }
        return type_map.get(class_name, f'{class_name} AI')

    def _draw_board(self):
        """Draws the game board and pieces."""
        screen.fill(WHITE)
        for i in range(ROWS):
            for j in range(COLS):
                rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
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
                        pygame.draw.rect(screen, GREY, rect.inflate(-10, -10)) # Unrevealed piece block

        if self.mode == 0: # Human mode
        # Highlight selected piece for human player
            human_player = self.players[0] # Human player is always player 0 in this setup
            if isinstance(human_player, HumanPlayer) and human_player.get_selected_pos():
                i, j = human_player.get_selected_pos()
                rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(screen, YELLOW, rect, 3)

        # Draw status bar
        status_bar_rect = pygame.Rect(0, SCREEN_HEIGHT - STATUS_BAR_HEIGHT, SCREEN_WIDTH, STATUS_BAR_HEIGHT)
        pygame.draw.rect(screen, BLACK, status_bar_rect) # Background for status bar
        
        current_player = self.players[self.current_player_id]
        player_name = "Red" if self.current_player_id == 0 else "Blue"
        text_color = RED if self.current_player_id == 0 else BLUE
        if current_player.ai_type == "Human Player":
            status_text = f"Current Turn: {player_name}"
        else:
            ai_type = self._get_player_type_name(current_player)
            status_text = f"Current Turn: {player_name} ({ai_type})"
            # If it's AI's turn and waiting for delay
            if (time.time() - self._last_human_move_time) < self.AI_DELAY_SECONDS:
                status_text += " - Thinking..." # Or any other message you want to display

        text_surface = font.render(status_text, True, text_color)
        screen.blit(text_surface, (10, SCREEN_HEIGHT - STATUS_BAR_HEIGHT + 10))
        
        red_type = self._get_player_type_name(self.players[0])
        blue_type = self._get_player_type_name(self.players[1])
        
        info_text = f"Red: {red_type} vs Blue: {blue_type}"
        info_surface = small_font.render(info_text, True, GREEN)
        screen.blit(info_surface, (160, SCREEN_HEIGHT - STATUS_BAR_HEIGHT + 40))

    def _check_game_over(self):
        """Checks for win/loss conditions and terminates the game if met."""
        red_pieces = self.board.get_player_pieces(0)
        blue_pieces = self.board.get_player_pieces(1)

        if not red_pieces:
            red_type = self._get_player_type_name(self.players[0])
            blue_type = self._get_player_type_name(self.players[1])
            self._game_over(f"Blue ({blue_type}) wins against Red ({red_type})!")
        elif not blue_pieces:
            red_type = self._get_player_type_name(self.players[0])
            blue_type = self._get_player_type_name(self.players[1])
            self._game_over(f"Red ({red_type}) wins against Blue ({blue_type})!")
        elif len(red_pieces) == 1 and len(blue_pieces) == 1:
            rr, rc = red_pieces[0]
            br, bc = blue_pieces[0]
            if not self.board.is_adjacent((rr, rc), (br, bc)):
                red_type = self._get_player_type_name(self.players[0])
                blue_type = self._get_player_type_name(self.players[1])
                self._game_over(f"Draw! Red ({red_type}) vs Blue ({blue_type}) - Last pieces not adjacent")

    def _game_over(self, message):
        """Displays game over message and exits."""
        print(message)
        self.running = False # Stop the main game loop

    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()

        while self.running:
            if self.mode == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    else:
                        current_player_obj = self.players[self.current_player_id]
                        if current_player_obj.ai_type == "Human Player":
                            if current_player_obj.handle_event(event, self.board):
                                self.current_player_id = 1 - self.current_player_id # Switch turn
                                self._last_human_move_time = time.time() # Record the time of human's last move

            # AI Player's turn logic
            current_player_obj = self.players[self.current_player_id]
            if current_player_obj.ai_type != "Human Player":
                # Check if enough time has passed since the human's last move
                if (time.time() - self._last_human_move_time) >= self.AI_DELAY_SECONDS:
                    if current_player_obj.take_turn(self.board):
                        self.current_player_id = 1 - self.current_player_id # Switch turn
                        # No need for time.sleep here as we already handled the delay
                        # but you could add a very short one if AI move is too fast visually
                        # time.sleep(0.1)
            
                        if self.mode == 1:
                            self._last_human_move_time = time.time()

            self._draw_board()
            pygame.display.flip()
            self._check_game_over()
            clock.tick(30)

        pygame.quit()
        sys.exit()

# --- Run the Game ---
if __name__ == "__main__":
    game = Game()
    game.run()