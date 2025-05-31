import pygame
import random
import sys
import time
import copy # 导入 copy 模块用于深拷贝

# --- Constants and Initialization ---
# Game Settings
ROWS, COLS = 7, 8
TILE_SIZE = 80
SCREEN_WIDTH, SCREEN_HEIGHT = COLS * TILE_SIZE, ROWS * TILE_SIZE + 40
STATUS_BAR_HEIGHT = 40 # Added for clarity

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 50, 50)
BLUE = (50, 50, 220)
GREY = (180, 180, 180)
YELLOW = (255, 255, 0)

pygame.init()
pygame.font.init() # Ensure font module is initialized
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("简化斗兽棋 - AI 对抗")
font = pygame.font.SysFont(None, 36)

# --- Game Classes ---

class Piece:
    """Represents a single game piece with player and strength."""
    def __init__(self, player, strength):
        self.player = player  # 0 = Red, 1 = Blue
        self.strength = strength
        self.revealed = False
    
    # 为了深拷贝 Piece 对象，我们需要实现 __copy__ 或 __deepcopy__
    # 对于这个简单的类，浅拷贝它的属性通常就足够了，但为了严谨，可以实现深拷贝
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

class Board:
    """Manages the game board state and piece placement."""
    def __init__(self):
        self.board = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self._initialize_pieces()

    def _initialize_pieces(self):
        """Initializes and shuffles all pieces on the board."""
        all_pieces = [Piece(player, strength)
                      for player in [0, 1]
                      for strength in range(1, 9)]
        random.shuffle(all_pieces)

        # Pieces are placed only on the top and bottom two rows
        valid_positions = []
        for r in range(ROWS):
            for c in range(COLS):
                # 初始布局修改为只放在最顶上两行和最底下两行 (0,1, ROWS-2, ROWS-1)
                if r < 1 or r >= ROWS - 1: # Rows 0, 1, 5, 6 (for ROWS=7)
                    valid_positions.append((r, c))

        random.shuffle(valid_positions)

        # 确保只放置与 pieces 数量相符的棋子
        # 你的 all_pieces 列表有 16 个棋子 (2 玩家 * 8 强度)
        # valid_positions 应该至少有 16 个位置。ROWS 7 * COLS 8 = 56
        # r < 2 (rows 0, 1) -> 2 * 8 = 16 positions
        # r >= ROWS - 2 (rows 5, 6) -> 2 * 8 = 16 positions
        # 总共 32 个位置。所以这里的 valid_positions[:len(all_pieces)] 是正确的。
        for piece, (r, c) in zip(all_pieces, valid_positions[:len(all_pieces)]):
            self.board[r][c] = piece

    def get_piece(self, row, col):
        """Returns the piece at the given coordinates."""
        if 0 <= row < ROWS and 0 <= col < COLS:
            return self.board[row][col]
        return None

    def set_piece(self, row, col, piece):
        """Sets a piece at the given coordinates."""
        if 0 <= row < ROWS and 0 <= col < COLS:
            self.board[row][col] = piece

    def is_adjacent(self, pos1, pos2):
        """Checks if two positions are adjacent (horizontal or vertical)."""
        r1, c1 = pos1
        r2, c2 = pos2
        return abs(r1 - r2) + abs(c1 - c2) == 1

    def try_move(self, start_pos, end_pos):
        """
        Attempts to move a piece from start_pos to end_pos, handling captures.
        Returns True if the move was successful, False otherwise.
        Note: This method modifies the board directly.
        """
        sr, sc = start_pos
        er, ec = end_pos
        piece_moving = self.get_piece(sr, sc)
        piece_at_target = self.get_piece(er, ec)

        # 1. 移动棋子不存在
        if not piece_moving:
            return False

        # 2. 目标位置不是相邻的
        if not self.is_adjacent(start_pos, end_pos):
            return False

        # 3. 目标位置为空
        if piece_at_target is None:
            self.set_piece(er, ec, piece_moving)
            self.set_piece(sr, sc, None)
            return True
        # 4. 目标位置有棋子
        else:
            # 4a. 目标棋子未翻开
            if not piece_at_target.revealed:
                piece_at_target.revealed = True # 翻开目标棋子
                # 如果是自己的未翻开棋子，只是翻开，不能移动过去。
                # 但仍然算作一回合有效动作。
                if piece_at_target.player == piece_moving.player:
                    # 自己的棋子不能移动到有自己未翻开棋子的位置，只翻开
                    return True # 翻开即结束本回合
                else:
                    # 如果是对手的未翻开棋子，进行捕获判断（强度比较）
                    if (piece_moving.strength > piece_at_target.strength and not (piece_moving.strength == 8 and piece_at_target.strength == 1)) or \
                       (piece_moving.strength == 1 and piece_at_target.strength == 8): # 鼠吃象
                        self.set_piece(er, ec, piece_moving)
                        self.set_piece(sr, sc, None)
                        return True
                    elif piece_moving.strength == piece_at_target.strength: # 同强度
                        self.set_piece(sr, sc, None)
                        self.set_piece(er, ec, None) 
                        return True
                    else: # 移动方被吃
                        self.set_piece(sr, sc, None)
                        return True # 完成捕获，结束本回合
            # 4b. 目标棋子已翻开
            else:
                # 4b-i. 目标棋子是自己的棋子 (已翻开)
                if piece_at_target.player == piece_moving.player:
                    return False # 不能吃自己的棋子，也不能移动到有自己已翻开棋子的格子

                # 4b-ii. 目标棋子是对手的棋子 (已翻开)
                else:
                    if (piece_moving.strength > piece_at_target.strength and not (piece_moving.strength == 8 and piece_at_target.strength == 1)) or \
                       (piece_moving.strength == 1 and piece_at_target.strength == 8): # 鼠吃象
                        self.set_piece(er, ec, piece_moving)
                        self.set_piece(sr, sc, None)
                        return True
                    elif piece_moving.strength < piece_at_target.strength or \
                         (piece_moving.strength == 8 and piece_at_target.strength == 1): # 象不能吃鼠
                        self.set_piece(sr, sc, None) # 移动方被吃
                        return True
                    else: # 同强度
                        self.set_piece(sr, sc, None)
                        self.set_piece(er, ec, None) # 两个棋子同强度
                        return True
        return False # 默认返回 False，表示移动未成功

    def get_player_pieces(self, player_id):
        """Returns a list of positions for a given player's pieces."""
        return [(r, c) for r in range(ROWS) for c in range(COLS)
                if self.board[r][c] and self.board[r][c].player == player_id]

    def get_all_possible_moves(self, player_id):
        """
        Generates all legal moves and reveal actions for the given player.
        Returns a list of tuples: ("reveal", (r, c)) or ("move", (sr, sc), (er, ec))
        Each action represents a valid way to end a turn.
        """
        possible_actions = []

        for r in range(ROWS):
            for c in range(COLS):
                piece = self.get_piece(r, c)
                if piece and piece.player == player_id:
                    if not piece.revealed:
                        # 可以翻开自己的未翻开棋子
                        possible_actions.append(("reveal", (r, c)))
                    else:
                        # 如果已翻开，可以尝试移动
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < ROWS and 0 <= nc < COLS:
                                # 对于移动动作，我们模拟执行，看是否是有效回合
                                # 这里深拷贝是为了 Minimax 模拟，不是给 RandomPlayer 用。
                                # RandomPlayer 在 take_turn 中会直接在实际 board 上尝试。
                                # 但这里生成的是所有可能的“合法”动作，所以模拟 check 是合理的。
                                temp_board_for_check = copy.deepcopy(self) # 模拟此动作
                                if temp_board_for_check.try_move((r, c), (nr, nc)):
                                    possible_actions.append(("move", (r, c), (nr, nc)))
        return possible_actions


class Player:
    """Base class for players."""
    def __init__(self, player_id):
        self.player_id = player_id

    def handle_event(self, event, board):
        """Handles Pygame events (e.g., mouse clicks for human players)."""
        # 对于AI玩家，这个方法通常不需要实现或直接返回False
        return False

    def take_turn(self, board):
        """Executes a turn for AI players."""
        raise NotImplementedError

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


class Game:
    """Main class to manage the game flow."""
    def __init__(self, agent=None, base_agent=None): # 这里我改了，但是不影响原本用法
        self.board_manager = Board()
        if agent is None:
            self.players = {
                # 修改这里，将人类玩家替换为AI玩家
                # 0: MinimaxPlayer(0, max_depth=3), # 红色AI
                0: RandomPlayer(0), # 红色AI
                1: MinimaxPlayer(1, max_depth=3)  # 蓝色AI
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
                pygame.draw.rect(screen, BLACK, rect, 1) # Cell border

                piece = self.board_manager.get_piece(i, j)
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
        red_pieces = self.board_manager.get_player_pieces(0)
        blue_pieces = self.board_manager.get_player_pieces(1)

        if not red_pieces:
            self._game_over("Blue AI wins!")
            return True
        elif not blue_pieces:
            self._game_over("Red AI wins!")
            return True
        # 更精确的和棋判断：只剩一颗棋子且无法互相捕获或不相邻
        elif len(red_pieces) == 1 and len(blue_pieces) == 1:
            rr, rc = red_pieces[0]
            br, bc = blue_pieces[0]
            red_piece = self.board_manager.get_piece(rr, rc)
            blue_piece = self.board_manager.get_piece(br, bc)

            # 只有当两个棋子都已翻开时，才能判断是否能互相捕获
            if red_piece.revealed and blue_piece.revealed:
                can_red_attack = (red_piece.strength > blue_piece.strength) or \
                                 (red_piece.strength == 1 and blue_piece.strength == 8)
                can_blue_attack = (blue_piece.strength > red_piece.strength) or \
                                  (blue_piece.strength == 1 and red_piece.strength == 8)
                
                # 如果它们相邻，且双方都无法捕获对方，则为和棋
                if self.board_manager.is_adjacent((rr, rc), (br, bc)):
                    if not can_red_attack and not can_blue_attack:
                        self._game_over("Draw! (Stalemate - last pieces cannot capture each other)")
                        return True
                else: # 如果不相邻，也无法捕获，则为和棋
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

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                # 移除了人类玩家事件处理逻辑

            current_player_obj = self.players[self.current_player_id]
            
            # AI 玩家的回合逻辑
            # AI 不依赖事件，直接在循环中执行
            time.sleep(self.AI_DELAY_SECONDS) # 增加延迟，方便观察

            if current_player_obj.take_turn(self.board_manager):
                # 仅在 AI 成功执行动作后切换回合
                if self._check_game_over(): # 立即检查游戏是否结束
                    self.running = False
                else:
                    self.current_player_id = 1 - self.current_player_id # 切换回合
            else:
                # 如果AI没有找到任何合法动作（极少发生，通常意味着游戏结束了）
                # 可以在这里添加一个平局判断或者其他处理
                print(f"Player {self.current_player_id} could not make a valid move. Game might be stuck or draw.")
                self._game_over("Draw! (No valid moves left for current player)")
                self.running = False # 结束游戏

            self._draw_board()
            pygame.display.flip()
            clock.tick(30) # 控制帧率

        pygame.quit()
        sys.exit()

# --- Run the Game ---
if __name__ == "__main__":
    game = Game()
    game.run()