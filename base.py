import pygame
import random
import copy

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
GREEN = (0, 255, 0)
LIGHT_GREY = (211, 211, 211)
DARK_GREY = (169, 169, 169)

pygame.init()
pygame.font.init() # Ensure font module is initialized
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("简化斗兽棋 - AI 对抗")
font = pygame.font.SysFont(None, 36)
small_font = pygame.font.SysFont(None, 24)

# --- Game Classes ---

class Piece:
    """Represents a single game piece with player and strength."""
    def __init__(self, player, strength):
        self.player = player  # 0 = Red, 1 = Blue
        self.strength = strength
        self.revealed = False
    
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