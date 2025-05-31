import pygame
import random
import sys
import time

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
pygame.display.set_caption("简化斗兽棋")
font = pygame.font.SysFont(None, 36)

# --- Game Classes ---

class Piece:
    """Represents a single game piece with player and strength."""
    def __init__(self, player, strength):
        self.player = player  # 0 = Red, 1 = Blue
        self.strength = strength
        self.revealed = False

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
                if r < 1 or r > ROWS - 2: # Corrected from original: Rows 0, 1, 5, 6
                    valid_positions.append((r, c))

        random.shuffle(valid_positions)

        for piece, (r, c) in zip(all_pieces, valid_positions[:len(all_pieces)]): # Ensure we only use as many positions as pieces
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
        """
        sr, sc = start_pos
        er, ec = end_pos
        piece_moving = self.get_piece(sr, sc)
        piece_at_target = self.get_piece(er, ec)

        if not piece_moving:
            return False

        # Move to an empty square
        if piece_at_target is None:
            self.set_piece(er, ec, piece_moving)
            self.set_piece(sr, sc, None)
            return True
        # Target piece is unrevealed
        elif not piece_at_target.revealed:
            piece_at_target.revealed = True
            if piece_at_target.player == piece_moving.player:
                # Cannot capture own unrevealed piece, just reveal and end turn
                return True
            else:
                # Capture unrevealed opponent's piece
                if piece_moving.strength >= piece_at_target.strength or \
                   (piece_moving.strength == 1 and piece_at_target.strength == 8): # Rat captures Elephant
                    self.set_piece(er, ec, piece_moving)
                    self.set_piece(sr, sc, None)
                else: # Opponent captures moving piece
                    self.set_piece(sr, sc, None)
                return True
        # Target piece is revealed and belongs to opponent
        elif piece_at_target.player != piece_moving.player:
            if piece_moving.strength >= piece_at_target.strength or \
               (piece_moving.strength == 1 and piece_at_target.strength == 8): # Rat captures Elephant
                self.set_piece(er, ec, piece_moving)
                self.set_piece(sr, sc, None)
                return True
            elif piece_moving.strength < piece_at_target.strength or \
                 (piece_moving.strength == 8 and piece_at_target.strength == 1): # Elephant cannot capture Rat
                self.set_piece(sr, sc, None) # Moving piece is captured
                return True
        return False

    def get_player_pieces(self, player_id):
        """Returns a list of positions for a given player's pieces."""
        return [(r, c) for r in range(ROWS) for c in range(COLS)
                if self.board[r][c] and self.board[r][c].player == player_id]

class Player:
    """Base class for players."""
    def __init__(self, player_id):
        self.player_id = player_id

    def handle_event(self, event, board):
        """Handles Pygame events (e.g., mouse clicks for human players)."""
        raise NotImplementedError

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
                if board.is_adjacent(self.selected_piece_pos, (row, col)):
                    moved = board.try_move(self.selected_piece_pos, (row, col))
                    if moved:
                        self.selected_piece_pos = None
                        return True # Move successful, turn ends
                    else:
                        # Move failed, deselect
                        self.selected_piece_pos = None
                else:
                    # Not adjacent, deselect current and try to select new
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
                elif piece.player == self.player_id:
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
        possible_actions = []

        # Find all possible reveal actions
        for r in range(ROWS):
            for c in range(COLS):
                piece = board.get_piece(r, c)
                if piece and piece.player == self.player_id and not piece.revealed:
                    possible_actions.append(("reveal", (r, c)))

        # Find all possible move actions
        for r in range(ROWS):
            for c in range(COLS):
                piece = board.get_piece(r, c)
                if piece and piece.player == self.player_id and piece.revealed:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < ROWS and 0 <= nc < COLS:
                            # Create a temporary board for speculative move
                            # Deep copy board state to avoid modifying actual game board during checks
                            temp_board_data = [[p for p in row] for row in board.board] # Shallow copy pieces, deep copy rows
                            
                            # Create a temporary Board object to use its try_move method
                            # Note: This is a bit heavy, a lighter check might be better for performance
                            # but for correctness with try_move's complex logic, it's safer.
                            temp_board_obj = Board()
                            temp_board_obj.board = temp_board_data # Assign the copied board data

                            # The `try_move` method modifies the board. To check if a move is valid
                            # without actually making it on the main board, we need to restore the state
                            # or use a check that doesn't modify the board.
                            # For simplicity here, we'll try the move on the temporary board.
                            # A more optimized approach would be to have a `can_move` method in Board.
                            
                            original_piece_moving = temp_board_obj.get_piece(r,c)
                            original_piece_at_target = temp_board_obj.get_piece(nr,nc)
                            
                            # Perform the speculative move check
                            if original_piece_moving and board.is_adjacent((r,c),(nr,nc)):
                                # Manual check for validity without changing board
                                # This is a simplified check, try_move handles full logic
                                # For a robust AI, you'd likely want a `is_valid_move` helper
                                is_move_potentially_valid = False
                                if original_piece_at_target is None:
                                    is_move_potentially_valid = True
                                elif not original_piece_at_target.revealed:
                                    is_move_potentially_valid = True # Can always reveal/capture unrevealed
                                elif original_piece_at_target.player != original_piece_moving.player:
                                    # Opponent's revealed piece
                                    if original_piece_moving.strength >= original_piece_at_target.strength or \
                                       (original_piece_moving.strength == 1 and original_piece_at_target.strength == 8):
                                        is_move_potentially_valid = True
                                    elif original_piece_moving.strength < original_piece_at_target.strength or \
                                         (original_piece_moving.strength == 8 and original_piece_at_target.strength == 1):
                                        is_move_potentially_valid = True
                                
                                if is_move_potentially_valid:
                                    possible_actions.append(("move", (r, c), (nr, nc)))


        random.shuffle(possible_actions)

        for action in possible_actions:
            if action[0] == "reveal":
                r, c = action[1]
                board.get_piece(r, c).revealed = True
                return True # Turn ended
            elif action[0] == "move":
                start_pos, end_pos = action[1], action[2]
                if board.try_move(start_pos, end_pos):
                    return True # Turn ended
        return False # No valid moves found

class Game:
    """Main class to manage the game flow."""
    def __init__(self):
        self.board_manager = Board()
        self.players = {
            0: HumanPlayer(0), # Red
            1: RandomPlayer(1)  # Blue
            # 1: HumanPlayer(1) # For two human players
        }
        self.current_player_id = 0
        self.running = True
        self._last_human_move_time = 0 # Track when human player last made a move
        self.AI_DELAY_SECONDS = 1.5 # The delay for AI moves

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

        # Highlight selected piece for human player
        human_player = self.players[0] # Human player is always player 0 in this setup
        if isinstance(human_player, HumanPlayer) and human_player.get_selected_pos():
            i, j = human_player.get_selected_pos()
            rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, YELLOW, rect, 3)

        # Draw status bar
        status_bar_rect = pygame.Rect(0, SCREEN_HEIGHT - STATUS_BAR_HEIGHT, SCREEN_WIDTH, STATUS_BAR_HEIGHT)
        pygame.draw.rect(screen, BLACK, status_bar_rect) # Background for status bar

        player_name = "Red" if self.current_player_id == 0 else "Blue"
        text_color = RED if self.current_player_id == 0 else BLUE
        status_text = f"Current Turn: {player_name}"

        # If it's AI's turn and waiting for delay
        if self.current_player_id == 1 and (time.time() - self._last_human_move_time) < self.AI_DELAY_SECONDS:
             status_text += " (AI thinking...)" # Or any other message you want to display

        text_surface = font.render(status_text, True, text_color)
        screen.blit(text_surface, (10, SCREEN_HEIGHT - STATUS_BAR_HEIGHT + 5))

    def _check_game_over(self):
        """Checks for win/loss conditions and terminates the game if met."""
        red_pieces = self.board_manager.get_player_pieces(0)
        blue_pieces = self.board_manager.get_player_pieces(1)

        if not red_pieces:
            self._game_over("Blue wins!")
        elif not blue_pieces:
            self._game_over("Red wins!")
        elif len(red_pieces) == 1 and len(blue_pieces) == 1:
            rr, rc = red_pieces[0]
            br, bc = blue_pieces[0]
            if not self.board_manager.is_adjacent((rr, rc), (br, bc)):
                self._game_over("Draw! (Last pieces not adjacent)")

    def _game_over(self, message):
        """Displays game over message and exits."""
        print(message)
        self.running = False # Stop the main game loop

    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                else:
                    current_player_obj = self.players[self.current_player_id]
                    if isinstance(current_player_obj, HumanPlayer):
                        if current_player_obj.handle_event(event, self.board_manager):
                            self.current_player_id = 1 - self.current_player_id # Switch turn
                            self._last_human_move_time = time.time() # Record the time of human's last move

            # AI Player's turn logic
            current_player_obj = self.players[self.current_player_id]
            if isinstance(current_player_obj, (RandomPlayer)):
                # Check if enough time has passed since the human's last move
                if (time.time() - self._last_human_move_time) >= self.AI_DELAY_SECONDS:
                    if current_player_obj.take_turn(self.board_manager):
                        self.current_player_id = 1 - self.current_player_id # Switch turn
                        # No need for time.sleep here as we already handled the delay
                        # but you could add a very short one if AI move is too fast visually
                        # time.sleep(0.1)

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