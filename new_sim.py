import pygame
import random
import sys
import time

# 游戏设置
ROWS, COLS = 7, 8
TILE_SIZE = 80
SCREEN_WIDTH, SCREEN_HEIGHT = COLS * TILE_SIZE, ROWS * TILE_SIZE + 40

# 颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 50, 50)
BLUE = (50, 50, 220)
GREY = (180, 180, 180)
YELLOW = (255, 255, 0)

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("简化斗兽棋")
font = pygame.font.SysFont(None, 36)

# 棋子类
class Piece:
    def __init__(self, player, strength):
        self.player = player  # 0 = 红方，1 = 蓝方
        self.strength = strength
        self.revealed = False

# 初始化棋盘
def create_board():
    board = [[None for _ in range(COLS)] for _ in range(ROWS)]

    all_pieces = [Piece(player, strength)
                  for player in [0, 1]
                  for strength in range(1, 9)]

    random.shuffle(all_pieces)
    positions = [(r, c) for r in range(ROWS) for c in range(COLS) if r < 1 or r > 5]
    random.shuffle(positions)

    for piece, (r, c) in zip(all_pieces, positions[:16]):
        board[r][c] = piece

    return board

board = create_board()
selected = None
current_player = 0

def draw_board():
    screen.fill(WHITE)
    for i in range(ROWS):
        for j in range(COLS):
            rect = pygame.Rect(j*TILE_SIZE, i*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            piece = board[i][j]
            if piece:
                if piece.revealed:
                    color = RED if piece.player == 0 else BLUE
                    pygame.draw.circle(screen, color, rect.center, TILE_SIZE//3)
                    text = font.render(str(piece.strength), True, WHITE)
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)
                else:
                    pygame.draw.rect(screen, GREY, rect.inflate(-10, -10))

    if selected:
        i, j = selected
        rect = pygame.Rect(j*TILE_SIZE, i*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(screen, YELLOW, rect, 3)

    status = f"Now: {'Red' if current_player == 0 else 'Blue'}'s turn"
    text_color = RED if current_player == 0 else BLUE
    text = font.render(status, True, text_color)
    screen.blit(text, (10, SCREEN_HEIGHT - 35))

def is_adjacent(pos1, pos2):
    r1, c1 = pos1
    r2, c2 = pos2
    return abs(r1 - r2) + abs(c1 - c2) == 1

def try_move(start, end):
    sr, sc = start
    er, ec = end
    p1 = board[sr][sc]
    p2 = board[er][ec]

    if p2 is None:
        board[er][ec] = p1
        board[sr][sc] = None
        return True
    elif not p2.revealed:
        p2.revealed = True
        if p2.player == p1.player:
            return True
        if p1.strength >= p2.strength or (p1.strength == 1 and p2.strength == 8):
            board[er][ec] = p1
            board[sr][sc] = None
        else:
            board[sr][sc] = None
        return True
    elif p2.player != p1.player:
        if p1.strength >= p2.strength or (p1.strength == 1 and p2.strength == 8):
            board[er][ec] = p1
            board[sr][sc] = None
            return True
        if p1.strength < p2.strength or (p1.strength == 8 and p2.strength == 1):
            board[sr][sc] = None
            return True
    return False

def if_terminate():
    red = [(r, c) for r in range(ROWS) for c in range(COLS) if board[r][c] and board[r][c].player == 0]
    blue = [(r, c) for r in range(ROWS) for c in range(COLS) if board[r][c] and board[r][c].player == 1]

    if not red:
        print("Blue wins!")
        pygame.quit()
        sys.exit()
    if not blue:
        print("Red wins!")
        pygame.quit()
        sys.exit()
    if len(red) == 1 and len(blue) == 1:
        rr, rc = red[0]
        br, bc = blue[0]
        if not is_adjacent((rr, rc), (br, bc)):
            print("Draw!")
            pygame.quit()
            sys.exit()

class HumanControl:
    def __init__(self, player):
        self.player = player
        self.selected = None

    def handle_event(self, event):
        global current_player,selected

        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            row, col = y // TILE_SIZE, x // TILE_SIZE

            if row >= ROWS or col >= COLS:
                return

            piece = board[row][col]
            if self.selected:
                if is_adjacent(self.selected, (row, col)):
                    moved = try_move(self.selected, (row, col))
                    if moved:
                        self.selected = None
                        selected = None
                        current_player = 1 - current_player
                    else:
                        self.selected = None
                        selected = None
                else:
                    self.selected = None
                    selected = None
            elif piece:
                if not piece.revealed:
                    piece.revealed = True
                    current_player = 1 - current_player
                elif piece.player == self.player:
                    self.selected = (row, col)
                    selected = (row, col)

class RandomPlayer:
    def __init__(self, player):
        self.player = player

    def step(self):
        global current_player

        actions = []

        for r in range(ROWS):
            for c in range(COLS):
                piece = board[r][c]
                if piece and piece.player == self.player:
                    if not piece.revealed:
                        actions.append(("reveal", (r, c)))
                    elif piece.revealed:
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < ROWS and 0 <= nc < COLS:
                                actions.append(("move", (r, c), (nr, nc)))

        random.shuffle(actions)
        for action in actions:
            if action[0] == "reveal":
                r, c = action[1]
                board[r][c].revealed = True
                current_player = 1 - current_player
                return
            elif action[0] == "move":
                start, end = action[1], action[2]
                if is_adjacent(start, end) and board[start[0]][start[1]]:
                    if try_move(start, end):
                        current_player = 1 - current_player
                        return

class HeuristicPlayer:
    def __init__(self,player):
        self.player = player
    def eva(self,board):
        score = 0



# 初始化控制器
red_player = HumanControl(0)
blue_player = RandomPlayer(1)

# 主循环
running = True
clock = pygame.time.Clock()

while running:
    draw_board()
    pygame.display.flip()
    if_terminate()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif current_player == 0:
            red_player.handle_event(event)


    if current_player == 1:

        blue_player.step()

    
    if current_player == 1:
        # print("shit")
        time.sleep(2000)
    clock.tick(30)

pygame.quit()
sys.exit()
