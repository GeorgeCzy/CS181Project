import pygame
import random
import sys

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

# 初始化棋盘：红蓝各8个棋子，随机分布于上下两端
def create_board():
    board = [[None for _ in range(COLS)] for _ in range(ROWS)]

    # 每方只有一个1~8棋子
    all_pieces = [Piece(player, strength)
                  for player in [0, 1]
                  for strength in range(1, 9)]  # 每方8个，不重复

    random.shuffle(all_pieces)

    # 可用位置只在上两行和下两行
    positions = [(r, c) for r in range(ROWS) for c in range(COLS) if r < 1 or r > 5]
    random.shuffle(positions)

    for piece, (r, c) in zip(all_pieces, positions[:16]):
        board[r][c] = piece

    return board

board = create_board()
selected = None
current_player = 0  # 0 = 红方，1 = 蓝方

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

    # 当前玩家信息（颜色提示）
    status = f"now: {"red's turn" if current_player == 0 else "blue's turn"}"
    text_color = RED if current_player == 0 else BLUE
    text = font.render(status, True, text_color)
    screen.blit(text, (10, SCREEN_HEIGHT - 35))

def is_adjacent(pos1, pos2):
    r1, c1 = pos1
    r2, c2 = pos2
    return abs(r1 - r2) + abs(c1 - c2) == 1

def try_move(start, end):
    global current_player
    sr, sc = start
    er, ec = end
    p1 = board[sr][sc]
    p2 = board[er][ec]

    if p2 is None:
        board[er][ec] = p1
        board[sr][sc] = None
        return True
    elif not p2.revealed:
        p2.revealed = True  # 翻开目标棋子
        if p2.player == p1.player:
            return True  # 相当于自己翻开一次
        if p1.strength >= p2.strength or (p1.strength == 1 and p2.strength == 8):
            board[er][ec] = p1
            board[sr][sc] = None
        else:
            board[sr][sc] = None  # 被吃掉
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
    red_count = sum(1 for row in board for piece in row if piece and piece.player == 0)
    blue_count = sum(1 for row in board for piece in row if piece and piece.player == 1)
    red_pieces = [(r, c) for r in range(ROWS) for c in range(COLS)
                  if board[r][c] and board[r][c].player == 0]
    blue_pieces = [(r, c) for r in range(ROWS) for c in range(COLS)
                   if board[r][c] and board[r][c].player == 1]

    if red_count == 0 or blue_count == 0:
        winner = "red" if blue_count == 0 else "blue"
        print(f"{winner}wins!")
        pygame.quit()
        sys.exit()
    elif red_count == 1 and blue_count == 1:
        r_pos = red_pieces[0]
        b_pos = blue_pieces[0]
        if not is_adjacent(r_pos, b_pos):
            print("Draw! Both sides have only one piece and they are not adjacent.")
            pygame.quit()
            sys.exit()

# 主循环
running = True
while running:
    draw_board()
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            row, col = y // TILE_SIZE, x // TILE_SIZE

            if row >= ROWS or col >= COLS:
                continue

            piece = board[row][col]
            if selected:
                if is_adjacent(selected, (row, col)):
                    moved = try_move(selected, (row, col))
                    if moved:
                        selected = None
                        current_player = 1 - current_player
                    else:
                        selected = None
                else:
                    selected = None
            elif piece:
                if not piece.revealed:
                    piece.revealed = True  # 任何未翻开的棋子都可以翻开
                    current_player = 1 - current_player
                elif piece.player == current_player:
                    selected = (row, col)
    
    if_terminate()

pygame.quit()
sys.exit()
