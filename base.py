import pygame
import random
import copy
import time
import sys
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Callable
from collections import deque
import os
import pickle

# --- Constants and Initialization ---
# Game Settings
ROWS, COLS = 7, 8
TILE_SIZE = 80
SCREEN_WIDTH, SCREEN_HEIGHT = COLS * TILE_SIZE, ROWS * TILE_SIZE + 40
STATUS_BAR_HEIGHT = 40  # Added for clarity

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

MAX_STEPS = 1000  # 最大允许步数

# --- Game Classes ---


class Piece:
    """Represents a single game piece with player and strength."""

    def __init__(self, player, strength):
        self._player = player  # 0 = Red, 1 = Blue
        self._strength = strength
        self._revealed = False

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    @property # 使用@property装饰器使得player, strength, reveal属性只读
    def player(self):
        """read only"""
        if self._revealed:
            return self._player
        return -1

    @property
    def strength(self):
        """read only"""
        if self._revealed:
            return self._strength
        return -1

    @property
    def revealed(self):
        """read only"""
        return self._revealed

    def get_player(self, Administrator=False): # 这两个是给board用的，能够看到所有的棋子
        """
        Returns the player ID of this piece.
        return -1 if the piece is not revealed.
        """
        if Administrator:
            return self._player
        return -1

    def get_strength(self, Administrator=False):
        """Returns the strength of this piece."""
        if Administrator:
            return self._strength
        return -1

    def reveal(self): # 只通过这个方法更改 piece 的 revealed 状态
        """only way to reveal"""
        self._revealed = True


def compare_strength(self_strength, other_strength):
    """
    Determines if this piece can capture another piece.
    return -1: cannot capture, 0: same strength, 1: can capture
    -2: one of the pieces is unrevealed.
    """
    if self_strength == -1 or other_strength == -1: # 如果其中一个没有翻开，无法比较
        return -2
    if self_strength == 8 and other_strength == 1:
        return -1
    if self_strength > other_strength:
        return 1
    if self_strength == 1 and other_strength == 8:
        return 1
    if self_strength == other_strength:
        return 0
    return -1  # Cannot capture


class Board:
    """Manages the game board state and piece placement."""

    def __init__(self):
        self.board = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self._initialize_pieces()
        self._died = {0: [], 1: []}  # Track died pieces for each player
        # 增加了 died 字典来收纳被吃掉的棋子，用于解决某些问题？我忘了为啥要了，但是后面用了

    def _initialize_pieces(self):
        """Initializes and shuffles all pieces on the board."""
        pieces = [(i, j) for i in range(2) for j in range(1, 9)] # 这个做了下简化，但基本没变
        random.shuffle(pieces)
        for i in [0, 6]:
            for j in range(8):
                chosen = pieces.pop()
                self.board[i][j] = Piece(chosen[0], chosen[1])

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

        # 1. 移动棋子不存在或未翻开
        if not piece_moving or not piece_moving.revealed:
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
                piece_at_target.reveal()  # 翻开目标棋子
                # 如果是自己的未翻开棋子，只是翻开，不能移动过去。
                # 但仍然算作一回合有效动作。
                if piece_at_target.get_player(Administrator=True) == piece_moving.get_player(Administrator=True):
                    # 自己的棋子不能移动到有自己未翻开棋子的位置，只翻开
                    return True  # 翻开即结束本回合
                else:
                    # 如果是对手的未翻开棋子，进行捕获判断（强度比较）
                    compare = compare_strength(
                        piece_moving.get_strength(Administrator=True),
                        piece_at_target.get_strength(Administrator=True),
                    )
                    if compare == 1:  # 可以捕获
                        self.set_piece(er, ec, piece_moving)
                        self.set_piece(sr, sc, None)
                        piece_at_target.reveal()  # 翻开被吃的棋子
                        self._died[1 - piece_moving.get_player(Administrator=True)].append(
                            piece_at_target
                        )
                        return True
                    elif compare == 0:  # 同强度
                        self.set_piece(sr, sc, None)
                        self.set_piece(er, ec, None)  # 两个棋子同强度
                        piece_at_target.reveal()  # 翻开被吃的棋子
                        self._died[1 - piece_moving.get_player(Administrator=True)].append(
                            piece_at_target
                        )
                        self._died[piece_moving.get_player(Administrator=True)].append(piece_moving)
                        return True
                    elif compare == -1:  # 移动方被吃
                        self.set_piece(sr, sc, None)  # 移动方被吃
                        self._died[piece_moving.get_player(Administrator=True)].append(piece_moving)
                        return True  # 完成捕获，结束本回合

            # 4b. 目标棋子已翻开
            else:
                # 4b-i. 目标棋子是自己的棋子 (已翻开)
                if piece_at_target.get_player(Administrator=True) == piece_moving.get_player(Administrator=True):
                    return False  # 不能吃自己的棋子，也不能移动到有自己已翻开棋子的格子

                # 4b-ii. 目标棋子是对手的棋子 (已翻开)
                else:
                    compare = compare_strength(
                        piece_moving.get_strength(Administrator=True),
                        piece_at_target.get_strength(Administrator=True),
                    )
                    if compare == 1:
                        self.set_piece(er, ec, piece_moving)
                        self.set_piece(sr, sc, None)
                        self._died[piece_at_target.get_player(Administrator=True)].append(
                            piece_at_target
                        )
                        return True
                    elif compare == 0:  # 同强度
                        self.set_piece(sr, sc, None)
                        self.set_piece(er, ec, None)
                        self._died[piece_at_target.get_player(Administrator=True)].append(
                            piece_at_target
                        )
                        self._died[piece_moving.get_player(Administrator=True)].append(piece_moving)
                        return True
                    elif compare == -1:  # 移动方被吃
                        self.set_piece(sr, sc, None)
                        self._died[piece_moving.get_player(Administrator=True)].append(piece_moving)
                        return True  # 完成捕获，结束本回合

        return False  # 默认返回 False，表示移动未成功

    def get_player_pieces(self, player_id):
        """Returns a list of positions for a given player's pieces."""
        return [
            (r, c)
            for r in range(ROWS)
            for c in range(COLS)
            if self.board[r][c] and self.board[r][c].player == player_id
        ]
    
    def get_unveal_pieces(self):
        """Returns a list of positions for all unrevealed pieces."""
        return [
            (r, c)
            for r in range(ROWS)
            for c in range(COLS)
            if self.board[r][c] and not self.board[r][c].revealed
        ]
        
    def get_all_pieces(self):
        """Returns a list of all pieces on the board."""
        return [
            (r, c)
            for r in range(ROWS)
            for c in range(COLS)
            if self.board[r][c] is not None
        ]

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
                if piece and not piece.revealed:
                    # 可以翻开所有未翻开的棋子
                    possible_actions.append(("reveal", (r, c)))
                if piece and piece.player == player_id: # 相等说明已经翻开
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < ROWS and 0 <= nc < COLS:
                            piece_at_target = self.get_piece(nr, nc)
                            if piece_at_target and piece_at_target.player == player_id: # 如果相等，则说明已经翻开了(piece有保护机制)
                                # 不能移动到有自己已翻开棋子的格子
                                continue
                            possible_actions.append(("move", (r, c), (nr, nc)))
        return possible_actions


class Player:
    """统一的玩家基类"""

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.ai_type = "Unknown"

        # 训练相关属性
        self.training_stats = {
            "episodes": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "total_reward": 0.0,
            "average_reward": 0.0,
            "win_rate": 0.0,
        }

    def take_turn(self, board: Board) -> bool:
        """执行一个回合"""
        raise NotImplementedError

    def handle_event(self, event, board: Board) -> bool:
        """处理事件（主要用于人类玩家）"""
        return False

    def update_stats(self, result: int, reward: float):
        """更新训练统计，平局算0.5胜率"""
        self.training_stats["episodes"] += 1
        self.training_stats["total_reward"] += reward

        if result == self.player_id:
            self.training_stats["wins"] += 1
        elif result == 1 - self.player_id:
            self.training_stats["losses"] += 1
        else:  # result == 2 或其他平局值
            self.training_stats["draws"] += 1

        total_games = (
            self.training_stats["wins"]
            + self.training_stats["losses"]
            + self.training_stats["draws"]
        )

        if total_games > 0:
            # 平局算0.5分
            effective_wins = (
                self.training_stats["wins"] + 0.5 * self.training_stats["draws"]
            )
            self.training_stats["win_rate"] = effective_wins / total_games
        else:
            self.training_stats["win_rate"] = 0.0

        # 计算平均奖励
        self.training_stats["avg_reward"] = self.training_stats["total_reward"] / max(
            total_games, 1
        )

    def get_stats(self) -> Dict:
        """获取训练统计"""
        return self.training_stats.copy()

    def reset_stats(self):
        """重置统计"""
        for key in self.training_stats:
            if isinstance(self.training_stats[key], (int, float)):
                self.training_stats[key] = (
                    0.0 if "rate" in key or "average" in key else 0
                )


# 比较泛化的 Game 类
class Game:
    def __init__(self, agent=None, base_agent=None, display=True, delay=0.0):
        self.display = display
        if self.display:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("简化斗兽棋 - AI 对抗")
            self.font = pygame.font.SysFont(None, 36)
            self.small_font = pygame.font.SysFont(None, 24)

        self.board = Board()

        if agent is None or base_agent is None:
            raise ValueError("Need two agents input")
        self.players = {
            0: agent if isinstance(agent, Player) else agent(0),  # 红色AI
            1: base_agent if isinstance(base_agent, Player) else agent(1),  # 蓝色AI
        }
        self.current_player_id = 0
        self.running = True
        self.AI_DELAY_SECONDS = delay  # AI行动之间的延迟，以便观察

    def _get_player_type_name(self, player):
        """获取玩家类型的显示名称"""
        if hasattr(player, "ai_type"):
            return player.ai_type

        class_name = player.__class__.__name__
        type_map = {
            "HumanPlayer": "Human Player",
            "RandomPlayer": "Random AI",
            "QLearningAgent": "Q-Learning AI",
            "DQNAgent": "Deep Q-Network AI",
            "ApproximateQAgent": "Approximate Q AI",
            "MinimaxPlayer": "Minimax AI",
            "AlphaBetaPlayer": "Alpha-Beta AI",
            "MCTSAgent": "MCTS AI",
        }
        return type_map.get(class_name, f"{class_name} AI")

    def _draw_board(self):
        """Draws the game board and pieces."""
        if not self.display:
            return
        self.screen.fill(WHITE)
        for i in range(ROWS):
            for j in range(COLS):
                rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                # 交替颜色
                if (i + j) % 2 == 0:
                    pygame.draw.rect(self.screen, GREEN, rect)  # 明色格子
                else:
                    pygame.draw.rect(self.screen, YELLOW, rect)  # 暗色格子
                pygame.draw.rect(self.screen, BLACK, rect, 1)  # Cell border

                piece = self.board.get_piece(i, j)
                if piece:
                    if piece.revealed:
                        color = RED if piece.player == 0 else BLUE
                        pygame.draw.circle(
                            self.screen, color, rect.center, TILE_SIZE // 3
                        )
                        text_color = WHITE  # Always white text on colored circle for better contrast
                        text = self.font.render(str(piece.strength), True, text_color)
                        text_rect = text.get_rect(center=rect.center)
                        self.screen.blit(text, text_rect)
                    else:
                        # 在棋盘上绘制未揭示的棋子
                        pygame.draw.circle(
                            self.screen, LIGHT_GREY, rect.center, TILE_SIZE // 3
                        )  # 使用 rect.center 代替 (x, y)

        status_bar_rect = pygame.Rect(
            0, SCREEN_HEIGHT - STATUS_BAR_HEIGHT, SCREEN_WIDTH, STATUS_BAR_HEIGHT
        )
        pygame.draw.rect(
            self.screen, BLACK, status_bar_rect
        )  # Background for status bar

        current_player = self.players[self.current_player_id]
        player_name = "Red" if self.current_player_id == 0 else "Blue"
        text_color = RED if self.current_player_id == 0 else BLUE

        ai_type = self._get_player_type_name(current_player)
        status_text = f"Current Turn: {player_name} ({ai_type})"

        text_surface = self.small_font.render(status_text, True, text_color)
        self.screen.blit(text_surface, (10, SCREEN_HEIGHT - STATUS_BAR_HEIGHT + 5))

        red_type = self._get_player_type_name(self.players[0])
        blue_type = self._get_player_type_name(self.players[1])

        info_text = f"Red: {red_type} vs Blue: {blue_type}"
        info_surface = self.small_font.render(info_text, True, GREEN)
        self.screen.blit(info_surface, (10, SCREEN_HEIGHT - STATUS_BAR_HEIGHT + 20))

    def _check_game_over(self):
        """-1: playing, 0: 0 win, 1: 1 win, 2: draw"""

        if len(self.board._died[0]) == 8:
            red_type = self._get_player_type_name(self.players[0])
            blue_type = self._get_player_type_name(self.players[1])
            self._game_over(f"Blue ({blue_type}) wins against Red ({red_type})!")
            return 1
        elif len(self.board._died[1]) == 8:
            red_type = self._get_player_type_name(self.players[0])
            blue_type = self._get_player_type_name(self.players[1])
            self._game_over(f"Red ({red_type}) wins against Blue ({blue_type})!")
            return 0
        elif len(self.board._died[0]) == 7 and len(self.board._died[1]) == 7:
            piece_1_pos, piece_2_pos = self.board.get_all_pieces()
            piece_1 = self.board.get_piece(piece_1_pos[0], piece_1_pos[1])
            piece_2 = self.board.get_piece(piece_2_pos[0], piece_2_pos[1])
            # 只有当两个棋子都已翻开时，才能判断是否能互相捕获, strength里有判断，省略
            if compare_strength(piece_1.strength, piece_2.strength) == 0:
                red_type = self._get_player_type_name(self.players[0])
                blue_type = self._get_player_type_name(self.players[1])
                self._game_over(f"Draw! Red ({red_type}) vs Blue ({blue_type})")
                return 2
            # 如果有一方或双方未翻开，游戏继续 (因为信息不完全，未来可能仍有变化)

        return -1  # 游戏未结束

    def _game_over(self, message):
        """Displays game over message and exits."""
        print(message)
        self.running = False  # Stop the main game loop

    def run(self):
        """
        Main game loop.
        return -1: playing, 0: 0 win, 1: 1 win, 2: draw
        """
        if self.display:
            clock = pygame.time.Clock()

        step_count = 0
        result = -1  # 初始状态，游戏未结束
        last_action_time = time.time()  # 用于控制 AI 行动间隔
        while self.running:
            if self.display:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    # 移除了人类玩家事件处理逻辑

            current_player_obj = self.players[self.current_player_id]

            # AI 玩家的回合逻辑
            # AI 不依赖事件，直接在循环中执行
            if self.display:
                current_time = time.time()
                if current_time - last_action_time < self.AI_DELAY_SECONDS:
                    continue
                last_action_time = current_time  # 更新上次行动时间

            if current_player_obj.take_turn(self.board):
                # 只在AI成功执行动作后更新计数器

                # 在AI成功执行动作后更新计数器
                step_count += 1
                result = self._check_game_over()
                if result != -1:
                    self.running = False
                elif step_count >= MAX_STEPS:
                    self._game_over(f"draw, exceeded {MAX_STEPS} steps")
                    self.running = False
                    result = 2  # 平局
                elif not self.board.get_player_pieces(self.current_player_id) and not self.board.get_unveal_pieces():
                    player_name = "Red AI" if self.current_player_id == 0 else "Blue AI"
                    self._game_over(f"{player_name} no movements avaliabe")
                    self.running = False
                    result = 1 - self.current_player_id  # 对手胜利
                else:
                    self.current_player_id = 1 - self.current_player_id  # 切换回合
            else:
                # 如果AI没有找到任何合法动作（极少发生，通常意味着游戏结束了）
                # 可以在这里添加一个平局判断或者其他处理
                if self.display:
                    print(
                        f"Player {self.current_player_id} could not make a valid move. Game might be stuck or draw."
                    )
                self._game_over(
                    "Draw! (No valid moves left for current player or stuck state)"
                )
                self.running = False  # 结束游戏
                result = 2

            if self.display:
                self._draw_board()
                pygame.display.flip()
                # pygame.time.wait(5000)  # 暂停5秒
                clock.tick(30)  # 控制帧率

        if self.display:
            pygame.quit()

        return result  # 返回游戏结果 (-1: playing, 0: 0 win, 1: 1 win, 2: draw)


class GameEnvironment:
    """统一的游戏环境类，支持自定义奖励函数, 并非管理员，只是Agent能看见的环境"""

    def __init__(self, reward_function=None):
        self.board = None
        self.current_player = 0
        self.reward_function = reward_function
        self.step_count = 0
        self.game_history = []  # 记录游戏历史
        self.reset()

    def reset(self) -> np.ndarray:
        """重置环境，返回初始状态"""
        self.board = Board()
        self.current_player = 0
        self.step_count = 0
        self.game_history = []
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """获取当前状态的数值表示"""
        state = np.zeros((ROWS, COLS, 4))

        for r in range(ROWS):
            for c in range(COLS):
                piece = self.board.get_piece(r, c)
                if piece:
                    state[r, c, 0] = piece.player
                    state[r, c, 1] = piece.strength / 8.0
                    state[r, c, 2] = 1 if piece.revealed else 0
                    state[r, c, 3] = 1

        flat_state = state.flatten()
        current_player_feature = np.array([self.current_player])
        return np.concatenate([flat_state, current_player_feature])

    def get_valid_actions(self, player_id: int) -> List[Tuple]:
        """获取指定玩家的所有有效动作"""
        return self.board.get_all_possible_moves(player_id)

    def step(self, action: Tuple) -> Tuple[np.ndarray, float, int, Dict]:
        """
        执行动作，返回 (下一状态, 奖励, 游戏结果, 信息)
        游戏结果: -1=继续, 0=玩家0胜, 1=玩家1胜, 2=平局
        """
        action_type, pos1, pos2 = action
        board_before = copy.deepcopy(self.board) if self.reward_function else None

        # 执行动作
        success = False
        if action_type == "reveal":
            r, c = pos1
            piece = self.board.get_piece(r, c)
            if piece and not piece.revealed:
                piece.reveal()
                success = True
        elif action_type == "move":
            success = self.board.try_move(pos1, pos2)

        if not success:
            reward = -1.0 if self.reward_function else 0.0
            print("invalid action", action)
            return self.get_state(), reward, -1, {"invalid": True}

        self.step_count += 1

        # 检查游戏结果
        result = self._check_game_result()

        # 计算奖励
        if self.reward_function:
            reward = self.reward_function.calculate_reward(
                board_before, self.board, action, self.current_player, result != -1
            )
        else:
            # 默认奖励：胜利+1，失败-1，其他0
            if result == self.current_player:
                reward = 1.0
            elif result != -1 and result != 2:
                reward = -1.0
            else:
                reward = 0.0

        # 记录历史
        self.game_history.append(
            {
                "action": action,
                "player": self.current_player,
                "reward": reward,
                "state_before": board_before,
                "state_after": copy.deepcopy(self.board),
            }
        )

        # 切换玩家
        self.current_player = 1 - self.current_player

        return self.get_state(), reward, result, {}

    def _check_game_result(self) -> int:
        """检查游戏结果"""
        if len(self.board._died[0]) == 8:
            return 1
        elif len(self.board._died[1]) == 8:
            return 0
        elif len(self.board._died[0]) == 7 and len(self.board._died[1]) == 7:
            piece_1_pos, piece_2_pos = self.board.get_all_pieces()
            piece_1 = self.board.get_piece(piece_1_pos[0], piece_1_pos[1])
            piece_2 = self.board.get_piece(piece_2_pos[0], piece_2_pos[1])
            # 只有当两个棋子都已翻开时，才能判断是否能互相捕获, strength里有判断，省略
            if compare_strength(piece_1.strength, piece_2.strength) == 0:
                return 2
            # 如果有一方或双方未翻开，游戏继续 (因为信息不完全，未来可能仍有变化)
        return -1  # 游戏未结束


class BaseTrainer:
    """统一的训练器基类 - 重构为批次统计"""

    def __init__(
        self,
        agent: Player,
        opponent: Player,
        reward_function=None,
        save_path: str = "model_data/",
    ):
        self.agent = agent
        self.opponent = opponent
        self.env = GameEnvironment(reward_function)
        self.save_path = save_path

        # 训练统计 - 批次级别
        self.training_history = {
            "episodes": [],
            "rewards": [],
            "wins": [],
            "losses": [],
            "draws": [],
            "win_rates": [],
            "average_rewards": [],
            "learning_rates": [],
            "epsilons": [],
            "discount_factors": [],
        }

        # 超参数更新控制
        # self.lr_update_frequency = {
        #     'phase1': 100,
        #     'phase2': 150,
        #     'phase3': 200
        # }

        # 批次统计 - 重置为批次级别
        self.batch_wins = 0
        self.batch_losses = 0
        self.batch_draws = 0
        self.batch_episodes = 0
        self.batch_rewards = []

        # 总体统计（仅用于全局记录）
        self.total_episodes = 0
        self.total_wins = 0
        self.total_losses = 0
        self.total_draws = 0

        # 当前训练阶段信息
        self.current_phase = "phase1"
        self.phase_episode_offset = 0  # 当前阶段的起始episode

        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)

    def reset_batch_stats(self):
        """重置批次统计"""
        self.batch_wins = 0
        self.batch_losses = 0
        self.batch_draws = 0
        self.batch_episodes = 0
        self.batch_rewards = []

    def set_phase(self, phase_name: str, episode_offset: int = 0):
        """设置当前训练阶段"""
        self.current_phase = phase_name
        self.phase_episode_offset = episode_offset
        self.reset_batch_stats()

    def get_batch_win_rate(self) -> float:
        """获取当前批次胜率"""
        if self.batch_episodes == 0:
            return 0.0
        effective_wins = self.batch_wins + 0.5 * self.batch_draws
        return effective_wins / self.batch_episodes

    def get_batch_avg_reward(self) -> float:
        """获取当前批次平均奖励"""
        if not self.batch_rewards:
            return 0.0
        return np.mean(self.batch_rewards)

    def train_episode(self, opponent=None, **kwargs) -> Tuple[float, int, int]:
        """训练一个回合，在episode结束时处理epsilon衰减"""
        if opponent:
            self.opponent = opponent

        board = Board()
        total_reward = 0
        steps = 0
        current_player = 0
        result = -1
        max_steps = 1000

        while True:
            if current_player == self.agent.player_id:
                # 智能体回合
                valid_actions = board.get_all_possible_moves(self.agent.player_id)
                if not valid_actions:
                    result = 1 - self.agent.player_id
                    break

                board_before = copy.deepcopy(board)
                action = self._agent_choose_action(board, valid_actions)

                # 执行动作
                success = self._execute_action(board, action, self.agent.player_id)
                if not success:
                    result = 1 - self.agent.player_id
                    break

                board_after = copy.deepcopy(board)

                # 检查游戏结果
                result = self._check_game_result(board)

                # 计算奖励
                if self.env.reward_function:
                    reward = self.env.reward_function.calculate_reward(
                        board_before, board_after, action, self.agent.player_id, result
                    )
                else:
                    reward = self._default_reward(result, self.agent.player_id)

                # 更新智能体
                self._agent_update(board_before, action, reward, board_after, result)

                total_reward += reward
                steps += 1

                if steps >= max_steps:
                    result = 2
                    break

                if result != -1:
                    break

                current_player = 1 - current_player

            else:
                # 对手回合
                if self.opponent.take_turn(board):
                    result = self._check_game_result(board)
                    steps += 1

                    if steps >= max_steps:
                        result = 2
                        break

                    if result != -1:
                        break
                    current_player = 1 - current_player
                else:
                    result = self.agent.player_id
                    break

        # 统一处理episode结束后的工作
        self._handle_episode_end(result, total_reward)

        return total_reward, steps, result

    def _handle_episode_end(self, result: int, total_reward: float):
        """统一处理episode结束后的工作 - 使用智能体的统一超参数控制"""
        # 更新批次统计
        self.batch_episodes += 1
        self.batch_rewards.append(total_reward)

        if result == self.agent.player_id:
            self.batch_wins += 1
        elif result == 1 - self.agent.player_id:
            self.batch_losses += 1
        else:
            self.batch_draws += 1

        # 更新总体统计
        self.total_episodes += 1
        if result == self.agent.player_id:
            self.total_wins += 1
        elif result == 1 - self.agent.player_id:
            self.total_losses += 1
        else:
            self.total_draws += 1

        # 更新智能体统计
        self.agent.update_stats(result, total_reward)

        # 添加结果追踪
        if hasattr(self.agent, '_recent_results'):
            self.agent._recent_results.append(result)
        else:
            self.agent._recent_results = deque(maxlen=200)
            self.agent._recent_results.append(result)
        
        batch_win_rate = self.get_batch_win_rate()
        current_episode_in_phase = self.batch_episodes - 1  # 从0开始

        # 使用智能体的统一分阶段控制方法
        if hasattr(self.agent, "decay_epsilon_by_phase") and hasattr(
            self.agent, "update_learning_rate_by_phase"
        ):
            # 新的统一控制方法
            self.agent.decay_epsilon_by_phase(
                self.current_phase, current_episode_in_phase, batch_win_rate
            )
            self.agent.update_learning_rate_by_phase(
                self.current_phase, current_episode_in_phase, batch_win_rate
            )
        else:
            # 后备方法
            if hasattr(self.agent, "decay_epsilon_by_phase"):
                self.agent.decay_epsilon_by_phase(
                    self.current_phase, current_episode_in_phase, batch_win_rate
                )
            elif hasattr(self.agent, "decay_epsilon"):
                self.agent.decay_epsilon(batch_win_rate)

            # 学习率更新使用传统方法
            if hasattr(self.agent, "update_learning_rate"):
                # 只在特定频率更新（后备机制）
                if current_episode_in_phase > 0 and current_episode_in_phase % 100 == 0:
                    self.agent.update_learning_rate(batch_win_rate)

        # 更新折扣因子 (AQ专有)
        if hasattr(self.agent, "update_discount_factor"):
            self.agent.update_discount_factor(batch_win_rate)

        # 更新训练历史
        self._update_training_history(total_reward, result)

    def _execute_action(self, board: Board, action: Tuple, player_id: int) -> bool:
        """执行动作"""
        try:
            if len(action) == 2:
                # 处理只有两个元素的动作 (action_type, pos)
                action_type, pos1 = action
                pos2 = None
            elif len(action) == 3:
                # 处理正常的三个元素的动作
                action_type, pos1, pos2 = action
            else:
                print(f"警告: 动作格式不正确: {action}")
                return False

            if action_type == "reveal":
                r, c = pos1
                piece = board.get_piece(r, c)
                if piece and not piece.revealed:
                    piece.reveal()
                    return True
            elif action_type == "move":
                if pos2 is not None:
                    return board.try_move(pos1, pos2)
                else:
                    print(f"警告: 移动动作缺少目标位置: {action}")
                    return False

            return False

        except Exception as e:
            print(f"执行动作时出错: {e}, 动作: {action}")
            return False

    def _check_game_result(self, board: Board) -> int:
        """检查游戏结果"""
        if len(board._died[0]) == 8:
            return 1
        elif len(board._died[1]) == 8:
            return 0
        elif len(board._died[0]) == 7 and len(board._died[1]) == 7:
            piece_1_pos, piece_2_pos = board.get_all_pieces()
            piece_1 = board.get_piece(piece_1_pos[0], piece_1_pos[1])
            piece_2 = board.get_piece(piece_2_pos[0], piece_2_pos[1])
            # 只有当两个棋子都已翻开时，才能判断是否能互相捕获, strength里有判断，省略
            if compare_strength(piece_1.strength, piece_2.strength) == 0:
                return 2
            # 如果有一方或双方未翻开，游戏继续 (因为信息不完全，未来可能仍有变化)

        return -1  # 游戏未结束

    def _default_reward(self, result: int, player_id: int) -> float:
        """默认奖励函数"""
        if result == player_id:
            return 1.0
        elif result != -1 and result != 2:
            return -1.0
        else:
            return 0.0

    def _agent_choose_action(self, board: Board, valid_actions: List[Tuple]) -> Tuple:
        """智能体选择动作（需要子类实现）"""
        raise NotImplementedError

    def _agent_update(
        self,
        board_before: Board,
        action: Tuple,
        reward: float,
        board_after: Board,
        result: int,
    ):
        """更新智能体（需要子类实现）"""
        pass

    def _update_training_history(self, reward: float, result: int):
        """更新训练历史 - 包含学习率等超参数"""
        stats = self.agent.get_stats()

        self.training_history["episodes"].append(stats["episodes"])
        self.training_history["rewards"].append(reward)
        self.training_history["wins"].append(1 if result == self.agent.player_id else 0)
        self.training_history["losses"].append(
            1 if result == 1 - self.agent.player_id else 0
        )
        self.training_history["draws"].append(1 if result == 2 else 0)
        self.training_history["win_rates"].append(stats["win_rate"])
        self.training_history["average_rewards"].append(stats["avg_reward"])

        # 记录超参数变化
        # 学习率
        if hasattr(self.agent, "get_learning_rate"):
            lr = self.agent.get_learning_rate()
        elif hasattr(self.agent, "learning_rate"):
            lr = self.agent.learning_rate
        else:
            lr = None
        self.training_history["learning_rates"].append(lr)

        # Epsilon
        if hasattr(self.agent, "epsilon"):
            epsilon = self.agent.epsilon
        else:
            epsilon = None
        self.training_history["epsilons"].append(epsilon)

        # 折扣因子
        if hasattr(self.agent, "discount_factor"):
            discount = self.agent.discount_factor
        elif hasattr(self.agent, "gamma"):
            discount = self.agent.gamma
        else:
            discount = None
        self.training_history["discount_factors"].append(discount)

    def train(
        self,
        episodes: int = 1000,
        save_interval: int = 100,
        print_interval: int = 10,
        **kwargs,
    ):
        """训练主循环 - 增强版进度显示"""
        print(f"开始训练 {episodes} 回合...")

        for episode in range(episodes):
            total_reward, steps, result = self.train_episode(**kwargs)

            # 打印进度
            if episode % print_interval == 0 or episode == episodes - 1:
                # 计算最近的胜率
                recent_window = min(20, len(self.win_history))
                recent_win_rate = (
                    np.mean(self.win_history[-recent_window:])
                    if self.win_history
                    else 0.0
                )

                # 获取当前超参数信息
                param_info = self._get_current_params_info()

                print(
                    f"回合 {episode}: 奖励 = {total_reward:.2f}, 步数 = {steps}, "
                    f"胜率 = {recent_win_rate:.3f}{param_info}"
                )

            # 定期保存
            if save_interval > 0 and episode % save_interval == 0 and episode > 0:
                self.save_model(f"{self.agent.__class__.__name__}_episode_{episode}")

        # 最终保存
        self.save_model(f"final_{self.agent.__class__.__name__}")
        print(f"模型已保存到 {self.save_path}")
        print("训练完成！")

        return {
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_history[-1] if self.win_history else 0.0,
            "win_history": self.win_history,
            "training_history": self.training_history,
        }

    def _get_current_params_info(self) -> str:
        """获取当前超参数信息字符串"""
        info_parts = []

        # Epsilon
        if hasattr(self.agent, "epsilon"):
            info_parts.append(f"ε = {self.agent.epsilon:.3f}")

        # 学习率
        if hasattr(self.agent, "get_learning_rate"):
            lr = self.agent.get_learning_rate()
            if lr is not None:
                info_parts.append(f"lr = {lr:.4f}")
        elif hasattr(self.agent, "learning_rate"):
            info_parts.append(f"lr = {self.agent.learning_rate:.4f}")

        # 折扣因子
        if hasattr(self.agent, "discount_factor"):
            info_parts.append(f"γ = {self.agent.discount_factor:.3f}")
        elif hasattr(self.agent, "gamma"):
            info_parts.append(f"γ = {self.agent.gamma:.3f}")

        return ", " + ", ".join(info_parts) if info_parts else ""

    def save_model(self, filename: str):
        """保存模型（需要子类实现）"""
        pass

    def get_training_history(self) -> Dict:
        """获取训练历史"""
        return self.training_history.copy()
