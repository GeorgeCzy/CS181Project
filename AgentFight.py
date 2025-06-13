import pygame
import random
import sys
import time
import copy  # 导入 copy 模块用于深拷贝
from MCTS import MCTSAgent
from base import (
    Player,
    Board,
    Piece,
    compare_strength,
    ROWS,
    COLS,
    TILE_SIZE,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    STATUS_BAR_HEIGHT,
    WHITE,
    BLACK,
    RED,
    BLUE,
    GREY,
    YELLOW,
    LIGHT_GREY,
    DARK_GREY,
    GREEN,
)


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
                return False  # Event not handled (clicked outside board)

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
                            return True  # Move successful, turn ends
                        else:
                            # 移动失败（例如，试图移动到自己已翻开的棋子），取消选择
                            self.selected_piece_pos = None
                            # 尝试选择新点击的棋子(第一次点击)
                            if piece:
                                self.selected_piece_pos = (row, col)
                    else:
                        # 目标位置不相邻，取消选择当前棋子，并尝试选择新棋子
                        if (
                            self.selected_piece_pos == piece_moving
                            and piece
                            and not piece.revealed
                        ):  # (第二次点击) 如果两次点的都是这个且没有翻开，相当于确认。human 这个我忘测试了，你可以跑跑看看
                            piece.reveal()
                            self.selected_piece_pos = None
                            return True  # Revealed piece, turn ends
                        else:
                            self.selected_piece_pos = (row, col)  # Select new piece
            elif piece:
                # No piece selected, try to select one
                if not piece.revealed:
                    piece.reveal()
                    return True  # Revealed piece, turn ends
                self.selected_piece_pos = (row, col)  # Select piece
        return False  # Event not handled or turn not ended

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
                if piece and not piece.revealed:
                    piece.reveal()
                    return True  # Turn ended
            elif action_type == "move":
                start_pos, end_pos = action[1], action[2]
                if board.try_move(
                    start_pos, end_pos
                ):  # 我没有加很多判断，我相信给出的 possible actions 是正确的
                    return True  # Turn ended
        return False  # No valid moves found


class GreedyPlayer(Player):
    """An AI player that uses heuristic evaluation to make greedy moves."""

    def __init__(self, player_id, print_messages=True):
        super().__init__(player_id)
        self.opponent_id = 1 - player_id
        self.print_messages = print_messages

    def _evaluate_board(self, board):
        """
        Evaluates the current board state using only revealed information.
        Score(s) = Σ(my revealed pieces) - λ₁Σ(opp revealed pieces) + λ₂·Mobility + λ₃·Advancing
        """
        score = 0
        piece_values = {8: 100, 7: 90, 6: 80, 5: 70, 4: 60, 3: 50, 2: 40, 1: 10}

        # 1. Base Score - only revealed pieces
        λ_1 = 0.8
        for r in range(ROWS):
            for c in range(COLS):
                piece = board.get_piece(r, c)
                if piece and piece.revealed:
                    if piece.player == self.player_id:
                        score += piece_values[piece.strength]
                    else:
                        score -= piece_values[piece.strength] * λ_1

        # 2. Positional Rewards - only revealed pieces
        λ_3 = 15
        for r in range(ROWS):
            for c in range(COLS):
                piece = board.get_piece(r, c)
                if piece and piece.revealed and piece.player == self.player_id:
                    if self.player_id == 0:  # Red player
                        advance_score = r * 5
                    else:  # Blue player
                        advance_score = (ROWS - 1 - r) * 5
                    score += advance_score * λ_3 / 10

                    # Central control bonus
                    center_cols = [COLS // 2 - 1, COLS // 2]
                    if c in center_cols:
                        score += 5

        # 3. Mobility - count available moves (includes valid reveal actions)
        λ_2 = 2
        possible_actions = board.get_all_possible_moves(self.player_id)
        score += len(possible_actions) * λ_2

        # 4. Safety Penalty - only between revealed pieces
        safety_weight = 10
        for r in range(ROWS):
            for c in range(COLS):
                piece = board.get_piece(r, c)
                # Only consider revealed friendly pieces for threats
                if not piece or not piece.revealed or piece.player != self.player_id:
                    continue

                # Check adjacent positions
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < ROWS and 0 <= nc < COLS):
                        continue

                    adj_piece = board.get_piece(nr, nc)
                    # Only consider revealed enemy pieces
                    if (
                        not adj_piece
                        or not adj_piece.revealed
                        or adj_piece.player == self.player_id
                    ):
                        continue

                    # Check threat direction
                    compare = compare_strength(piece.strength, adj_piece.strength)
                    if compare == -1:
                        score -= safety_weight
                    # Check if we threaten opponent
                    elif compare == 1:
                        score += safety_weight * 1.5

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
            return 10
        elif action_type == "move":
            start_pos, end_pos = action[1], action[2]
            action_successful = temp_board.try_move(start_pos, end_pos)

        if not action_successful:
            return float("-inf")  # Invalid action

        # Get new board score after action
        new_score = self._evaluate_board(temp_board)

        return new_score - current_score

    def take_turn(self, board):
        """
        Chooses the best action based on heuristic evaluation.
        This is a greedy approach - evaluates all possible moves and picks the best one.
        """
        if self.print_messages:
            print(f"Greedy Player {self.player_id} thinking...")
        start_time = time.time()

        # Get all possible actions
        possible_actions = board.get_all_possible_moves(self.player_id)

        if not possible_actions:
            return False  # No valid moves

        # Evaluate each action and find the best one
        best_action = None
        best_score = float("-inf")

        for action in possible_actions:
            action_score = self._evaluate_action(board, action)

            # Add small random factor to break ties
            action_score += random.uniform(-0.1, 0.1)

            if action_score > best_score:
                best_score = action_score
                best_action = action

        end_time = time.time()
        if self.print_messages:
            print(
                f"Greedy found best action: {best_action} with score improvement {best_score:.2f} in {end_time - start_time:.3f} seconds"
            )

        # Execute the best action
        if best_action:
            action_type = best_action[0]
            if action_type == "reveal":
                r, c = best_action[1]
                piece = board.get_piece(r, c)
                if piece and not piece.revealed:
                    piece.reveal()
                    return True
            elif action_type == "move":
                start_pos, end_pos = best_action[1], best_action[2]
                return board.try_move(start_pos, end_pos)
        return False


class MinimaxPlayer(Player):
    def __init__(
        self, player_id, max_depth=3, print_messages=True
    ):  # Added max_depth for search
        super().__init__(player_id)
        self.max_depth = max_depth
        self.opponent_id = 1 - player_id
        self.print_messages = print_messages

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
        # if not opponent_pieces: # Opponent has no pieces left
        #     return float('inf') # Win state is highly favorable
        # if not self_pieces: # Self has no pieces left
        #     return float('-inf') # Loss state is highly unfavorable
        if (
            len(board._died[self.player_id]) == 8
            and len(board._died[self.opponent_id]) == 8
        ):
            return 0
        if len(board._died[self.player_id]) == 8:
            return float("-inf")
        if len(board._died[self.opponent_id]) == 8:
            return float("inf")

        score += (
            len(board._died[self.opponent_id]) - len(board._died[self.player_id])
        ) * 100  # Each piece is worth 100 points

        # 2. Strength sum difference (prioritize stronger pieces)
        # 只计算已翻开棋子的强度
        self_strength_sum = sum(
            board.get_piece(r, c).strength
            for r, c in self_pieces
            if board.get_piece(r, c).revealed
        )
        opponent_strength_sum = sum(
            board.get_piece(r, c).strength
            for r, c in opponent_pieces
            if board.get_piece(r, c).revealed
        )
        score += (
            self_strength_sum - opponent_strength_sum
        ) * 10  # Each strength point worth 10

        # 3. Revealed vs Unrevealed pieces
        # Incentivize revealing own pieces
        for r, c in self_pieces:
            if not board.get_piece(r, c).revealed:
                score += 5  # Small bonus for having unrevealed pieces to reveal (potential future power)

        # Penalize opponent for having unrevealed pieces (unknown threat)
        for r, c in opponent_pieces:
            if not board.get_piece(r, c).revealed:
                score -= 10  # Small penalty for opponent having hidden pieces

        # 4. Proximity to opponent's pieces for revealed pieces (threats/opportunities)
        for sr, sc in self_pieces:
            self_piece = board.get_piece(sr, sc)
            if self_piece and self_piece.revealed:
                for er, ec in opponent_pieces:
                    opponent_piece = board.get_piece(er, ec)
                    if opponent_piece and opponent_piece.revealed:
                        if board.is_adjacent((sr, sc), (er, ec)):
                            compare = compare_strength(
                                self_piece.strength, opponent_piece.strength
                            )
                            if compare == 1:
                                score += 20  # Strong capture opportunity
                            elif compare == -1:
                                score -= 15  # Danger of being captured

        return score

    def _minimax(self, board, depth, maximizing_player_id, alpha=float("-inf"), beta=float("inf")):
        """带 Alpha-Beta 剪枝的 Minimax 搜索"""
        # Base case: max_depth reached or game over
        # 优化：在每次评估时，判断游戏是否结束，避免不必要的递归
        self_pieces_count = 8 - len(board._died[self.player_id])
        opponent_pieces_count = 8 - len(board._died[self.opponent_id])

        if depth == 0 or self_pieces_count == 0 or opponent_pieces_count == 0:
            return self._evaluate(board), None  # Return score and no move

        current_player = maximizing_player_id
        is_maximizing_player = current_player == self.player_id
        best_move = None

        # 如果是最大化玩家（AI自己）
        if is_maximizing_player:
            max_eval = float("-inf")
            # 获取当前玩家所有可能的动作
            possible_actions = board.get_all_possible_moves(current_player)
            random.shuffle(
                possible_actions
            )  # 打乱顺序，增加AI行为多样性（当多个动作分数相同）

            for action in possible_actions:
                # 对棋盘进行深拷贝以模拟动作
                temp_board = copy.deepcopy(board)
                action_type = action[0]

                if action_type == "reveal":
                    r, c = action[1]
                    # 不需要验证动作正确性，正确性在board里验证过了
                    # 假设翻开的棋子可能目前未翻开的任意棋子，进行单层expectmax评估
                    eval = 0
                    unveal_pieces = board.get_unveal_pieces()
                    if not unveal_pieces:  # 防止除零错误
                        continue
                    
                    for unveal_pos in unveal_pieces:
                        # 注意，这里是board，使得unveal piece保持不变
                        # 轮流将每一个未翻开的棋子翻开并置于reveal的位置
                        # 实现方法是将该未翻开棋子与原本要翻开位置的棋子交换位置
                        temp_board = copy.deepcopy(board)  # 每一次temp board重置
                        unveal_piece = temp_board.get_piece(
                            unveal_pos[0], unveal_pos[1]
                        )  # 这个是实际我们翻开的棋子
                        current_piece = temp_board.get_piece(
                            r, c
                        )  # 这个是本来要翻开位置的棋子
                        temp_board.set_piece(
                            r, c, unveal_piece
                        )  # 将未翻开的棋子放在要翻开的棋子位置
                        temp_board.set_piece(
                            unveal_pos[0], unveal_pos[1], current_piece
                        )  # 将原本要翻开的棋子放在未翻开的棋子位置
                        unveal_piece.reveal()  # 翻开棋子
                        eval += self._minimax(temp_board, 0, 1 - current_player)[
                            0
                        ]  # 递归调用 Minimax, 只评估一层
                    eval = eval / len(board.get_unveal_pieces())  # 取平均值作为评估
                    
                elif action_type == "move":
                    start_pos, end_pos = action[1], action[2]
                    # 在临时棋盘上尝试移动
                    action_performed = temp_board.try_move(start_pos, end_pos)
                    if not action_performed:
                        raise ValueError(
                            "Invalid move attempted in Minimax evaluation", action
                        )
                    # 递归调用 Minimax，切换到对手玩家
                    eval = self._minimax(temp_board, depth - 1, 1 - current_player)[0]
                if eval > max_eval:
                    max_eval = eval
                    best_move = action
                    
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta剪枝
                
            return max_eval, best_move
        # 如果是最小化玩家（对手）
        else:
            min_eval = float("inf")
            # 获取对手所有可能的动作
            possible_actions = board.get_all_possible_moves(current_player)
            random.shuffle(possible_actions)  # 打乱顺序

            for action in possible_actions:
                temp_board = copy.deepcopy(board)
                action_type = action[0]

                if action_type == "reveal":
                    r, c = action[1]
                    eval = 0
                    unveal_pieces = board.get_unveal_pieces()
                    if not unveal_pieces:  # 防止除零错误
                        continue
                    for unveal_pos in unveal_pieces:
                        # 轮流将每一个未翻开的棋子翻开并置于reveal的位置
                        temp_board = copy.deepcopy(board)
                        unveal_piece = temp_board.get_piece(
                            unveal_pos[0], unveal_pos[1]
                        )  # 这个是实际我们翻开的棋子
                        current_piece = temp_board.get_piece(
                            r, c
                        )  # 这个是本来要翻开位置的棋子
                        temp_board.set_piece(
                            r, c, unveal_piece
                        )  # 将未翻开的棋子放在翻开的棋子位置
                        temp_board.set_piece(
                            unveal_pos[0], unveal_pos[1], current_piece
                        )  # 将原本要翻开的棋子放在未翻开的棋子位置
                        unveal_piece.reveal()  # 翻开棋子
                        eval += self._minimax(temp_board, 0, 1 - current_player)[0]
                    eval = eval / len(board.get_unveal_pieces())  # 取平均值作为评估
                elif action_type == "move":
                    start_pos, end_pos = action[1], action[2]
                    action_performed = temp_board.try_move(start_pos, end_pos)
                    if not action_performed:
                        raise ValueError(
                            "Invalid move attempted in Minimax evaluation", action
                        )
                    # 递归调用 Minimax，切换回 AI 玩家
                    eval = self._minimax(temp_board, depth - 1, 1 - current_player)[0]
                if eval < min_eval:
                    min_eval = eval
                    best_move = action
                    
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha剪枝
                
            return min_eval, best_move

    def take_turn(self, board):
        if self.print_messages:
            print(f"Minimax Player {self.player_id} thinking...")
        start_time = time.time()

        # 运行 Minimax 搜索
        value, best_action = self._minimax(board, self.max_depth, self.player_id)

        end_time = time.time()
        if self.print_messages:
            print(
                f"Minimax found best action: {best_action} with value {value} in {end_time - start_time:.2f} seconds"
            )

        if best_action:
            action_type = best_action[0]
            if action_type == "reveal":
                r, c = best_action[1]
                piece = board.get_piece(r, c)
                if piece and not piece.revealed:
                    piece.reveal()
                    return True
            elif action_type == "move":
                start_pos, end_pos = best_action[1], best_action[2]
                return board.try_move(start_pos, end_pos)
        return False  # 如果没有找到最佳动作或动作失败，不应该发生


MAX_STEPS = 1000  # 最大允许步数


class Game:

    def __init__(self, agent=None, base_agent=None, display=True, delay=0.0):
        self.display = display
        if self.display:
            pygame.init()
            pygame.font.init()  # Ensure font module is initialized
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("简化斗兽棋 - AI 对抗")
            self.font = pygame.font.SysFont(None, 36)
            self.small_font = pygame.font.SysFont(None, 24)

        self.board = Board()
        if agent is None:
            self.players = {
                # 修改这里，将人类玩家替换为AI玩家
                # 0: MinimaxPlayer(0, max_depth=1, print_messages=False),  # 红色AI
                # 0: QLearningPlayer(0), # 红色AI
                0: RandomPlayer(0),  # 红色AI
                1: GreedyPlayer(1, print_messages=False),  # 蓝色AI
                # 或者可以是一个Minimax vs Random
                # 0: MinimaxPlayer(0, max_depth=3),
                # 1: RandomPlayer(1)
            }
        else:
            base_agent = base_agent if base_agent else RandomPlayer
            self.players = {
                0: agent if isinstance(agent, Player) else agent(0),  # 红色AI
                1: base_agent if isinstance(base_agent, Player) else agent(1),  # 蓝色AI
            }
        self.current_player_id = 0
        self.running = True
        self.AI_DELAY_SECONDS = delay  # AI行动之间的延迟，以便观察

    def _draw_board(self):
        """Draws the game board and pieces."""
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
                        # pygame.draw.rect(self.screen, GREY, rect.inflate(-10, -10)) # Unrevealed piece block

        # Highlight selected piece for human player (不再需要，但保留以防将来修改为人类玩家)
        # human_player = self.players[0]
        # if isinstance(human_player, HumanPlayer) and human_player.get_selected_pos():
        #     i, j = human_player.get_selected_pos()
        #     rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        #     pygame.draw.rect(self.screen, YELLOW, rect, 3)

        # Draw status bar
        status_bar_rect = pygame.Rect(
            0, SCREEN_HEIGHT - STATUS_BAR_HEIGHT, SCREEN_WIDTH, STATUS_BAR_HEIGHT
        )
        pygame.draw.rect(
            self.screen, BLACK, status_bar_rect
        )  # Background for status bar

        player_name = "Red AI" if self.current_player_id == 0 else "Blue AI"
        text_color = RED if self.current_player_id == 0 else BLUE
        status_text = f"Current Turn: {player_name} (AI thinking...)"

        text_surface = self.font.render(status_text, True, text_color)
        self.screen.blit(text_surface, (10, SCREEN_HEIGHT - STATUS_BAR_HEIGHT + 5))

    def _check_game_over(self):
        """Checks for win/loss conditions and terminates the game if met."""
        # red_pieces = self.board.get_player_pieces(0)
        # blue_pieces = self.board.get_player_pieces(1)

        if len(self.board._died[0]) == 8:
            self._game_over("Blue AI wins!")
            return True
        elif len(self.board._died[1]) == 8:
            self._game_over("Red AI wins!")
            return True
        elif len(self.board._died[0]) == 7 and len(self.board._died[1]) == 7:
            piece_1, piece_2 = self.board.get_all_pieces()
            # 只有当两个棋子都已翻开时，才能判断是否能互相捕获, strength里有判断，省略
            if compare_strength(piece_1.strength, piece_2.strength) == 0:
                self._game_over("Draw! (Last pieces are equal)")
                return True
            # 如果有一方或双方未翻开，游戏继续 (因为信息不完全，未来可能仍有变化)
        return False  # 游戏未结束

    def _game_over(self, message):
        """Displays game over message and exits."""
        print(message)
        self.running = False  # Stop the main game loop
        # Optional: Keep the window open for a few seconds before closing

    def run(self):
        """Main game loop."""
        if self.display:
            clock = pygame.time.Clock()

        step_count = 0
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
                time.sleep(self.AI_DELAY_SECONDS)  # 增加延迟，方便观察

            if current_player_obj.take_turn(self.board):
                # 只在AI成功执行动作后更新计数器

                # 在AI成功执行动作后更新计数器
                step_count += 1
                if self._check_game_over():
                    self.running = False
                elif step_count >= MAX_STEPS:
                    self._game_over(f"draw, exceeded {MAX_STEPS} steps")
                    self.running = False
                # elif not self.board.get_player_pieces(self.current_player_id): # _check_game_over 里检查过了
                #     player_name = "Red AI" if self.current_player_id == 0 else "Blue AI"
                #     self._game_over(f"{player_name} no movements avaliabe")
                #     self.running = False
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

            if self.display:
                self._draw_board()
                pygame.display.flip()
                # pygame.time.wait(5000) # 暂停5秒
                clock.tick(30)  # 控制帧率

        if self.display:
            pygame.quit()
            sys.exit()


# --- Run the Game ---
if __name__ == "__main__":
    game = Game(delay=2)
    game.run()
