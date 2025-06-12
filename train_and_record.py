import numpy as np
import time
import os
import argparse
import copy
from datetime import datetime
from typing import Dict, Any, Callable, Type, List, Tuple
from base import Player, BaseTrainer, Board
from training_data_manager import TrainingDataManager


class MixedOpponent(Player):
    """混合对手：随机选择不同策略的AI"""

    def __init__(self, opponents: List[Tuple[Player, float]], player_id: int):
        super().__init__(player_id)
        self.opponents = opponents
        self.current_opponent = None
        self.ai_type = "Mixed AI (Random+Greedy+Minimax)"
        self._select_opponent()

    def _select_opponent(self):
        """根据概率随机选择一个对手"""
        if not self.opponents:
            return

        opponents, probs = zip(*self.opponents)
        self.current_opponent = np.random.choice(opponents, p=probs)

    def take_turn(self, board: Board) -> bool:
        """委托给当前选择的对手"""
        if self.current_opponent is None:
            self._select_opponent()

        return self.current_opponent.take_turn(board)


def create_opponent(opponent_type: str, player_id: int = 1):
    """
    创建不同类型的对手

    Args:
        opponent_type: 对手类型 ('random', 'greedy', 'minimax', 'mixed')
        player_id: 玩家ID（默认为1）
    """
    if opponent_type == "random":
        from AgentFight import RandomPlayer

        return RandomPlayer(player_id)

    elif opponent_type == "greedy":
        from AgentFight import GreedyPlayer

        return GreedyPlayer(player_id, print_messages=False)  # 关闭打印信息

    elif opponent_type == "minimax":
        from AgentFight import MinimaxPlayer

        return MinimaxPlayer(
            player_id, max_depth=2, print_messages=False
        )  # 深度2，平衡强度和速度

    elif opponent_type == "mixed":
        # 混合对手：70%随机，20%贪心，10%minimax
        opponents = [
            (RandomPlayer(player_id), 0.7),
            (GreedyPlayer(player_id, print_messages=False), 0.2),
            (MinimaxPlayer(player_id, max_depth=1, print_messages=False), 0.1),
        ]
        return MixedOpponent(opponents, player_id)

    else:
        print(f"未知对手类型: {opponent_type}，使用随机对手")
        from AgentFight import RandomPlayer

        return RandomPlayer(player_id)


def get_agent_specific_info(agent, print_q_loss: bool = True) -> str:
    """获取智能体特定的信息字符串"""
    info_parts = []

    # Q值信息 (AQ 专有)
    if hasattr(agent, "q_values_history") and agent.q_values_history and print_q_loss:
        recent_q = np.mean(agent.q_values_history[-20:])
        info_parts.append(f"Q值 = {recent_q:.3f}")

    # 损失信息 (两者都有)
    if hasattr(agent, "losses") and agent.losses and print_q_loss:
        recent_loss = np.mean(agent.losses[-10:])
        info_parts.append(f"损失 = {recent_loss:.4f}")

    return ", " + ", ".join(info_parts) if info_parts else ""


def get_agent_hyperparams_info(agent) -> str:
    """获取智能体超参数信息"""
    info_parts = []

    # Epsilon
    if hasattr(agent, "epsilon"):
        info_parts.append(f"ε = {agent.epsilon:.3f}")

    # 学习率
    if hasattr(agent, "get_learning_rate"):
        lr = agent.get_learning_rate()
        if lr is not None:
            info_parts.append(f"lr = {lr:.6f}")
    elif hasattr(agent, "learning_rate"):
        info_parts.append(f"lr = {agent.learning_rate:.6f}")

    # 折扣因子 (AQ 专有)
    if hasattr(agent, "discount_factor"):
        info_parts.append(f"γ = {agent.discount_factor:.3f}")
    elif hasattr(agent, "gamma"):
        info_parts.append(f"γ = {agent.gamma:.3f}")

    return ", " + ", ".join(info_parts) if info_parts else ""


def update_agent_hyperparams(agent, batch_win_rate: float, episode: int, phase: str):
    """统一更新智能体超参数, 主要逻辑移到智能体内部"""
    # 主要的超参数更新逻辑现在在BaseTrainer._handle_episode_end中处理
    # 这里只处理特殊情况或后备逻辑

    # 折扣因子更新 (AQ 专有) - 如果BaseTrainer没有处理的话
    if hasattr(agent, "update_discount_factor") and not hasattr(agent, "phase_configs"):
        # 只有非DQN类型的智能体才需要在这里处理
        if hasattr(agent, "update_discount_factor"):
            agent.update_discount_factor(batch_win_rate)


def calculate_recent_win_rate(
    data_manager, start_episode: int, episode: int, window: int
) -> float:
    """计算最近窗口内的胜率"""
    if not data_manager:
        return 0.0

    recent_wins = 0
    recent_episodes = min(window, episode + 1)
    for i in range(
        max(0, start_episode + episode - recent_episodes + 1),
        start_episode + episode + 1,
    ):
        if (
            len(data_manager.current_session["training_history"]["wins"]) > i
            and data_manager.current_session["training_history"]["wins"][i] == 1
        ):
            recent_wins += 1

    return recent_wins / recent_episodes if recent_episodes > 0 else 0.0


def train_with_curriculum_generic(
    agent: Player,
    opponent: Player,
    trainer_class: Type[BaseTrainer],
    episodes: int = 2000,
    data_manager: TrainingDataManager = None,
    print_interval: int = 50,
    opponent_type: str = "random",
    **trainer_kwargs,
):
    """优化的课程式训练函数 - 使用分阶段epsilon控制"""
    total_episodes = episodes
    agent_type = agent.__class__.__name__

    training_start_time = time.time()
    phase_times = {}

    print(f"开始 {agent_type} 分阶段课程式训练，对手策略: {opponent_type}")

    # 显示分阶段配置 - 使用新的统一配置键名称
    if hasattr(agent, "phase_configs"):
        print(f"\n=== {agent_type} 分阶段配置 ===")
        for phase, config in agent.phase_configs.items():
            # 只使用新的配置键名称
            epsilon_force = config.get("epsilon_force_until", 0)
            epsilon_min = config.get("epsilon_min", 0.0)
            lr_force = config.get("lr_force_until", 0)
            lr_freq = config.get("lr_update_frequency", 100)

            print(f"{phase} ({config['description']}):")
            print(f"  Epsilon: 强制{epsilon_force}回合, 最小ε={epsilon_min:.2f}")
            print(f"  学习率: 强制{lr_force}回合, 频率={lr_freq}")

            # 如果是CNN-DQN，显示额外信息
            if "cnn_weight" in config:
                print(f"  CNN权重: {config['cnn_weight']:.1f}")

    # 保存原始参数
    original_epsilon_decay = getattr(agent, "epsilon_decay", None)

    # 阶段配置 - 保持4:4:2的分配
    phase1_ratio = 0.4
    phase2_ratio = 0.4
    phase3_ratio = 0.2
    print(f"使用标准阶段分配: {phase1_ratio:.0%}/{phase2_ratio:.0%}/{phase3_ratio:.0%}")

    # 阶段1: 基础学习
    phase1_episodes = int(total_episodes * phase1_ratio)
    print(f"\n阶段1: 基础学习 ({phase1_episodes} episodes) - 对手: 随机AI")

    phase1_opponent = create_opponent("random", 1 - agent.player_id)
    trainer = trainer_class(agent, phase1_opponent, **trainer_kwargs)

    # 阶段1设置 - 使用分阶段控制，不再手动设置epsilon_decay
    if hasattr(trainer, "target_update_freq"):
        trainer.target_update_freq = 50

    phase1_start_time = time.time()
    phase1_stats = run_training_phase(
        trainer,
        agent,
        phase1_episodes,
        "phase1",
        0,
        data_manager,
        print_interval,
        phase1_start_time,
    )
    phase1_end_time = time.time()
    phase_times["phase1"] = phase1_end_time - phase1_start_time

    print(f"\n阶段1完成! 当前状态:{get_agent_hyperparams_info(agent)}")
    print(
        f"阶段1胜率: {phase1_stats['win_rate']:.3f}, 平均奖励: {phase1_stats['avg_reward']:.3f}, "
        f"平均步长: {phase1_stats.get('avg_steps', 0):.1f}"
    )
    print(
        f"阶段1总耗时: {phase_times['phase1']:.1f}秒, 平均: {phase_times['phase1']/phase1_episodes:.2f}秒/回合"
    )

    # 阶段2: 进阶学习
    phase2_episodes = int(total_episodes * phase2_ratio)

    if opponent_type == "progressive":
        phase2_opponent = create_opponent("greedy", 1 - agent.player_id)
        print(f"\n阶段2: 进阶学习 ({phase2_episodes} episodes) - 对手: 贪心AI")
    elif opponent_type in ["greedy", "minimax"]:
        phase2_opponent = create_opponent(opponent_type, 1 - agent.player_id)
        opponent_name = "贪心AI" if opponent_type == "greedy" else "Minimax AI"
        print(f"\n阶段2: 进阶学习 ({phase2_episodes} episodes) - 对手: {opponent_name}")
    elif opponent_type == "mixed":
        phase2_opponent = create_opponent("mixed", 1 - agent.player_id)
        print(f"\n阶段2: 进阶学习 ({phase2_episodes} episodes) - 对手: 混合AI")
    else:
        phase2_opponent = create_opponent("random", 1 - agent.player_id)
        print(f"\n阶段2: 进阶学习 ({phase2_episodes} episodes) - 对手: 随机AI")

    trainer.opponent = phase2_opponent

    if hasattr(trainer, "target_update_freq"):
        trainer.target_update_freq = 75

    phase2_start_time = time.time()
    phase2_stats = run_training_phase(
        trainer,
        agent,
        phase2_episodes,
        "phase2",
        phase1_episodes,
        data_manager,
        print_interval,
        phase2_start_time,
    )
    phase2_end_time = time.time()
    phase_times["phase2"] = phase2_end_time - phase2_start_time

    print(f"\n阶段2完成! 当前状态:{get_agent_hyperparams_info(agent)}")
    print(
        f"阶段2胜率: {phase2_stats['win_rate']:.3f}, 平均奖励: {phase2_stats['avg_reward']:.3f}, "
        f"平均步长: {phase2_stats.get('avg_steps', 0):.1f}"
    )
    print(
        f"阶段2总耗时: {phase_times['phase2']:.1f}秒, 平均: {phase_times['phase2']/phase2_episodes:.2f}秒/回合"
    )

    # 阶段3: 策略精炼
    phase3_episodes = total_episodes - phase1_episodes - phase2_episodes

    if opponent_type == "progressive":
        phase3_opponent = create_opponent("minimax", 1 - agent.player_id)
        print(f"\n阶段3: 策略精炼 ({phase3_episodes} episodes) - 对手: Minimax AI")
    elif opponent_type == "minimax":
        phase3_opponent = trainer.opponent
        print(f"\n阶段3: 策略精炼 ({phase3_episodes} episodes) - 对手: Minimax AI")
    elif opponent_type in ["greedy", "mixed"]:
        phase3_opponent = trainer.opponent
        opponent_name = "贪心AI" if opponent_type == "greedy" else "混合AI"
        print(f"\n阶段3: 策略精炼 ({phase3_episodes} episodes) - 对手: {opponent_name}")
    else:
        phase3_opponent = create_opponent("greedy", 1 - agent.player_id)
        print(f"\n阶段3: 策略精炼 ({phase3_episodes} episodes) - 对手: 贪心AI")

    trainer.opponent = phase3_opponent

    if hasattr(trainer, "target_update_freq"):
        trainer.target_update_freq = 100

    phase3_start_time = time.time()
    phase3_stats = run_training_phase(
        trainer,
        agent,
        phase3_episodes,
        "phase3",
        phase1_episodes + phase2_episodes,
        data_manager,
        print_interval,
        phase3_start_time,
    )
    phase3_end_time = time.time()
    phase_times["phase3"] = phase3_end_time - phase3_start_time

    # 总结训练时间
    total_training_time = phase3_end_time - training_start_time

    print(f"\n{agent_type} 分阶段课程训练完成!")
    print(f"最终状态:{get_agent_hyperparams_info(agent)}")
    print(
        f"阶段3胜率: {phase3_stats['win_rate']:.3f}, 平均奖励: {phase3_stats['avg_reward']:.3f}, "
        f"平均步长: {phase3_stats.get('avg_steps', 0):.1f}"
    )

    if hasattr(agent, "get_stats"):
        total_stats = agent.get_stats()
        print(f"总体胜率: {total_stats['win_rate']:.3f}")

    print(f"\n=== 分阶段训练耗时统计 ===")
    print(
        f"阶段1 ({phase1_episodes} episodes): {phase_times['phase1']:.1f}秒 (平均 {phase_times['phase1']/phase1_episodes:.2f}秒/回合)"
    )
    print(
        f"阶段2 ({phase2_episodes} episodes): {phase_times['phase2']:.1f}秒 (平均 {phase_times['phase2']/phase2_episodes:.2f}秒/回合)"
    )
    print(
        f"阶段3 ({phase3_episodes} episodes): {phase_times['phase3']:.1f}秒 (平均 {phase_times['phase3']/phase3_episodes:.2f}秒/回合)"
    )
    print(f"总训练时间: {total_training_time:.1f}秒 ({total_training_time/60:.1f}分钟)")
    print(f"总体平均: {total_training_time/total_episodes:.2f}秒/回合")

    # 详细统计
    if data_manager:
        summary = data_manager.get_summary_stats()
        print(f"\n=== 详细训练统计 ===")
        if "average_steps" in summary:
            print(
                f"平均步数: {summary['average_steps']:.1f} ± {summary['steps_std']:.1f}"
            )
            print(f"步数范围: {summary['min_steps']} - {summary['max_steps']}")
        if "average_episode_time" in summary:
            print(
                f"平均每回合时间: {summary['average_episode_time']:.3f} ± {summary['episode_time_std']:.3f} 秒"
            )
        print(f"总胜率: {summary['final_win_rate']:.3f}")
        print(
            f"平均奖励: {summary['average_reward']:.3f} ± {summary['reward_std']:.3f}"
        )

    return data_manager.current_session["training_history"] if data_manager else {}


def run_training_phase(
    trainer: BaseTrainer,
    agent: Player,
    num_episodes: int,
    phase_name: str,
    episode_offset: int,
    data_manager: TrainingDataManager,
    print_interval: int,
    phase_start_time: float,
):
    """运行单个训练阶段 - 超参数更新由BaseTrainer统一处理"""

    # 设置训练器的阶段信息
    trainer.set_phase(phase_name, episode_offset)

    # 设置智能体的阶段信息
    if hasattr(agent, "set_phase"):
        agent.set_phase(phase_name)

    # 添加步长统计
    batch_steps = []

    for episode in range(num_episodes):
        episode_start_time = time.time()
        total_reward, steps, result = trainer.train_episode()
        episode_end_time = time.time()
        episode_time = episode_end_time - episode_start_time

        # 记录步长
        batch_steps.append(steps)

        # 定期输出进度 - 使用trainer的批次统计
        if episode % print_interval == 0 or episode == num_episodes - 1:
            batch_win_rate = trainer.get_batch_win_rate()
            batch_avg_reward = trainer.get_batch_avg_reward()

            # 计算平均步长 - 使用最近收集的步长数据
            if batch_steps:
                # 计算从上次打印到现在的平均步长
                if episode >= print_interval:
                    # 取最近print_interval个episode的步长
                    recent_steps = (
                        batch_steps[-print_interval:]
                        if len(batch_steps) >= print_interval
                        else batch_steps
                    )
                else:
                    # 取从开始到现在的所有步长
                    recent_steps = batch_steps
                avg_steps = sum(recent_steps) / len(recent_steps)
            else:
                avg_steps = 0.0

            # 计算耗时
            current_time = time.time()
            batch_time = current_time - phase_start_time
            avg_time_per_episode = batch_time / (episode + 1)

            # 组装输出信息
            param_info = get_agent_hyperparams_info(agent)
            agent_info = get_agent_specific_info(agent, print_q_loss=True)
            time_info = (
                f", 用时 = {batch_time:.1f}s, 平均 = {avg_time_per_episode:.2f}s/ep"
            )

            print(
                f"{phase_name.upper()} - 回合 {episode}: 奖励 = {total_reward:.2f}, 步数 = {steps}, "
                f"胜 = {trainer.batch_wins}, 负 = {trainer.batch_losses}, 平 = {trainer.batch_draws}, "
                f"批次胜率 = {batch_win_rate:.3f}, 平均奖励 = {batch_avg_reward:.3f}, "
                f"平均步长 = {avg_steps:.1f}{param_info}{agent_info}{time_info}"
            )

            # 记录批次统计到数据管理器 - 传递平均步长
            if data_manager:
                data_manager.log_batch_stats(
                    trainer.batch_wins,
                    trainer.batch_losses,
                    trainer.batch_draws,
                    batch_win_rate,
                    avg_steps,  # 使用计算出的平均步长
                )

        # 超参数更新现在完全由BaseTrainer._handle_episode_end处理
        # 这里不再需要手动调用update_agent_hyperparams

        # 记录每个episode的详细数据
        if data_manager:
            data_manager.log_episode(
                episode=episode_offset + episode,
                reward=total_reward,
                result=result,
                learning_rate=(
                    agent.get_learning_rate()
                    if hasattr(agent, "get_learning_rate")
                    else getattr(agent, "learning_rate", None)
                ),
                epsilon=getattr(agent, "epsilon", None),
                loss=(
                    agent.losses[-1]
                    if hasattr(agent, "losses") and agent.losses
                    else None
                ),
                phase=f'{phase_name}_{"exploration" if phase_name == "phase1" else "balance" if phase_name == "phase2" else "refinement"}',
                steps=steps,
                episode_time=episode_time,
                discount_factor=getattr(agent, "discount_factor", None),
            )

    # 计算整个阶段的平均步长
    final_avg_steps = sum(batch_steps) / len(batch_steps) if batch_steps else 0.0

    return {
        "wins": trainer.batch_wins,
        "losses": trainer.batch_losses,
        "draws": trainer.batch_draws,
        "win_rate": trainer.get_batch_win_rate(),
        "avg_reward": trainer.get_batch_avg_reward(),
        "avg_steps": final_avg_steps,  # 添加平均步长到返回结果
    }


def train_or_load_model_generic(
    agent_class: Type[Player],
    trainer_class: Type[BaseTrainer],
    agent_config: Dict[str, Any],
    model_name: str,
    force_retrain: bool = False,
    episodes: int = 2000,
    lr_strategy: str = "adaptive",
    print_interval: int = 50,
    opponent_type: str = "random",
    exploration_strategy: str = "guided",  # 新增参数
    **trainer_kwargs,
):
    """
    通用的训练或加载模型函数 - 支持选择对手类型和探索策略
    """
    # 更新智能体配置中的探索策略
    if "exploration_strategy" in agent_config:
        agent_config["exploration_strategy"] = exploration_strategy

    # 创建智能体实例
    agent = agent_class(**agent_config)

    # 创建初始对手（在课程训练中会被替换）
    initial_opponent = create_opponent("random", 1 - agent.player_id)

    # 设置学习率策略
    if hasattr(agent, "enable_adaptive_lr") and hasattr(agent, "disable_adaptive_lr"):
        if lr_strategy == "adaptive":
            agent.enable_adaptive_lr()
            print("使用自适应学习率策略")
        elif lr_strategy == "fixed":
            agent.disable_adaptive_lr()
            print("使用固定学习率策略")
        else:  # hybrid for DQN
            print(f"使用{lr_strategy}学习率策略")

    # 打印探索策略信息
    if hasattr(agent, "exploration_strategy"):
        print(f"探索策略: {agent.exploration_strategy}")

    # 检查是否存在已训练的模型
    model_path = os.path.join("model_data", f"{model_name}.pkl")
    model_exists = os.path.exists(model_path)

    if model_exists and not force_retrain:
        print(f"发现已训练的模型: {model_path}")
        if agent.load_model(model_name):
            print("模型加载成功!")
            agent.epsilon = 0.0
            agent.ai_type = f"{agent.__class__.__name__} (Trained)"
        else:
            print("模型加载失败，将重新训练...")
            model_exists = False

    if not model_exists or force_retrain:
        if force_retrain:
            print("强制重新训练模型...")
        else:
            print("未找到已训练模型，开始训练...")

        print(f"对手策略: {opponent_type}")

        # 创建数据管理器
        data_manager = TrainingDataManager()
        session_name = f"{agent.__class__.__name__}_{lr_strategy}_{opponent_type}_{exploration_strategy}_{episodes}eps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        data_manager.start_session(agent, session_name)

        # 使用通用课程式训练，包含对手类型
        combined_history = train_with_curriculum_generic(
            agent,
            initial_opponent,
            trainer_class,
            episodes,
            data_manager,
            print_interval,
            opponent_type,
            **trainer_kwargs,
        )

        # 结束数据记录会话
        final_stats = {
            "training_episodes": episodes,
            "lr_strategy": lr_strategy,
            "opponent_type": opponent_type,  # 记录对手类型
            "exploration_strategy": exploration_strategy,  # 记录探索策略
            "final_epsilon": getattr(agent, "epsilon", None),
            "final_learning_rate": (
                agent.get_learning_rate()
                if hasattr(agent, "get_learning_rate")
                else getattr(agent, "learning_rate", None)
            ),
            "final_discount_factor": getattr(agent, "discount_factor", None),
        }
        data_manager.end_session(agent, final_stats)

        # 绘制训练历史
        data_manager.plot_training_history()

        # 保存模型
        agent.save_model(model_name)

        print(f"训练完成! 最终epsilon: {getattr(agent, 'epsilon', 'N/A')}")
        if hasattr(agent, "get_stats"):
            print(f"最终胜率: {agent.get_stats()['win_rate']:.3f}")

        # 设置为测试模式
        agent.epsilon = 0.0
        agent.ai_type = f"{agent.__class__.__name__} (Trained)"

    return agent, create_opponent("random", 1 - agent.player_id)  # 返回随机对手用于测试


def test_agent_generic(
    agent: Player,
    opponent: Player,
    num_games: int = 100,
    show_individual: bool = False,
    display: bool = False,
):
    """通用的智能体测试函数"""
    from base import Game

    # 设置为测试模式
    if hasattr(agent, "set_training_mode"):
        agent.set_training_mode(False)

    print(f"\n开始测试 {num_games} 场游戏...")
    print(f"测试模式 - epsilon: {getattr(agent, 'epsilon', 'N/A')}")

    wins = 0
    draws = 0

    for i in range(num_games):
        if display:
            game = Game(agent, opponent, display=True, delay=0.1)
        else:
            game = Game(agent, opponent, display=False)
        result = game.run()

        if result == 0:  # Agent wins
            wins += 1
        elif result == 2:  # Draw
            draws += 1

        if show_individual and num_games <= 10:
            print(
                f"游戏 {i+1}: {'胜利' if result == 0 else '失败' if result == 1 else '平局'}"
            )

    win_rate = wins / num_games
    draw_rate = draws / num_games

    print(f"\n测试结果:")
    print(f"胜利: {wins}/{num_games} ({win_rate:.3f})")
    print(f"平局: {draws}/{num_games} ({draw_rate:.3f})")
    print(f"失败: {num_games - wins - draws}/{num_games}")
    print(f"有效胜率 (胜+0.5*平): {win_rate + 0.5 * draw_rate:.3f}")

    return {
        "wins": wins,
        "draws": draws,
        "losses": num_games - wins - draws,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "effective_win_rate": win_rate + 0.5 * draw_rate,
    }


# 智能体配置映射
AGENT_CONFIGS = {
    "dqn": {
        "agent_class": None,  # 将在运行时导入
        "trainer_class": None,
        "agent_config": {
            "player_id": 0,
            "state_size": 448,  # 修改：7*8*8 = 448 (增强状态)
            "action_size": 280,
            "learning_rate": 1e-2,
            "epsilon": 0.9,
            "epsilon_min": 0.02,
            "epsilon_decay": 0.995,
            "batch_size": 64,
            "memory_size": 40000,
            "use_dueling": True,
            "use_double": True,
            "exploration_strategy": "guided",  # 新增：默认使用引导探索
        },
        "model_name": "final_D3QNAgent",
        "trainer_kwargs": {},
    },
    "cnn-dqn": {  # 新增CNN-DQN配置
        "agent_class": None,
        "trainer_class": None,
        "agent_config": {
            "player_id": 0,
            "action_size": 280,
            "learning_rate": 1e-2,  # 稍微降低学习率
            "epsilon": 0.9,
            "epsilon_min": 0.02,
            "epsilon_decay": 0.995,
            "batch_size": 64,  # 降低batch size以适应更复杂的网络
            "memory_size": 40000,
            "exploration_strategy": "guided",
            "cnn_model_path": None,  # 可以指定预训练CNN模型路径
        },
        "model_name": "final_CNNDQNAgent",
        "trainer_kwargs": {},
    },
    "aq": {
        "agent_class": None,  # 将在运行时导入
        "trainer_class": None,
        "agent_config": {
            "player_id": 0,
            "learning_rate": 0.02,
            "epsilon": 0.9,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.995,
            "discount_factor": 0.3,
            "discount_max": 0.95,
            "discount_growth": 0.002,
        },
        "model_name": "final_ApproximateQAgent",
        "trainer_kwargs": {},
    },
}


def get_agent_imports(agent_type: str):
    """动态导入智能体类"""
    if agent_type == "dqn":
        from DQN import DQNAgent, DQNTrainer

        return DQNAgent, DQNTrainer
    elif agent_type == "cnn-dqn":  # 新增CNN-DQN导入
        from DQN_CNN import CNNEnhancedDQNAgent, CNNDQNTrainer

        return CNNEnhancedDQNAgent, CNNDQNTrainer
    elif agent_type == "aq":
        from ApproximateQAgent import ApproximateQAgent, ApproximateQTrainer

        return ApproximateQAgent, ApproximateQTrainer
    else:
        raise ValueError(f"不支持的智能体类型: {agent_type}")


def main():
    """主函数 - 支持CNN-DQN"""
    parser = argparse.ArgumentParser(description="通用智能体训练和测试系统")
    parser.add_argument(
        "--agent",
        choices=["dqn", "cnn-dqn", "aq"],
        required=True,
        help="选择智能体类型: dqn (DQN Agent), cnn-dqn (CNN-DQN Agent) 或 aq (Approximate Q Agent)",
    )
    parser.add_argument("--retrain", action="store_true", help="强制重新训练模型")
    parser.add_argument("--episodes", type=int, default=2000, help="训练回合数")
    parser.add_argument("--test-games", type=int, default=100, help="测试游戏数量")
    parser.add_argument("--no-display", action="store_true", help="不显示游戏界面")
    parser.add_argument(
        "--lr-strategy",
        choices=["adaptive", "fixed", "hybrid"],
        default="adaptive",
        help="学习率调整策略",
    )
    parser.add_argument("--test-only", action="store_true", help="仅测试，不训练")
    parser.add_argument(
        "--print-interval", type=int, default=50, help="训练进度输出间隔"
    )

    # 对手选择参数
    parser.add_argument(
        "--opponent",
        choices=["random", "greedy", "minimax", "mixed", "progressive"],
        default="progressive",
        help="训练对手类型: random(随机), greedy(贪心), minimax(极小极大), mixed(混合), progressive(渐进式)",
    )

    # 探索策略参数（对DQN和CNN-DQN有效）
    parser.add_argument(
        "--exploration",
        choices=["random", "guided"],
        default="guided",
        help="探索策略: random(随机探索), guided(引导探索，基于贪心评估)",
    )

    # CNN模型路径参数（仅对CNN-DQN有效）
    parser.add_argument(
        "--cnn-model",
        type=str,
        default=None,
        help="预训练CNN模型路径（仅对CNN-DQN有效）",
    )

    args = parser.parse_args()

    # 获取智能体配置
    config = AGENT_CONFIGS[args.agent].copy()
    agent_class, trainer_class = get_agent_imports(args.agent)
    config["agent_class"] = agent_class
    config["trainer_class"] = trainer_class

    # 如果是CNN-DQN，添加CNN模型路径
    if args.agent == "cnn-dqn" and args.cnn_model:
        config["agent_config"]["cnn_model_path"] = args.cnn_model

    print(f"选择的智能体: {agent_class.__name__}")
    print(f"选择的对手策略: {args.opponent}")
    if args.agent in ["dqn", "cnn-dqn"]:
        print(f"选择的探索策略: {args.exploration}")

    # 训练或加载模型
    if not args.test_only:
        agent, opponent = train_or_load_model_generic(
            agent_class=config["agent_class"],
            trainer_class=config["trainer_class"],
            agent_config=config["agent_config"],
            model_name=config["model_name"],
            force_retrain=args.retrain,
            episodes=args.episodes,
            lr_strategy=args.lr_strategy,
            print_interval=args.print_interval,
            opponent_type=args.opponent,  # 传递对手类型
            exploration_strategy=args.exploration,  # 传递探索策略
            **config["trainer_kwargs"],
        )
    else:
        # 仅测试模式：直接加载模型
        print("仅测试模式，加载已训练模型...")
        agent_config = config["agent_config"].copy()
        agent = agent_class(**agent_config)
        if not agent.load_model(config["model_name"]):
            print("无法加载模型，请先训练!")
            return
        opponent = create_opponent("random", 1)  # 测试时使用随机对手

    # 测试
    test_results = test_agent_generic(
        agent,
        opponent,
        num_games=args.test_games,
        show_individual=(args.test_games <= 10),
        display=not args.no_display,
    )

    print(f"\n{agent.__class__.__name__} 测试完成!")


if __name__ == "__main__":
    main()

# 运行示例:
# 使用引导探索的渐进式训练（推荐）
# python train_and_record.py --agent dqn --retrain --episodes 3000 --opponent progressive --exploration guided --lr-strategy hybrid --test-games 100 --no-display --print-interval 100

# python train_and_record.py --agent cnn-dqn --retrain --episodes 3000 --opponent progressive --exploration guided --lr-strategy adaptive --test-games 100 --no-display --print-interval 100

# 使用随机探索对比
# python train_and_record.py --agent dqn --retrain --episodes 3000 --opponent progressive --exploration random --lr-strategy hybrid --test-games 100 --no-display --print-interval 50

# 与贪心AI训练，使用引导探索
# python train_and_record.py --agent dqn --retrain --episodes 3000 --opponent greedy --exploration guided --lr-strategy adaptive --test-games 100 --no-display --print-interval 50

# 与Minimax AI训练，使用引导探索
# python train_and_record.py --agent dqn --retrain --episodes 3000 --opponent minimax --exploration guided --lr-strategy adaptive --test-games 100 --no-display --print-interval 50

# 仅测试
# python train_and_record.py --agent dqn --test-only --test-games 50 --no-display
