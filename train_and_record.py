import numpy as np
import time
import os
import argparse
import copy
from datetime import datetime
from typing import Dict, Any, Callable, Type, List, Tuple
from base import Player, BaseTrainer, Board
from training_data_manager import TrainingDataManager


# 智能体配置映射
AGENT_CONFIGS = {
    "dqn": {
        "agent_class": None,  # 将在运行时导入
        "trainer_class": None,
        "agent_config": {
            "player_id": 0,
            "state_size": 448,  # 修改：7*8*8 = 448 (增强状态)
            "action_size": 280,
            "learning_rate": 5e-2,
            "epsilon": 0.9,
            "epsilon_min": 0.02,
            "epsilon_decay": 0.995,
            "batch_size": 128,
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
            "learning_rate": 5e-2,  # 稍微降低学习率
            "epsilon": 0.9,
            "epsilon_min": 0.02,
            "epsilon_decay": 0.995,
            "batch_size": 128,  # 降低batch size以适应更复杂的网络
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
    "ql": {  # 新增QL配置
        "agent_class": None,  # 将在运行时导入
        "trainer_class": None,
        "agent_config": {
            "player_id": 0,
            "learning_rate": 0.1,
            "discount_factor": 0.95,
            "epsilon": 0.9,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.05,
        },
        "model_name": "final_QLearningAgent",
        "trainer_kwargs": {},
    },
}


TRAINING_PRESETS = [  # 训练的预设，可以根据需求更改
    {
        "name": "DQN_standard_training",
        "agent_type": "dqn",
        "episodes": 3000,
        "opponent_type": "progressive",
        "exploration": "guided",
        "lr_strategy": "hybrid",
        "test_games": 100,
        "print_interval": 100,
        "display": False,
        "phase_config": {
            "phase1": {
                "ratio": 1.0,  # 占总回合的比例
                "opponent": "random",
                "epsilon_start": 0.9,
                "epsilon_end": 0.02,
                "epsilon_force_until": 200,
                "learning_rate": 0.001,
                "learning_rate_min": 0.0001,
                "learning_rate_max": 0.01,
            },
            "phase2": {
                "ratio": 0.0,
                "opponent": "greedy",
                "epsilon_start": 0.9,
                "epsilon_end": 0.2,
                "epsilon_force_until": 400,
                "learning_rate": 0.0005,
                "learning_rate_min": 0.0001,
                "learning_rate_max": 0.005,
            },
            "phase3": {
                "ratio": 0.0,
                "opponent": "minimax",
                "epsilon_start": 0.6,
                "epsilon_end": 0.1,
                "epsilon_force_until": 200,
                "learning_rate": 0.0003,
                "learning_rate_min": 0.0001,
                "learning_rate_max": 0.001,
            },
        },
    },
    {
        "name": "CNN-DQN_training",
        "agent_type": "cnn-dqn",
        "episodes": 4000,
        "opponent_type": "progressive",
        "exploration": "guided",
        "lr_strategy": "adaptive",
        "test_games": 100,
        "print_interval": 100,
        "display": False,
        "phase_config": {
            "phase1": {
                "ratio": 0.35,
                "opponent": "random",
                "epsilon_start": 0.95,
                "epsilon_end": 0.8,
                "epsilon_force_until": 350,
                "learning_rate": 0.001,
                "learning_rate_min": 0.0001,
                "learning_rate_max": 0.005,
            },
            "phase2": {
                "ratio": 0.45,
                "opponent": "greedy",
                "epsilon_start": 0.8,
                "epsilon_end": 0.5,
                "epsilon_force_until": 250,
                "learning_rate": 0.0005,
                "learning_rate_min": 0.00008,
                "learning_rate_max": 0.002,
            },
            "phase3": {
                "ratio": 0.2,
                "opponent": "minimax",
                "epsilon_start": 0.5,
                "epsilon_end": 0.05,
                "epsilon_force_until": 100,
                "learning_rate": 0.0002,
                "learning_rate_min": 0.00005,
                "learning_rate_max": 0.001,
            },
        },
    },
    {
        "name": "QL_training",
        "agent_type": "ql",
        "episodes": 3000,
        "opponent_type": "progressive",
        "lr_strategy": "adaptive",
        "test_games": 100,
        "print_interval": 100,
        "display": False,
        "phase_config": {
            "phase1": {
                "ratio": 1.0,
                "opponent": "random",
                "epsilon_start": 0.9,
                "epsilon_end": 0.02,
                "epsilon_force_until": 100,
                "learning_rate": 0.01,
                "learning_rate_min": 0.001,
                "learning_rate_max": 0.05,
            },
            "phase2": {
                "ratio": 0.0,
                "opponent": "greedy",
                "epsilon_start": 0.7,
                "epsilon_end": 0.4,
                "epsilon_force_until": 100,
                "learning_rate": 0.1,
                "learning_rate_min": 0.03,
                "learning_rate_max": 0.2,
            },
            "phase3": {
                "ratio": 0.0,
                "opponent": "minimax",
                "epsilon_start": 0.4,
                "epsilon_end": 0.05,
                "epsilon_force_until": 50,
                "learning_rate": 0.05,
                "learning_rate_min": 0.01,
                "learning_rate_max": 0.1,
            },
        },
    },
    {
        "name": "DQN_guided_to_random_training",
        "agent_type": "dqn",
        "episodes": 4000,  # 总训练回合数
        "opponent_type": "random_only",  # 新的对手类型，全程使用random
        "exploration": "guided_to_random",  # 新的探索策略
        "lr_strategy": "hybrid",
        "test_games": 100,
        "print_interval": 100,
        "display": False,
        "phase_config": {
            "phase1": {
                "ratio": 0.4,  # 60%的时间用guided探索
                "opponent": "random",
                "epsilon_start": 0.9,
                "epsilon_end": 0.02,  # 第一阶段结束时epsilon应该达到0.02
                "epsilon_force_until": 400,  # 前500回合保持高epsilon
                "learning_rate": 0.001,
                "learning_rate_min": 0.0001,
                "learning_rate_max": 0.01,
                "exploration_strategy": "guided",  # 第一阶段使用guided
                "target_update_freq": 200,  # 目标网络更新频率
            },
            "phase2": {
                "ratio": 0.6,  # 60%的时间用random探索
                "opponent": "random",
                "epsilon_start": 0.5,
                "epsilon_end": 0.02,
                "epsilon_force_until": 200,  # 前200回合保持epsilon稳定
                "learning_rate": 0.0005,
                "learning_rate_min": 0.0001,
                "learning_rate_max": 0.005,
                "exploration_strategy": "random",  # 第二阶段使用random
                "target_update_freq": 1000,  # 降低更新频率到1000
            },
            "phase3": {
                "ratio": 0.0,  # 跳过第三阶段
                "opponent": "random",
                "epsilon_start": 0.02,
                "epsilon_end": 0.02,
                "epsilon_force_until": 0,
                "learning_rate": 0.0003,
                "learning_rate_min": 0.0001,
                "learning_rate_max": 0.001,
            },
        },
    },
    {
        "name": "AQ_training",
        "agent_type": "aq",
        "episodes": 2000,
        "opponent_type": "progressive",
        "lr_strategy": "adaptive",
        "test_games": 100,
        "print_interval": 100,
        "display": False,
        "phase_config": {
            "phase1": {
                "ratio": 1.0,
                "opponent": "random",
                "epsilon_start": 0.9,
                "epsilon_end": 0.02,
                "epsilon_force_until": 100,
                "learning_rate": 0.02,
                "learning_rate_min": 0.005,
                "learning_rate_max": 0.01,
            },
            "phase2": {
                "ratio": 0.0,
                "opponent": "greedy",
                "epsilon_start": 0.7,
                "epsilon_end": 0.4,
                "epsilon_force_until": 100,
                "learning_rate": 0.1,
                "learning_rate_min": 0.03,
                "learning_rate_max": 0.2,
            },
            "phase3": {
                "ratio": 0.0,
                "opponent": "minimax",
                "epsilon_start": 0.4,
                "epsilon_end": 0.05,
                "epsilon_force_until": 50,
                "learning_rate": 0.05,
                "learning_rate_min": 0.01,
                "learning_rate_max": 0.1,
            },
        },
    },
]


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
        opponent_type: 对手类型 ('random', 'greedy', 'minimax', 'mixed', 'random_only')
        player_id: 玩家ID（默认为1）
    """
    if opponent_type == "random" or opponent_type == "random_only":
        from AgentFight import RandomPlayer

        return RandomPlayer(player_id)

    elif opponent_type == "greedy":
        from AgentFight import GreedyPlayer

        return GreedyPlayer(player_id, print_messages=False)

    elif opponent_type == "minimax":
        from AgentFight import MinimaxPlayer

        return MinimaxPlayer(player_id, max_depth=2, print_messages=False)

    elif opponent_type == "mixed":
        # 混合对手：70%随机，20%贪心，10%minimax
        from AgentFight import RandomPlayer, GreedyPlayer, MinimaxPlayer

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
    """优化的课程式训练函数 - 使用分阶段epsilon控制，支持日志记录"""
    total_episodes = episodes
    agent_type = agent.__class__.__name__

    training_start_time = time.time()
    phase_times = {}

    start_message = f"开始 {agent_type} 分阶段课程式训练，对手策略: {opponent_type}"
    print(start_message)
    if data_manager:
        data_manager.log_message(start_message)

    # 显示分阶段配置 - 使用新的统一配置键名称
    if hasattr(agent, "phase_configs"):
        config_message = f"\n=== {agent_type} 分阶段配置 ==="
        for phase, config in agent.phase_configs.items():
            # 只使用新的配置键名称
            epsilon_force = config.get("epsilon_force_until", 0)
            epsilon_min = config.get("epsilon_min", 0.0)
            lr_force = config.get("lr_force_until", 0)
            lr_freq = config.get("lr_update_frequency", 100)

            config_message += f"\n{phase} ({config['description']}):"
            config_message += (
                f"\n  Epsilon: 强制{epsilon_force}回合, 最小ε={epsilon_min:.2f}"
            )
            config_message += f"\n  学习率: 强制{lr_force}回合, 频率={lr_freq}"

            # 如果是CNN-DQN，显示额外信息
            if "cnn_weight" in config:
                config_message += f"\n  CNN权重: {config['cnn_weight']:.1f}"

        print(config_message)
        if data_manager:
            data_manager.log_message(config_message)

    # 保存原始参数
    original_epsilon_decay = getattr(agent, "epsilon_decay", None)

    # 阶段配置 - 保持4:4:2的分配
    phase1_ratio = 0.4
    phase2_ratio = 0.4
    phase3_ratio = 0.2

    ratio_message = (
        f"使用标准阶段分配: {phase1_ratio:.0%}/{phase2_ratio:.0%}/{phase3_ratio:.0%}"
    )
    print(ratio_message)
    if data_manager:
        data_manager.log_message(ratio_message)

    # 阶段1: 基础学习
    phase1_episodes = int(total_episodes * phase1_ratio)

    if data_manager:
        data_manager.log_phase_start("阶段1: 基础学习", phase1_episodes, "random")

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

    if data_manager:
        data_manager.log_phase_end(
            "阶段1",
            phase1_stats,
            {
                "total_time": phase_times["phase1"],
                "avg_time": phase_times["phase1"] / phase1_episodes,
            },
        )

    completion_message = f"\n阶段1完成! 当前状态:{get_agent_hyperparams_info(agent)}"
    completion_message += f"\n阶段1胜率: {phase1_stats['win_rate']:.3f}, 平均奖励: {phase1_stats['avg_reward']:.3f}, "
    completion_message += f"平均步长: {phase1_stats.get('avg_steps', 0):.1f}"
    completion_message += f"\n阶段1总耗时: {phase_times['phase1']:.1f}秒, 平均: {phase_times['phase1']/phase1_episodes:.2f}秒/回合"

    print(completion_message)
    if data_manager:
        data_manager.log_message(completion_message)

    # 阶段2: 进阶学习
    phase2_episodes = int(total_episodes * phase2_ratio)
    
    if phase2_episodes > 0:
        if opponent_type == "progressive":
            phase2_opponent = create_opponent("greedy", 1 - agent.player_id)
            phase2_opponent_name = "贪心AI"
        elif opponent_type in ["greedy", "minimax"]:
            phase2_opponent = create_opponent(opponent_type, 1 - agent.player_id)
            phase2_opponent_name = "贪心AI" if opponent_type == "greedy" else "Minimax AI"
        elif opponent_type == "mixed":
            phase2_opponent = create_opponent("mixed", 1 - agent.player_id)
            phase2_opponent_name = "混合AI"
        else:
            phase2_opponent = create_opponent("random", 1 - agent.player_id)
            phase2_opponent_name = "随机AI"

        if data_manager:
            data_manager.log_phase_start(
                "阶段2: 进阶学习", phase2_episodes, phase2_opponent_name
            )

        print(
            f"\n阶段2: 进阶学习 ({phase2_episodes} episodes) - 对手: {phase2_opponent_name}"
        )

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

        if data_manager:
            data_manager.log_phase_end(
                "阶段2",
                phase2_stats,
                {
                    "total_time": phase_times["phase2"],
                    "avg_time": phase_times["phase2"] / phase2_episodes,
                },
            )

        completion_message = f"\n阶段2完成! 当前状态:{get_agent_hyperparams_info(agent)}"
        completion_message += f"\n阶段2胜率: {phase2_stats['win_rate']:.3f}, 平均奖励: {phase2_stats['avg_reward']:.3f}, "
        completion_message += f"平均步长: {phase2_stats.get('avg_steps', 0):.1f}"
        completion_message += f"\n阶段2总耗时: {phase_times['phase2']:.1f}秒, 平均: {phase_times['phase2']/phase2_episodes:.2f}秒/回合"

        print(completion_message)
        if data_manager:
            data_manager.log_message(completion_message)
    else:
        phase2_stats = {"wins": 0, "losses": 0, "draws": 0, "win_rate": 0.0, "avg_reward": 0.0, "avg_steps": 0.0}
        phase_times["phase2"] = 0.0
        print("\n阶段2: 跳过 (比例设置为0)")
        if data_manager:
            data_manager.log_message("阶段2: 跳过 (比例设置为0)")

    # 阶段3: 策略精炼
    phase3_episodes = total_episodes - phase1_episodes - phase2_episodes

    if phase3_episodes > 0:
        if opponent_type == "progressive":
            phase3_opponent = create_opponent("minimax", 1 - agent.player_id)
            phase3_opponent_name = "Minimax AI"
        elif opponent_type == "minimax":
            phase3_opponent = trainer.opponent
            phase3_opponent_name = "Minimax AI"
        elif opponent_type in ["greedy", "mixed"]:
            phase3_opponent = trainer.opponent
            phase3_opponent_name = "贪心AI" if opponent_type == "greedy" else "混合AI"
        else:
            phase3_opponent = create_opponent("greedy", 1 - agent.player_id)
            phase3_opponent_name = "贪心AI"

        if data_manager:
            data_manager.log_phase_start(
                "阶段3: 策略精炼", phase3_episodes, phase3_opponent_name
            )

        print(
            f"\n阶段3: 策略精炼 ({phase3_episodes} episodes) - 对手: {phase3_opponent_name}"
        )

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

        if data_manager:
            data_manager.log_phase_end(
                "阶段3",
                phase3_stats,
                {
                    "total_time": phase_times["phase3"],
                    "avg_time": phase_times["phase3"] / phase3_episodes,
                },
            )
    else:
        phase3_stats = {"wins": 0, "losses": 0, "draws": 0, "win_rate": 0.0, "avg_reward": 0.0, "avg_steps": 0.0}
        phase_times["phase3"] = 0.0
        print("\n阶段3: 跳过 (比例设置为0)")
        if data_manager:
            data_manager.log_message("阶段3: 跳过 (比例设置为0)")

    # 训练完成后的最终处理
    if hasattr(trainer, "finalize_training"):
        trainer.finalize_training()
    else:
        # 后备方案
        if hasattr(agent, "update_target_network"):
            agent.update_target_network()
            print("训练完成，执行最终目标网络更新")

        if hasattr(agent, "set_training_mode"):
            agent.set_training_mode(False)

    # 总结训练时间
    total_training_time = phase3_end_time - training_start_time

    final_message = f"\n{agent_type} 分阶段课程训练完成!"
    final_message += f"\n最终状态:{get_agent_hyperparams_info(agent)}"
    if phase3_episodes > 0:
        final_message += f"\n阶段3胜率: {phase3_stats['win_rate']:.3f}, 平均奖励: {phase3_stats['avg_reward']:.3f}, "
        final_message += f"平均步长: {phase3_stats.get('avg_steps', 0):.1f}"

    if hasattr(agent, "get_stats"):
        total_stats = agent.get_stats()
        final_message += f"\n总体胜率: {total_stats['win_rate']:.3f}"

    final_message += f"\n\n=== 分阶段训练耗时统计 ==="
    final_message += f"\n阶段1 ({phase1_episodes} episodes): {phase_times['phase1']:.1f}秒 (平均 {phase_times['phase1']/phase1_episodes:.2f}秒/回合)"
    if phase2_episodes > 0:
        final_message += f"\n阶段2 ({phase2_episodes} episodes): {phase_times['phase2']:.1f}秒 (平均 {phase_times['phase2']/phase2_episodes:.2f}秒/回合)"
    if phase3_episodes > 0:
        final_message += f"\n阶段3 ({phase3_episodes} episodes): {phase_times['phase3']:.1f}秒 (平均 {phase_times['phase3']/phase3_episodes:.2f}秒/回合)"
    final_message += (
        f"\n总训练时间: {total_training_time:.1f}秒 ({total_training_time/60:.1f}分钟)"
    )
    final_message += f"\n总体平均: {total_training_time/total_episodes:.2f}秒/回合"

    print(final_message)
    if data_manager:
        data_manager.log_message(final_message)

    # 详细统计
    if data_manager:
        summary = data_manager.get_summary_stats()
        detail_message = f"\n=== 详细训练统计 ==="
        if "average_steps" in summary:
            detail_message += f"\n平均步数: {summary['average_steps']:.1f} ± {summary['steps_std']:.1f}"
            detail_message += (
                f"\n步数范围: {summary['min_steps']} - {summary['max_steps']}"
            )
        if "average_episode_time" in summary:
            detail_message += f"\n平均每回合时间: {summary['average_episode_time']:.3f} ± {summary['episode_time_std']:.3f} 秒"
        detail_message += f"\n总胜率: {summary['final_win_rate']:.3f}"
        detail_message += (
            f"\n平均奖励: {summary['average_reward']:.3f} ± {summary['reward_std']:.3f}"
        )

        print(detail_message)
        data_manager.log_message(detail_message)

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

        if data_manager and (episode + 1) % 100 == 0:
            # 获取训练历史数据
            history = data_manager.current_session["training_history"]

            # 获取最近100轮的数据
            recent_wins = (
                trainer.batch_wins
                if episode < 100
                else sum(history["wins_game"][-100:])
            )
            recent_losses = (
                trainer.batch_losses
                if episode < 100
                else sum(history["losses_game"][-100:])
            )
            recent_draws = (
                trainer.batch_draws
                if episode < 100
                else sum(history["draws_game"][-100:])
            )
            recent_win_rate = (recent_wins + 0.5 * recent_draws) / 100

            # 获取最近100轮的平均步长
            if batch_steps:
                recent_steps = (
                    batch_steps[-100:] if len(batch_steps) >= 100 else batch_steps
                )
                recent_avg_steps = sum(recent_steps) / len(recent_steps)
            else:
                recent_avg_steps = 0.0

            # 使用新方法记录每百轮统计
            data_manager.log_hundred_stats(
                recent_wins,
                recent_losses,
                recent_draws,
                recent_win_rate,
                recent_avg_steps,
            )

        # 定期输出进度 - 使用trainer的批次统计
        if episode % print_interval == 0 or episode == num_episodes - 1:
            batch_win_rate = trainer.get_batch_win_rate()
            batch_avg_reward = trainer.get_batch_avg_reward()
            recent_win_rate = trainer.get_recent_win_rate(100)

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

            # 显示批次胜率和近期(100轮)胜率
            win_info = f"批次胜率 = {batch_win_rate:.3f}"
            if recent_win_rate is not None:
                win_info += f", 近期胜率 = {recent_win_rate:.3f}"

            progress_message = (
                f"{phase_name.upper()} - 回合 {episode}: 奖励 = {total_reward:.2f}, 步数 = {steps}, "
                f"胜 = {trainer.batch_wins}, 负 = {trainer.batch_losses}, 平 = {trainer.batch_draws}, "
                f"{win_info}, 平均奖励 = {batch_avg_reward:.3f}, "
                f"平均步长 = {avg_steps:.1f}{param_info}{agent_info}{time_info}"
            )

            # 同时输出到控制台和日志文件
            print(progress_message)
            if data_manager:
                data_manager.log_message(progress_message)

            # 记录批次统计到数据管理器 - 传递平均步长
            # if data_manager:
            #     data_manager.log_batch_stats(
            #         trainer.batch_wins,
            #         trainer.batch_losses,
            #         trainer.batch_draws,
            #         batch_win_rate,
            #         avg_steps,  # 使用计算出的平均步长
            #     )

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


def get_agent_imports(agent_type: str):
    """动态导入智能体类"""
    if agent_type == "dqn":
        from DQN import DQNAgent, DQNTrainer

        return DQNAgent, DQNTrainer
    elif agent_type == "cnn-dqn":
        from DQN_CNN import CNNEnhancedDQNAgent, CNNDQNTrainer

        return CNNEnhancedDQNAgent, CNNDQNTrainer
    elif agent_type == "aq":
        from ApproximateQAgent import ApproximateQAgent, ApproximateQTrainer

        return ApproximateQAgent, ApproximateQTrainer
    elif agent_type == "ql":  # 新增QL导入
        from QlearningAgent import QLearningAgent, QLearningTrainer

        return QLearningAgent, QLearningTrainer
    else:
        raise ValueError(f"不支持的智能体类型: {agent_type}")


def apply_preset_to_agent_config(preset, agent_config):
    """根据预设配置更新智能体配置"""
    # 基础参数更新
    if "epsilon" in preset:
        agent_config["epsilon"] = preset["epsilon"]
    if "epsilon_min" in preset:
        agent_config["epsilon_min"] = preset["epsilon_min"]
    if "epsilon_decay" in preset:
        agent_config["epsilon_decay"] = preset["epsilon_decay"]
    if "learning_rate" in preset:
        agent_config["learning_rate"] = preset["learning_rate"]

    # 更新探索策略
    if "exploration_strategy" in preset:
        agent_config["exploration_strategy"] = preset["exploration_strategy"]
    elif "exploration" in preset:
        agent_config["exploration_strategy"] = preset["exploration"]

    return agent_config


def apply_preset_to_agent(agent, preset, phase_name):
    """根据预设配置和阶段名称更新智能体参数"""
    if "phase_config" in preset and phase_name in preset["phase_config"]:
        phase_preset = preset["phase_config"][phase_name]

        # 更新epsilon
        if hasattr(agent, "epsilon") and "epsilon_start" in phase_preset:
            agent.epsilon = phase_preset["epsilon_start"]
            print(f"设置 {phase_name} 初始epsilon: {agent.epsilon}")

        # 更新epsilon_min
        if hasattr(agent, "epsilon_min") and "epsilon_end" in phase_preset:
            agent.epsilon_min = phase_preset["epsilon_end"]
            print(f"设置 {phase_name} 最终epsilon: {agent.epsilon_min}")

        # 更新学习率
        if hasattr(agent, "learning_rate") and "learning_rate" in phase_preset:
            agent.learning_rate = phase_preset["learning_rate"]
            print(f"设置 {phase_name} 学习率: {agent.learning_rate}")
        elif (
            hasattr(agent, "get_learning_rate")
            and hasattr(agent, "set_learning_rate")
            and "learning_rate" in phase_preset
        ):
            agent.set_learning_rate(phase_preset["learning_rate"])
            print(f"设置 {phase_name} 学习率: {agent.get_learning_rate()}")

        # 更新学习率范围
        if hasattr(agent, "lr_min") and "learning_rate_min" in phase_preset:
            agent.lr_min = phase_preset["learning_rate_min"]
            print(f"设置 {phase_name} 最小学习率: {agent.lr_min}")
        if hasattr(agent, "lr_max") and "learning_rate_max" in phase_preset:
            agent.lr_max = phase_preset["learning_rate_max"]
            print(f"设置 {phase_name} 最大学习率: {agent.lr_max}")

        # 新增：更新探索策略
        if (
            hasattr(agent, "set_exploration_strategy")
            and "exploration_strategy" in phase_preset
        ):
            agent.set_exploration_strategy(phase_preset["exploration_strategy"])

        # 更新phase_configs（如果存在）
        if hasattr(agent, "phase_configs") and phase_name in agent.phase_configs:
            if "epsilon_force_until" in phase_preset:
                agent.phase_configs[phase_name]["epsilon_force_until"] = phase_preset[
                    "epsilon_force_until"
                ]
            if "epsilon_end" in phase_preset:
                agent.phase_configs[phase_name]["epsilon_min"] = phase_preset[
                    "epsilon_end"
                ]

    return agent


def train_with_curriculum_from_preset(
    agent: Player,
    opponent: Player,
    trainer_class: Type[BaseTrainer],
    preset: Dict[str, Any] = None,
    data_manager: TrainingDataManager = None,
):
    """从预设配置获取训练参数的课程训练函数"""
    if not preset:
        print("未提供预设配置，使用默认参数训练")
        return train_with_curriculum_generic(
            agent,
            opponent,
            trainer_class,
            episodes=2000,
            data_manager=data_manager,
            print_interval=50,
            opponent_type="progressive",
        )

    # 获取预设参数
    total_episodes = preset.get("episodes", 2000)
    print_interval = preset.get("print_interval", 50)
    opponent_type = preset.get("opponent_type", "progressive")
    phase_config = preset.get("phase_config", {})

    agent_type = agent.__class__.__name__
    training_start_time = time.time()
    phase_times = {}

    start_message = f"开始使用预设'{preset['name']}'进行{agent_type}分阶段课程式训练"
    start_message += f"\n总回合数: {total_episodes}, 对手策略: {opponent_type}"
    print(start_message)
    if data_manager:
        data_manager.log_message(start_message)

    # 显示分阶段配置
    config_message = f"\n=== 预设训练配置细节 ==="
    for phase, config in phase_config.items():
        config_message += f"\n{phase}:"
        config_message += f"\n  比例: {config['ratio']:.0%}"
        config_message += f"\n  对手: {config['opponent']}"
        config_message += (
            f"\n  Epsilon: {config['epsilon_start']:.2f} → {config['epsilon_end']:.2f}"
        )
        config_message += f"\n  强制期: {config['epsilon_force_until']}回合"
        config_message += f"\n  学习率: {config['learning_rate']:.6f} [{config['learning_rate_min']:.6f} - {config['learning_rate_max']:.6f}]"

    print(config_message)
    if data_manager:
        data_manager.log_message(config_message)

    # 阶段1: 基础学习
    phase1_ratio = phase_config.get("phase1", {}).get("ratio", 0.4)
    phase1_episodes = int(total_episodes * phase1_ratio)
    phase1_opponent_type = phase_config.get("phase1", {}).get("opponent", "random")

    if data_manager:
        data_manager.log_phase_start(
            "阶段1: 基础学习", phase1_episodes, phase1_opponent_type
        )

    apply_preset_to_agent(agent, preset, "phase1")
    phase1_opponent = create_opponent(phase1_opponent_type, 1 - agent.player_id)
    trainer = trainer_class(agent, phase1_opponent)

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

    if data_manager:
        data_manager.log_phase_end(
            "阶段1",
            phase1_stats,
            {
                "total_time": phase_times["phase1"],
                "avg_time": phase_times["phase1"] / phase1_episodes,
            },
        )

    # 阶段2: 进阶学习
    phase2_ratio = phase_config.get("phase2", {}).get("ratio", 0.4)
    phase2_episodes = int(total_episodes * phase2_ratio)
    phase2_opponent_type = phase_config.get("phase2", {}).get("opponent", "greedy")

    if phase2_episodes > 0:  # 只有当阶段2有回合时才执行
        if data_manager:
            data_manager.log_phase_start(
                "阶段2: 进阶学习", phase2_episodes, phase2_opponent_type
            )

        print(
            f"\n阶段2: 进阶学习 ({phase2_episodes} episodes) - 对手: {phase2_opponent_type}"
        )

        # 应用阶段2配置
        apply_preset_to_agent(agent, preset, "phase2")

        phase2_opponent = create_opponent(phase2_opponent_type, 1 - agent.player_id)
        trainer.opponent = phase2_opponent

        # 开始阶段2训练
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

        if data_manager:
            data_manager.log_phase_end(
                "阶段2",
                phase2_stats,
                {
                    "total_time": phase_times["phase2"],
                    "avg_time": phase_times["phase2"] / phase2_episodes,
                },
            )

        completion_message = (
            f"\n阶段2完成! 当前状态:{get_agent_hyperparams_info(agent)}"
        )
        completion_message += f"\n阶段2胜率: {phase2_stats['win_rate']:.3f}, 平均奖励: {phase2_stats['avg_reward']:.3f}, "
        completion_message += f"平均步长: {phase2_stats.get('avg_steps', 0):.1f}"
        completion_message += f"\n阶段2总耗时: {phase_times['phase2']:.1f}秒, 平均: {phase_times['phase2']/phase2_episodes:.2f}秒/回合"

        print(completion_message)
        if data_manager:
            data_manager.log_message(completion_message)
    else:
        phase2_stats = {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "win_rate": 0.0,
            "avg_reward": 0.0,
            "avg_steps": 0.0,
        }
        phase_times["phase2"] = 0.0
        print("\n阶段2: 跳过 (比例设置为0)")
        if data_manager:
            data_manager.log_message("阶段2: 跳过 (比例设置为0)")

    # 阶段3: 策略精炼
    phase3_ratio = phase_config.get("phase3", {}).get("ratio", 0.2)
    phase3_episodes = int(total_episodes * phase3_ratio)
    if phase3_episodes == 0:
        phase3_episodes = (
            total_episodes - phase1_episodes - phase2_episodes
        )  # 确保总数正确
    phase3_opponent_type = phase_config.get("phase3", {}).get("opponent", "minimax")

    if phase3_episodes > 0:  # 只有当阶段3有回合时才执行
        if data_manager:
            data_manager.log_phase_start(
                "阶段3: 策略精炼", phase3_episodes, phase3_opponent_type
            )

        print(
            f"\n阶段3: 策略精炼 ({phase3_episodes} episodes) - 对手: {phase3_opponent_type}"
        )

        # 应用阶段3配置
        apply_preset_to_agent(agent, preset, "phase3")

        phase3_opponent = create_opponent(phase3_opponent_type, 1 - agent.player_id)
        trainer.opponent = phase3_opponent

        # 开始阶段3训练
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

        if data_manager:
            data_manager.log_phase_end(
                "阶段3",
                phase3_stats,
                {
                    "total_time": phase_times["phase3"],
                    "avg_time": phase_times["phase3"] / phase3_episodes,
                },
            )

        completion_message = (
            f"\n阶段3完成! 当前状态:{get_agent_hyperparams_info(agent)}"
        )
        completion_message += f"\n阶段3胜率: {phase3_stats['win_rate']:.3f}, 平均奖励: {phase3_stats['avg_reward']:.3f}, "
        completion_message += f"平均步长: {phase3_stats.get('avg_steps', 0):.1f}"
        completion_message += f"\n阶段3总耗时: {phase_times['phase3']:.1f}秒, 平均: {phase_times['phase3']/phase3_episodes:.2f}秒/回合"

        print(completion_message)
        if data_manager:
            data_manager.log_message(completion_message)
    else:
        phase3_stats = {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "win_rate": 0.0,
            "avg_reward": 0.0,
            "avg_steps": 0.0,
        }
        phase_times["phase3"] = 0.0
        print("\n阶段3: 跳过 (比例设置为0)")
        if data_manager:
            data_manager.log_message("阶段3: 跳过 (比例设置为0)")

    # 训练完成后的最终处理
    if hasattr(trainer, "finalize_training"):
        trainer.finalize_training()
    else:
        # 后备方案：直接更新目标网络
        if hasattr(agent, "update_target_network"):
            agent.update_target_network()
            print("训练完成，执行最终目标网络更新")

        # 设置为测试模式
        if hasattr(agent, "set_training_mode"):
            agent.set_training_mode(False)

    # 总结训练时间
    total_training_time = time.time() - training_start_time

    # 总体统计
    total_wins = phase1_stats["wins"] + phase2_stats["wins"] + phase3_stats["wins"]
    total_losses = (
        phase1_stats["losses"] + phase2_stats["losses"] + phase3_stats["losses"]
    )
    total_draws = phase1_stats["draws"] + phase2_stats["draws"] + phase3_stats["draws"]
    total_games = total_wins + total_losses + total_draws
    overall_win_rate = (
        (total_wins + 0.5 * total_draws) / total_games if total_games > 0 else 0.0
    )

    final_summary_message = f"\n{'='*60}"
    final_summary_message += f"\n{agent_type} 预设'{preset['name']}'分阶段课程训练完成!"
    final_summary_message += f"\n最终状态:{get_agent_hyperparams_info(agent)}"
    final_summary_message += (
        f"\n总训练时间: {total_training_time:.1f}秒 ({total_training_time/60:.1f}分钟)"
    )
    final_summary_message += f"\n总体胜率: {overall_win_rate:.3f} (胜:{total_wins}, 负:{total_losses}, 平:{total_draws})"

    # 分阶段时间统计
    final_summary_message += f"\n\n=== 分阶段训练耗时统计 ==="
    if phase1_episodes > 0:
        final_summary_message += f"\n阶段1 ({phase1_episodes} episodes): {phase_times['phase1']:.1f}秒 (平均 {phase_times['phase1']/phase1_episodes:.2f}秒/回合)"
    if phase2_episodes > 0:
        final_summary_message += f"\n阶段2 ({phase2_episodes} episodes): {phase_times['phase2']:.1f}秒 (平均 {phase_times['phase2']/phase2_episodes:.2f}秒/回合)"
    if phase3_episodes > 0:
        final_summary_message += f"\n阶段3 ({phase3_episodes} episodes): {phase_times['phase3']:.1f}秒 (平均 {phase_times['phase3']/phase3_episodes:.2f}秒/回合)"

    final_summary_message += (
        f"\n总体平均: {total_training_time/total_episodes:.2f}秒/回合"
    )
    final_summary_message += f"\n{'='*60}"

    print(final_summary_message)
    if data_manager:
        data_manager.log_message(final_summary_message)

    return data_manager.current_session["training_history"] if data_manager else {}


# 修改main函数
def main():
    """主函数 - 支持命令行参数和预设配置"""
    parser = argparse.ArgumentParser(description="通用智能体训练和测试系统")
    parser.add_argument(
        "--agent",
        choices=["dqn", "cnn-dqn", "aq", "ql"],
        help="选择智能体类型: dqn, cnn-dqn, aq, ql",
    )
    parser.add_argument("--retrain", action="store_true", help="强制重新训练模型")
    parser.add_argument("--episodes", type=int, help="训练回合数")
    parser.add_argument("--test-games", type=int, default=100, help="测试游戏数量")
    parser.add_argument("--no-display", action="store_true", help="不显示游戏界面")
    parser.add_argument(
        "--lr-strategy", choices=["adaptive", "fixed", "hybrid"], help="学习率调整策略"
    )
    parser.add_argument("--test-only", action="store_true", help="仅测试，不训练")
    parser.add_argument("--print-interval", type=int, help="训练进度输出间隔")
    parser.add_argument(
        "--opponent",
        choices=["random", "greedy", "minimax", "mixed", "progressive", "random_only"],  # 添加random_only
        help="训练对手类型",
    )
    parser.add_argument("--exploration", choices=["random", "guided"], help="探索策略")
    parser.add_argument(
        "--cnn-model",
        type=str,
        default=None,
        help="预训练CNN模型路径（仅对CNN-DQN有效）",
    )
    parser.add_argument(
        "--use-preset", action="store_true", help="使用预设配置而非命令行参数"
    )
    parser.add_argument("--preset-id", type=int, default=None, help="预设配置ID (0-3)")  # 改为None

    args = parser.parse_args()

    # ===== 以下是手动配置部分，可以在此修改 =====
    # 选择预设ID（0-4）
    PRESET_ID = 4  # 您想要的预设ID

    # 是否使用预设配置（True）或命令行参数（False）
    USE_PRESET = True

    # 是否强制重新训练
    FORCE_RETRAIN = True

    # 测试时是否显示游戏界面
    DISPLAY_GAME = False

    # 测试游戏数量
    TEST_GAMES = 100
    # =======================================

    # 确定使用命令行参数还是预设配置 - 修复优先级逻辑
    use_preset = args.use_preset if args.use_preset else USE_PRESET
    preset_id = args.preset_id if args.preset_id is not None else PRESET_ID
    force_retrain = args.retrain if args.retrain else FORCE_RETRAIN
    display_game = not args.no_display if args.no_display else not DISPLAY_GAME
    test_games = args.test_games if args.test_games else TEST_GAMES

    if use_preset:
        # 使用预设配置
        if preset_id < 0 or preset_id >= len(TRAINING_PRESETS):
            print(f"预设ID {preset_id} 无效，使用默认预设 (ID=0)")
            preset_id = 0

        preset = TRAINING_PRESETS[preset_id].copy()

        print(f"\n使用预设配置: {preset['name']}")
        print(f"预设ID: {preset_id}")  # 添加调试信息
        print(f"总回合数: {preset.get('episodes', 2000)}")
        print(f"对手类型: {preset.get('opponent_type', 'progressive')}")

        agent_type = preset["agent_type"]
    else:
        # 使用命令行参数
        if not args.agent:
            print("未指定智能体类型，需要使用 --agent 参数")
            return
        agent_type = args.agent

    # 获取智能体配置
    if agent_type not in AGENT_CONFIGS:
        print(f"不支持的智能体类型: {agent_type}")
        return

    config = AGENT_CONFIGS[agent_type].copy()
    agent_class, trainer_class = get_agent_imports(agent_type)
    config["agent_class"] = agent_class
    config["trainer_class"] = trainer_class

    # 使用预设配置或命令行参数更新配置
    if use_preset:
        # 更新智能体配置
        agent_config = config["agent_config"]
        agent_config = apply_preset_to_agent_config(preset, agent_config)
        config["agent_config"] = agent_config

        # 如果是CNN-DQN，添加CNN模型路径
        if agent_type == "cnn-dqn" and args.cnn_model:
            config["agent_config"]["cnn_model_path"] = args.cnn_model
    else:
        # 使用命令行参数
        if agent_type == "cnn-dqn" and args.cnn_model:
            config["agent_config"]["cnn_model_path"] = args.cnn_model

        if args.exploration and "exploration_strategy" in config["agent_config"]:
            config["agent_config"]["exploration_strategy"] = args.exploration

    # 创建智能体实例
    agent = config["agent_class"](**config["agent_config"])

    # 设置学习率策略
    lr_strategy = (
        preset.get("lr_strategy", "adaptive") if use_preset else args.lr_strategy
    )
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
    model_path = os.path.join("model_data", f"{config['model_name']}.pkl")
    model_exists = os.path.exists(model_path)

    if model_exists and not force_retrain:
        print(f"发现已训练的模型: {model_path}")
        if agent.load_model(config["model_name"]):
            print("模型加载成功!")
            agent.epsilon = 0.0
            agent.ai_type = f"{agent.__class__.__name__} (Trained)"
        else:
            print("模型加载失败，将重新训练...")
            model_exists = False

    initial_opponent = create_opponent("random", 1 - agent.player_id)

    if (not model_exists or force_retrain) and not args.test_only:
        if force_retrain:
            print("强制重新训练模型...")
        else:
            print("未找到已训练模型，开始训练...")

        # 创建数据管理器
        data_manager = TrainingDataManager()
        if use_preset:
            session_name = f"{agent.__class__.__name__}_{preset['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            exploration = args.exploration if args.exploration else "guided"
            opponent_type = args.opponent if args.opponent else "progressive"
            episodes = args.episodes if args.episodes else 2000
            session_name = f"{agent.__class__.__name__}_{opponent_type}_{exploration}_{episodes}eps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        data_manager.start_session(agent, session_name)

        # 使用预设配置或通用课程式训练
        if use_preset:
            combined_history = train_with_curriculum_from_preset(
                agent, initial_opponent, trainer_class, preset, data_manager
            )
        else:
            opponent_type = args.opponent if args.opponent else "progressive"
            episodes = args.episodes if args.episodes else 2000
            print_interval = args.print_interval if args.print_interval else 50

            combined_history = train_with_curriculum_generic(
                agent,
                initial_opponent,
                trainer_class,
                episodes,
                data_manager,
                print_interval,
                opponent_type,
                **config["trainer_kwargs"],
            )

        # 结束数据记录会话
        if use_preset:
            final_stats = {
                "training_episodes": preset.get("episodes", 2000),
                "lr_strategy": lr_strategy,
                "opponent_type": preset.get("opponent_type", "progressive"),
                "exploration_strategy": preset.get("exploration", "guided"),
                "final_epsilon": getattr(agent, "epsilon", None),
                "final_learning_rate": (
                    agent.get_learning_rate()
                    if hasattr(agent, "get_learning_rate")
                    else getattr(agent, "learning_rate", None)
                ),
                "final_discount_factor": getattr(agent, "discount_factor", None),
            }
        else:
            final_stats = {
                "training_episodes": args.episodes if args.episodes else 2000,
                "lr_strategy": lr_strategy if lr_strategy else "adaptive",
                "opponent_type": args.opponent if args.opponent else "progressive",
                "exploration_strategy": (
                    args.exploration if args.exploration else "guided"
                ),
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
        agent.save_model(config["model_name"])

        print(f"训练完成! 最终epsilon: {getattr(agent, 'epsilon', 'N/A')}")
        if hasattr(agent, "get_stats"):
            print(f"最终胜率: {agent.get_stats()['win_rate']:.3f}")

        # 设置为测试模式
        agent.epsilon = 0.0
        agent.ai_type = f"{agent.__class__.__name__} (Trained)"
    elif args.test_only:
        # 仅测试模式：直接加载模型
        print("仅测试模式，加载已训练模型...")
        agent_config = config["agent_config"].copy()
        agent = config["agent_class"](**agent_config)
        if not agent.load_model(config["model_name"]):
            print("无法加载模型，请先训练!")
            return

    # 测试
    test_opponent = create_opponent("random", 1 - agent.player_id)
    test_results = test_agent_generic(
        agent,
        test_opponent,
        num_games=test_games,
        show_individual=(test_games <= 10),
        display=not display_game,
    )

    print(f"\n{agent.__class__.__name__} 测试完成!")


if __name__ == "__main__":
    main()

# 直接运行main，或者命令行输入都可以，main的话，需要调整 往上数 200 行左右，以及最上面的预设

# 命令行运行示例:
# 使用引导探索的渐进式训练（推荐）

# python train_and_record.py --agent dqn --retrain --episodes 3000 --opponent progressive --exploration guided --lr-strategy hybrid --test-games 100 --no-display --print-interval 100

# python train_and_record.py --agent cnn-dqn --retrain --episodes 3000 --opponent progressive --exploration guided --lr-strategy adaptive --test-games 100 --no-display --print-interval 100

# ython train_and_record.py --agent aq --retrain --episodes 3000 --opponent progressive --lr-strategy adaptive --test-games 100 --no-display --print-interval 100

# python train_and_record.py --agent ql --retrain --episodes 3000 --opponent progressive --lr-strategy adaptive --test-games 100 --no-display --print-interval 100

# 使用随机探索对比
# python train_and_record.py --agent dqn --retrain --episodes 3000 --opponent progressive --exploration random --lr-strategy hybrid --test-games 100 --no-display --print-interval 50

# 与贪心AI训练，使用引导探索
# python train_and_record.py --agent dqn --retrain --episodes 3000 --opponent greedy --exploration guided --lr-strategy adaptive --test-games 100 --no-display --print-interval 50

# 与Minimax AI训练，使用引导探索
# python train_and_record.py --agent dqn --retrain --episodes 3000 --opponent minimax --exploration guided --lr-strategy adaptive --test-games 100 --no-display --print-interval 50

# 仅测试
# python train_and_record.py --agent dqn --test-only --test-games 50 --no-display

# python train_and_record.py --agent dqn --test-only --test-games 2
