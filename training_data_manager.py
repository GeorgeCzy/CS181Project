import json
import pickle
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端以便于保存图像


class TrainingDataManager:
    """训练数据管理器 - 独立于模型的训练历史记录"""

    def __init__(self, save_dir: str = "training_logs/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.current_session = {
            "start_time": datetime.now().isoformat(),
            "agent_type": "",
            "hyperparameters": {},
            "training_history": {
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
                "losses": [],
                "phases": [],  # 训练阶段标记
                "timestamps": [],  # 时间戳
                "steps": [],  # 新增：每个episode的步数
                "episode_times": [],  # 新增：每个episode的耗时
                "batch_wins": [],  # 新增：批次胜利数
                "batch_losses": [],  # 新增：批次失败数
                "batch_draws": [],  # 新增：批次平局数
                "batch_win_rates": [],  # 新增：批次胜率
                "average_steps": [],  # 新增：批次平均步长
            },
            "final_stats": {},
        }

    def start_session(self, agent, session_name: str = None):
        """开始新的训练会话"""
        if session_name is None:
            session_name = (
                f"{agent.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        self.session_name = session_name
        self.current_session["agent_type"] = agent.__class__.__name__
        self.current_session["start_time"] = datetime.now().isoformat()

        # 保存超参数
        if hasattr(agent, "learning_rate"):
            self.current_session["hyperparameters"][
                "learning_rate"
            ] = agent.learning_rate
        if hasattr(agent, "epsilon"):
            self.current_session["hyperparameters"]["initial_epsilon"] = agent.epsilon
        if hasattr(agent, "epsilon_decay"):
            self.current_session["hyperparameters"][
                "epsilon_decay"
            ] = agent.epsilon_decay
        if hasattr(agent, "epsilon_min"):
            self.current_session["hyperparameters"]["epsilon_min"] = agent.epsilon_min
        if hasattr(agent, "batch_size"):
            self.current_session["hyperparameters"]["batch_size"] = agent.batch_size
        if hasattr(agent, "use_dueling"):
            self.current_session["hyperparameters"]["use_dueling"] = agent.use_dueling
        if hasattr(agent, "use_double"):
            self.current_session["hyperparameters"]["use_double"] = agent.use_double

        print(f"训练会话开始: {session_name}")

    def log_episode(
        self,
        episode: int,
        reward: float,
        result: int,
        learning_rate: float = None,
        epsilon: float = None,
        loss: float = None,
        phase: str = None,
        steps: int = None,
        episode_time: float = None,
        discount_factor: float = None,  # 新增：折扣因子
        q_value: float = None,          # 新增：Q值
    ):
        """记录单个episode的数据 - 扩展版本，支持AQ特有数据"""
        history = self.current_session["training_history"]

        history["episodes"].append(episode)
        history["rewards"].append(reward)
        history["wins"].append(1 if result == 0 else 0)  # 假设agent是player 0
        history["losses"].append(1 if result == 1 else 0)
        history["draws"].append(1 if result == 2 else 0)
        history["timestamps"].append(datetime.now().isoformat())

        # 新增字段
        if steps is not None:
            history["steps"].append(steps)
        if episode_time is not None:
            history["episode_times"].append(episode_time)

        if learning_rate is not None:
            history["learning_rates"].append(learning_rate)
        if epsilon is not None:
            history["epsilons"].append(epsilon)
        if loss is not None:
            history["losses"].append(loss)
        if discount_factor is not None:
            history["discount_factors"].append(discount_factor)
        if phase is not None:
            history["phases"].append(phase)
        else:
            history["phases"].append("training")

        # 计算累积统计
        total_games = len(history["episodes"])
        wins = sum(history["wins"])
        draws = sum(history["draws"])

        if total_games > 0:
            win_rate = (wins + 0.5 * draws) / total_games
            avg_reward = np.mean(history["rewards"])
        else:
            win_rate = 0.0
            avg_reward = 0.0

        history["win_rates"].append(win_rate)
        history["average_rewards"].append(avg_reward)

    def log_batch_stats(
        self,
        batch_wins: int,
        batch_losses: int,
        batch_draws: int,
        batch_win_rate: float,
        average_steps: float,
    ):
        """记录批次统计数据"""
        history = self.current_session["training_history"]

        history["batch_wins"].append(batch_wins)
        history["batch_losses"].append(batch_losses)
        history["batch_draws"].append(batch_draws)
        history["batch_win_rates"].append(batch_win_rate)
        history["average_steps"].append(average_steps)

    def update_from_trainer_history(self, trainer_history: Dict):
        """从训练器历史更新数据"""
        for key, values in trainer_history.items():
            if key in self.current_session["training_history"] and values:
                self.current_session["training_history"][key] = values

    def end_session(self, agent, final_stats: Dict = None):
        """结束训练会话"""
        self.current_session["end_time"] = datetime.now().isoformat()

        # 保存最终统计
        if hasattr(agent, "get_stats"):
            self.current_session["final_stats"] = agent.get_stats()

        if final_stats:
            self.current_session["final_stats"].update(final_stats)

        # 保存最终超参数状态
        final_params = {}
        if hasattr(agent, "epsilon"):
            final_params["final_epsilon"] = agent.epsilon
        if hasattr(agent, "get_learning_rate"):
            final_params["final_learning_rate"] = agent.get_learning_rate()
        elif hasattr(agent, "learning_rate"):
            final_params["final_learning_rate"] = agent.learning_rate

        self.current_session["final_hyperparameters"] = final_params

        # 保存到文件
        self.save_session()
        print(f"训练会话结束: {self.session_name}")

    def save_session(self, filename: str = None):
        """保存当前会话到文件"""
        if filename is None:
            filename = f"{self.session_name}_training_log"

        # 保存为JSON格式（可读性好）
        json_path = os.path.join(self.save_dir, f"{filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.current_session, f, indent=2, ensure_ascii=False)

        # 保存为pickle格式（包含numpy数组）
        pickle_path = os.path.join(self.save_dir, f"{filename}.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.current_session, f)

        print(f"训练数据已保存:")
        print(f"  JSON: {json_path}")
        print(f"  Pickle: {pickle_path}")

    def load_session(self, filename: str) -> Dict:
        """加载训练会话"""
        pickle_path = os.path.join(self.save_dir, f"{filename}.pkl")
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                return pickle.load(f)

        json_path = os.path.join(self.save_dir, f"{filename}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)

        return None

    def plot_training_history(self, save_plots: bool = True):
        """绘制扩展的训练历史 - 包含步长和时间分析"""
        history = self.current_session["training_history"]

        if not history["episodes"]:
            print("没有训练数据可绘制")
            return

        # 创建更大的图形布局：3行4列
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        fig.suptitle(f"Training History - {self.session_name}", fontsize=16)

        episodes = history["episodes"]
        total_episodes = len(episodes)

        # 第一行 - 基础性能指标
        # 1. 胜率变化
        if history["win_rates"]:
            win_rates = history["win_rates"][:total_episodes]
            axes[0, 0].plot(episodes[: len(win_rates)], win_rates, "b-", alpha=0.7)
            axes[0, 0].set_title("Win Rate Over Time")
            axes[0, 0].set_xlabel("Episodes")
            axes[0, 0].set_ylabel("Win Rate")
            axes[0, 0].grid(True)

        # 2. 奖励变化
        if history["rewards"]:
            rewards = history["rewards"][:total_episodes]
            axes[0, 1].plot(episodes[: len(rewards)], rewards, "g-", alpha=0.7)
            # 添加移动平均线
            if len(rewards) > 10:
                window = min(50, len(rewards) // 10)
                if window > 0:
                    ma_rewards = np.convolve(
                        rewards, np.ones(window) / window, mode="valid"
                    )
                    ma_episodes = episodes[window - 1 : len(rewards)]
                    if len(ma_episodes) == len(ma_rewards):
                        axes[0, 1].plot(
                            ma_episodes,
                            ma_rewards,
                            "r-",
                            linewidth=2,
                            label=f"MA({window})",
                        )
                        axes[0, 1].legend()
            axes[0, 1].set_title("Reward Over Time")
            axes[0, 1].set_xlabel("Episodes")
            axes[0, 1].set_ylabel("Reward")
            axes[0, 1].grid(True)

        # 3. Epsilon变化
        if history["epsilons"]:
            epsilons = history["epsilons"][:total_episodes]
            axes[0, 2].plot(episodes[: len(epsilons)], epsilons, "r-", alpha=0.7)
            axes[0, 2].set_title("Epsilon Decay")
            axes[0, 2].set_xlabel("Episodes")
            axes[0, 2].set_ylabel("Epsilon")
            axes[0, 2].grid(True)

        # 4. 学习率变化
        if history["learning_rates"]:
            learning_rates = [lr for lr in history["learning_rates"] if lr is not None]
            if learning_rates:
                learning_rates = learning_rates[:total_episodes]
                axes[0, 3].plot(
                    episodes[: len(learning_rates)], learning_rates, "m-", alpha=0.7
                )
                axes[0, 3].set_title("Learning Rate Over Time")
                axes[0, 3].set_xlabel("Episodes")
                axes[0, 3].set_ylabel("Learning Rate")
                axes[0, 3].grid(True)

        # 第二行 - 新增的步长和时间分析
        # 5. 每回合步数变化
        if history["steps"]:
            steps = history["steps"][:total_episodes]
            axes[1, 0].plot(episodes[: len(steps)], steps, "orange", alpha=0.7)
            # 添加移动平均线
            if len(steps) > 10:
                window = min(50, len(steps) // 10)
                if window > 0:
                    ma_steps = np.convolve(
                        steps, np.ones(window) / window, mode="valid"
                    )
                    ma_episodes = episodes[window - 1 : len(steps)]
                    if len(ma_episodes) == len(ma_steps):
                        axes[1, 0].plot(
                            ma_episodes,
                            ma_steps,
                            "red",
                            linewidth=2,
                            label=f"MA({window})",
                        )
                        axes[1, 0].legend()
            axes[1, 0].set_title("Steps per Episode")
            axes[1, 0].set_xlabel("Episodes")
            axes[1, 0].set_ylabel("Steps")
            axes[1, 0].grid(True)

        # 6. 每回合耗时
        if history["episode_times"]:
            episode_times = history["episode_times"][:total_episodes]
            axes[1, 1].plot(
                episodes[: len(episode_times)], episode_times, "purple", alpha=0.7
            )
            # 添加移动平均线
            if len(episode_times) > 10:
                window = min(50, len(episode_times) // 10)
                if window > 0:
                    ma_times = np.convolve(
                        episode_times, np.ones(window) / window, mode="valid"
                    )
                    ma_episodes = episodes[window - 1 : len(episode_times)]
                    if len(ma_episodes) == len(ma_times):
                        axes[1, 1].plot(
                            ma_episodes,
                            ma_times,
                            "red",
                            linewidth=2,
                            label=f"MA({window})",
                        )
                        axes[1, 1].legend()
            axes[1, 1].set_title("Episode Time")
            axes[1, 1].set_xlabel("Episodes")
            axes[1, 1].set_ylabel("Time (seconds)")
            axes[1, 1].grid(True)

        # 7. 批次胜率变化（如果有数据）
        if history["batch_win_rates"]:
            batch_episodes = list(
                range(0, len(history["batch_win_rates"]) * 50, 50)
            )  # 假设每50个episode一个批次
            axes[1, 2].plot(
                batch_episodes,
                history["batch_win_rates"],
                "cyan",
                marker="o",
                alpha=0.7,
            )
            axes[1, 2].set_title("Batch Win Rate")
            axes[1, 2].set_xlabel("Episodes")
            axes[1, 2].set_ylabel("Batch Win Rate")
            axes[1, 2].grid(True)

        # 8. 批次平均步长（如果有数据）
        if history["average_steps"]:
            batch_episodes = list(range(0, len(history["average_steps"]) * 50, 50))
            axes[1, 3].plot(
                batch_episodes, history["average_steps"], "brown", marker="s", alpha=0.7
            )
            axes[1, 3].set_title("Batch Average Steps")
            axes[1, 3].set_xlabel("Episodes")
            axes[1, 3].set_ylabel("Average Steps")
            axes[1, 3].grid(True)

        # 第三行 - 损失和累积统计
        # 9. 损失变化
        if history["losses"]:
            valid_losses = [loss for loss in history["losses"] if loss is not None]
            if valid_losses:
                if len(valid_losses) > total_episodes:
                    loss_x = np.linspace(episodes[0], episodes[-1], len(valid_losses))
                else:
                    loss_x = episodes[: len(valid_losses)]

                axes[2, 0].plot(loss_x, valid_losses, "orange", alpha=0.7)

                if len(valid_losses) > 10:
                    window = min(50, len(valid_losses) // 10)
                    if window > 0:
                        ma_losses = np.convolve(
                            valid_losses, np.ones(window) / window, mode="valid"
                        )
                        ma_x = loss_x[window - 1 : len(valid_losses)]
                        if len(ma_x) == len(ma_losses):
                            axes[2, 0].plot(
                                ma_x,
                                ma_losses,
                                "red",
                                linewidth=2,
                                label=f"MA({window})",
                            )
                            axes[2, 0].legend()

                axes[2, 0].set_title("Training Loss Over Time")
                axes[2, 0].set_xlabel("Episodes")
                axes[2, 0].set_ylabel("Loss")
                axes[2, 0].grid(True)

        # 10. 胜负平累积统计
        wins = history["wins"][:total_episodes]
        losses_data = history["losses"][:total_episodes]
        draws = history["draws"][:total_episodes]

        wins_cum = np.cumsum(wins)
        losses_cum = np.cumsum(losses_data)
        draws_cum = np.cumsum(draws)

        axes[2, 1].plot(
            episodes[: len(wins_cum)], wins_cum, "g-", label="Wins", alpha=0.7
        )
        axes[2, 1].plot(
            episodes[: len(losses_cum)], losses_cum, "r-", label="Losses", alpha=0.7
        )
        axes[2, 1].plot(
            episodes[: len(draws_cum)], draws_cum, "y-", label="Draws", alpha=0.7
        )
        axes[2, 1].set_title("Cumulative Game Results")
        axes[2, 1].set_xlabel("Episodes")
        axes[2, 1].set_ylabel("Count")
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        # 11. 步数分布直方图
        if history["steps"]:
            axes[2, 2].hist(
                history["steps"], bins=30, alpha=0.7, color="skyblue", edgecolor="black"
            )
            axes[2, 2].set_title("Steps Distribution")
            axes[2, 2].set_xlabel("Steps per Episode")
            axes[2, 2].set_ylabel("Frequency")
            axes[2, 2].grid(True)

            # 添加统计信息
            mean_steps = np.mean(history["steps"])
            std_steps = np.std(history["steps"])
            axes[2, 2].axvline(
                mean_steps,
                color="red",
                linestyle="--",
                label=f"Mean: {mean_steps:.1f}±{std_steps:.1f}",
            )
            axes[2, 2].legend()

        # 12. 时间效率分析（如果有数据）
        if history["episode_times"]:
            axes[2, 3].hist(
                history["episode_times"],
                bins=30,
                alpha=0.7,
                color="lightgreen",
                edgecolor="black",
            )
            axes[2, 3].set_title("Episode Time Distribution")
            axes[2, 3].set_xlabel("Time per Episode (seconds)")
            axes[2, 3].set_ylabel("Frequency")
            axes[2, 3].grid(True)

            # 添加统计信息
            mean_time = np.mean(history["episode_times"])
            std_time = np.std(history["episode_times"])
            axes[2, 3].axvline(
                mean_time,
                color="red",
                linestyle="--",
                label=f"Mean: {mean_time:.2f}±{std_time:.2f}s",
            )
            axes[2, 3].legend()

        plt.tight_layout()

        if save_plots:
            plot_path = os.path.join(
                self.save_dir, f"{self.session_name}_training_plots.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"训练图表已保存: {plot_path}")

        plt.close()

    def get_summary_stats(self) -> Dict:
        """获取训练摘要统计 - 包含新增指标"""
        history = self.current_session["training_history"]

        if not history["episodes"]:
            return {}

        summary = {
            "total_episodes": len(history["episodes"]),
            "total_wins": sum(history["wins"]),
            "total_losses": sum(history["losses"]),
            "total_draws": sum(history["draws"]),
            "final_win_rate": history["win_rates"][-1] if history["win_rates"] else 0,
            "average_reward": np.mean(history["rewards"]) if history["rewards"] else 0,
            "reward_std": np.std(history["rewards"]) if history["rewards"] else 0,
        }

        # 新增统计
        if history["steps"]:
            summary["average_steps"] = np.mean(history["steps"])
            summary["steps_std"] = np.std(history["steps"])
            summary["min_steps"] = min(history["steps"])
            summary["max_steps"] = max(history["steps"])

        if history["episode_times"]:
            summary["average_episode_time"] = np.mean(history["episode_times"])
            summary["episode_time_std"] = np.std(history["episode_times"])
            summary["total_training_time"] = sum(history["episode_times"])

        if history["epsilons"]:
            summary["initial_epsilon"] = history["epsilons"][0]
            summary["final_epsilon"] = history["epsilons"][-1]

        if history["learning_rates"]:
            valid_lrs = [lr for lr in history["learning_rates"] if lr is not None]
            if valid_lrs:
                summary["initial_learning_rate"] = valid_lrs[0]
                summary["final_learning_rate"] = valid_lrs[-1]

        return summary
