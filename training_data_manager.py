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
        
        self.log_file = None
        self.log_dir = save_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.current_session = {
            "start_time": datetime.now().isoformat(),
            "agent_type": "",
            "hyperparameters": {},
            "training_history": {
                "episodes": [],
                "rewards": [],
                "wins_game": [],
                "losses_game": [],
                "draws_game": [],
                "win_rates": [],
                "average_rewards": [],
                "learning_rates": [],
                "epsilons": [],
                "discount_factors": [],
                "losses_data": [],
                "phases": [],  # 训练阶段标记
                "timestamps": [],  # 时间戳
                "steps": [],  # 新增：每个episode的步数
                "episode_times": [],  # 新增：每个episode的耗时
                "hundred_wins": [],  # 每一百轮胜利数
                "hundred_losses": [],  # 每一百轮失败数
                "hundred_draws": [],  # 每一百轮平局数
                "hundred_win_rates": [],  # 每一百轮胜率
                "hundred_average_steps": [],  # 每一百轮平均步长
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
        
        log_filename = f"training_log_{session_name}.txt"
        self.log_file = os.path.join(self.log_dir, log_filename)
        
        # 清除或创建新的日志文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== 训练日志 - {session_name} ===\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"智能体类型: {agent.__class__.__name__}\n")
            f.write("=" * 60 + "\n\n")
        
        print(f"训练日志将保存到: {self.log_file}")

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
        discount_factor: float = None,
        q_value: float = None,
    ):
        """记录单个episode的数据 - 更新键名"""
        history = self.current_session["training_history"]

        history["episodes"].append(episode)
        history["rewards"].append(reward)
        history["wins_game"].append(1 if result == 0 else 0)  # 假设agent是player 0
        history["losses_game"].append(1 if result == 1 else 0)
        history["draws_game"].append(1 if result == 2 else 0)
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
            history["losses_data"].append(loss)  # losses保持不变，因为它存储的是网络训练损失
        if discount_factor is not None:
            history["discount_factors"].append(discount_factor)
        if phase is not None:
            history["phases"].append(phase)
        else:
            history["phases"].append("training")

        # 计算累积统计
        total_games = len(history["episodes"])
        wins = sum(history["wins_game"])
        draws = sum(history["draws_game"])

        if total_games > 0:
            win_rate = (wins + 0.5 * draws) / total_games
            avg_reward = np.mean(history["rewards"])
        else:
            win_rate = 0.0
            avg_reward = 0.0

        history["win_rates"].append(win_rate)
        history["average_rewards"].append(avg_reward)
        
        # 每100局更新hundred_统计数据
        if total_games % 100 == 0 and total_games > 0:
            # 提取最近100局的数据
            recent_wins = sum(history["wins_game"][-100:])
            recent_losses = sum(history["losses_game"][-100:])
            recent_draws = sum(history["draws_game"][-100:])
            recent_steps = history["steps"][-100:] if "steps" in history and history["steps"] else []
            
            # 更新统计数据
            history["hundred_wins"].append(recent_wins)
            history["hundred_losses"].append(recent_losses)
            history["hundred_draws"].append(recent_draws)
            history["hundred_win_rates"].append((recent_wins + 0.5 * recent_draws) / 100)
            
            if recent_steps:
                history["hundred_average_steps"].append(sum(recent_steps) / len(recent_steps))

    def log_hundred_stats(
        self,
        hundred_wins: int,
        hundred_losses: int,
        hundred_draws: int,
        hundred_win_rate: float,
        hundred_average_steps: float,
    ):
        """记录每百轮统计数据"""
        history = self.current_session["training_history"]

        history["hundred_wins"].append(hundred_wins)
        history["hundred_losses"].append(hundred_losses)
        history["hundred_draws"].append(hundred_draws)
        history["hundred_win_rates"].append(hundred_win_rate)
        history["hundred_average_steps"].append(hundred_average_steps)

    def update_from_trainer_history(self, trainer_history: Dict):
        """从训练器历史更新数据"""
        for key, values in trainer_history.items():
            if key in self.current_session["training_history"] and values:
                self.current_session["training_history"][key] = values
                
    def log_message(self, message: str):
        """将消息写入日志文件"""
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    f.write(f"[{timestamp}] {message}\n")
            except Exception as e:
                print(f"写入日志文件失败: {e}")
    
    def log_phase_start(self, phase_name: str, episodes: int, opponent_type: str):
        """记录阶段开始信息"""
        message = f"\n{'='*50}\n{phase_name.upper()}: 开始训练 ({episodes} episodes)\n对手类型: {opponent_type}\n{'='*50}"
        self.log_message(message)
        print(message)
    
    def log_phase_end(self, phase_name: str, stats: dict, time_info: dict):
        """记录阶段结束信息"""
        message = f"\n{phase_name.upper()} 完成:\n"
        message += f"  胜率: {stats['win_rate']:.3f}\n"
        message += f"  平均奖励: {stats['avg_reward']:.3f}\n"
        message += f"  平均步长: {stats.get('avg_steps', 0):.1f}\n"
        message += f"  总耗时: {time_info['total_time']:.1f}秒\n"
        if 'avg_time' in time_info and time_info['avg_time'] > 0:
            message += f"  平均时间: {time_info['avg_time']:.2f}秒/回合\n"
        else:
            message += f"  平均时间: 无法计算\n"
        message += "-" * 50
        self.log_message(message)
        print(message)

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
        
        if self.log_file:
            message = f"\n{'='*60}\n训练会话结束\n"
            message += f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            message += f"最终epsilon: {final_stats.get('final_epsilon', 'N/A')}\n"
            message += f"最终学习率: {final_stats.get('final_learning_rate', 'N/A')}\n"
            message += f"训练回合数: {final_stats.get('training_episodes', 'N/A')}\n"
            message += f"对手类型: {final_stats.get('opponent_type', 'N/A')}\n"
            message += f"探索策略: {final_stats.get('exploration_strategy', 'N/A')}\n"
            message += "=" * 60
            self.log_message(message)

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

        # 创建图形布局
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
        # if history["batch_win_rates"]:
        #     batch_episodes = list(
        #         range(0, len(history["batch_win_rates"]) * 50, 50)
        #     )  # 假设每50个episode一个批次
        #     axes[1, 2].plot(
        #         batch_episodes,
        #         history["batch_win_rates"],
        #         "cyan",
        #         marker="o",
        #         alpha=0.7,
        #     )
        #     axes[1, 2].set_title("Batch Win Rate")
        #     axes[1, 2].set_xlabel("Episodes")
        #     axes[1, 2].set_ylabel("Batch Win Rate")
        #     axes[1, 2].grid(True)

        # 8. 批次平均步长（如果有数据）
        # if history["average_steps"]:
        #     batch_episodes = list(range(0, len(history["average_steps"]) * 50, 50))
        #     axes[1, 3].plot(
        #         batch_episodes, history["average_steps"], "brown", marker="s", alpha=0.7
        #     )
        #     axes[1, 3].set_title("Batch Average Steps")
        #     axes[1, 3].set_xlabel("Episodes")
        #     axes[1, 3].set_ylabel("Average Steps")
        #     axes[1, 3].grid(True)
        
        # 7. 改为每100轮胜率变化
        if history["hundred_win_rates"]:
            # 计算对应的回合数
            hundred_episodes = [(i+1)*100 for i in range(len(history["hundred_win_rates"]))]
            
            axes[1, 2].plot(hundred_episodes, history["hundred_win_rates"], "cyan", marker="o", alpha=0.7)
            axes[1, 2].set_title("Win Rate (per 100 episodes)")
            axes[1, 2].set_xlabel("Episodes")
            axes[1, 2].set_ylabel("Win Rate")
            axes[1, 2].grid(True)

        # 8. 改为每100轮平均步长
        if history["hundred_average_steps"]:
            # 计算对应的回合数
            hundred_episodes = [(i+1)*100 for i in range(len(history["hundred_average_steps"]))]
            
            axes[1, 3].plot(hundred_episodes, history["hundred_average_steps"], "brown", marker="s", alpha=0.7)
            axes[1, 3].set_title("Average Steps (per 100 episodes)")
            axes[1, 3].set_xlabel("Episodes")
            axes[1, 3].set_ylabel("Average Steps")
            axes[1, 3].grid(True)

        # 第三行 - 损失和累积统计
        # 9. 损失变化
        if history["losses_data"]:
            valid_losses = [loss for loss in history["losses_data"] if loss is not None]
            if valid_losses:
                if len(valid_losses) > total_episodes:
                    loss_x = np.linspace(episodes[0], episodes[-1], len(valid_losses))
                else:
                    loss_x = episodes[: len(valid_losses)]
                
                # 计算95%分位数作为上限，过滤极端异常值
                loss_upper_limit = np.percentile(valid_losses, 95)
                
                # 过滤掉极端异常值以便更好地可视化
                filtered_losses = [min(loss, loss_upper_limit) for loss in valid_losses]
                
                axes[2, 0].plot(loss_x, filtered_losses, "orange", alpha=0.7)
                
                # 添加注释说明已过滤极端值
                axes[2, 0].set_title("Training Loss Over Time (95th percentile cap)")
                axes[2, 0].set_xlabel("Episodes")
                axes[2, 0].set_ylabel("Loss")
                axes[2, 0].grid(True)

        # 10. 胜负平累积统计
        wins = history["wins_game"][:total_episodes]
        losses = history["losses_game"][:total_episodes]
        draws = history["draws_game"][:total_episodes]

        min_len = min(len(wins), len(losses), len(draws))
        wins = wins[:min_len]
        losses = losses[:min_len]
        draws = draws[:min_len]
        
        wins_cum = np.cumsum(wins)
        losses_cum = np.cumsum(losses)
        draws_cum = np.cumsum(draws)

        axes[2, 1].plot(episodes[: len(wins_cum)], wins_cum, "g-", label="Wins", alpha=0.7)
        axes[2, 1].plot(episodes[: len(losses_cum)], losses_cum, "r-", label="Losses", alpha=0.7)
        axes[2, 1].plot(episodes[: len(draws_cum)], draws_cum, "y-", label="Draws", alpha=0.7)
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
            "total_wins": sum(history["wins_game"]),
            "total_losses": sum(history["losses_game"]),
            "total_draws": sum(history["draws_game"]),
            "final_win_rate": history["win_rates"][-1] if history["win_rates"] else 0,
            "average_reward": np.mean(history["rewards"]) if history["rewards"] else 0,
            "reward_std": np.std(history["rewards"]) if history["rewards"] else 0,
        }
        
        # 添加每百轮统计信息
        if history["hundred_win_rates"]:
            summary["recent_hundred_win_rate"] = history["hundred_win_rates"][-1]
        if history["hundred_average_steps"]:
            summary["recent_hundred_avg_steps"] = history["hundred_average_steps"][-1]
        
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
    
    
def plot_saved_training_data(data_path=None, history=None, output_path=None, session_name="Training Results"):
    """
    单独绘制训练历史数据图表，可从文件加载或直接提供数据
    
    Args:
        data_path: 训练历史数据的JSON文件路径，与history参数二选一
        history: 训练历史数据字典，与data_path参数二选一
        output_path: 图表保存路径，不指定则自动生成
        session_name: 会话名称，用于图表标题
    """
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from datetime import datetime
    matplotlib.use("Agg")  # 使用非交互式后端以便于保存图像
    
    if history is None and data_path is None:
        raise ValueError("必须提供训练历史数据或数据文件路径")
        
    # 从文件加载数据
    if history is None:
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
                if "training_history" in data:
                    history = data["training_history"]
                else:
                    history = data  # 假设整个文件就是训练历史
        except Exception as e:
            raise ValueError(f"无法从文件加载数据: {e}")
    
    # 确保输出路径存在
    if output_path is None:
        output_dir = "training_plots"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"training_plot_{timestamp}.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
    # 准备数据
    episodes = history.get("episodes", [])
    if not episodes:
        episodes = list(range(1, len(history.get("rewards", [])) + 1))
        
    total_episodes = len(episodes)
    if total_episodes == 0:
        raise ValueError("训练历史中未找到有效的回合数据")
    
    # 创建3x4的图表网格
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(20, 15))
    axes = fig.subplots(3, 4)
    
    # 使用英文标题避免字体问题
    fig.suptitle(f"Training History - {session_name}", fontsize=16)
    
    # 第一行 - 基础性能指标
    # 1. 胜率随时间变化
    if "win_rates" in history and history["win_rates"]:
        win_rates = history["win_rates"]
        axes[0, 0].plot(episodes[:len(win_rates)], win_rates, "g-", alpha=0.7)
        axes[0, 0].set_title("Win Rate Over Time")
        axes[0, 0].set_xlabel("Episodes")
        axes[0, 0].set_ylabel("Win Rate")
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(True)
    
    # 2. 奖励随时间变化
    if "rewards" in history and history["rewards"]:
        rewards = np.array(history["rewards"])
        axes[0, 1].plot(episodes[:len(rewards)], rewards, "b-", alpha=0.7)
        # 添加移动平均线
        if len(rewards) > 10:
            window = min(50, len(rewards) // 10)
            if window > 0:
                ma_rewards = np.convolve(rewards, np.ones(window) / window, mode="valid")
                ma_episodes = episodes[window - 1:len(rewards)]
                if len(ma_episodes) == len(ma_rewards):
                    axes[0, 1].plot(ma_episodes, ma_rewards, "r-", linewidth=2, label=f"MA({window})")
                    axes[0, 1].legend()
        axes[0, 1].set_title("Reward Over Time")
        axes[0, 1].set_xlabel("Episodes")
        axes[0, 1].set_ylabel("Reward")
        axes[0, 1].grid(True)
    
    # 3. Epsilon衰减
    if "epsilons" in history and history["epsilons"]:
        epsilons = history["epsilons"]
        axes[0, 2].plot(episodes[:len(epsilons)], epsilons, "r-", alpha=0.7)
        axes[0, 2].set_title("Epsilon Decay")
        axes[0, 2].set_xlabel("Episodes")
        axes[0, 2].set_ylabel("Epsilon")
        axes[0, 2].set_ylim([0, 1])
        axes[0, 2].grid(True)
    
    # 4. 学习率随时间变化
    if "learning_rates" in history and history["learning_rates"]:
        learning_rates = [lr for lr in history["learning_rates"] if lr is not None]
        if learning_rates:
            axes[0, 3].plot(episodes[:len(learning_rates)], learning_rates, "m-", alpha=0.7)
            axes[0, 3].set_title("Learning Rate Over Time")
            axes[0, 3].set_xlabel("Episodes")
            axes[0, 3].set_ylabel("Learning Rate")
            axes[0, 3].grid(True)
    
    # 第二行 - 步数和时间分析
    # 5. 回合步数随时间变化
    if "steps" in history and history["steps"]:
        steps = history["steps"]
        axes[1, 0].plot(episodes[:len(steps)], steps, "orange", alpha=0.7)
        # 添加移动平均线
        if len(steps) > 10:
            window = min(50, len(steps) // 10)
            if window > 0:
                ma_steps = np.convolve(steps, np.ones(window) / window, mode="valid")
                ma_episodes = episodes[window - 1:len(steps)]
                if len(ma_episodes) == len(ma_steps):
                    axes[1, 0].plot(ma_episodes, ma_steps, "red", linewidth=2, label=f"MA({window})")
                    axes[1, 0].legend()
        axes[1, 0].set_title("Steps per Episode")
        axes[1, 0].set_xlabel("Episodes")
        axes[1, 0].set_ylabel("Steps")
        axes[1, 0].grid(True)
    
    # 6. 每个回合的运行时间
    if "episode_times" in history and history["episode_times"]:
        episode_times = history["episode_times"]
        axes[1, 1].plot(episodes[:len(episode_times)], episode_times, "purple", alpha=0.7)
        # 添加移动平均线
        if len(episode_times) > 10:
            window = min(50, len(episode_times) // 10)
            if window > 0:
                ma_times = np.convolve(episode_times, np.ones(window) / window, mode="valid")
                ma_episodes = episodes[window - 1:len(episode_times)]
                if len(ma_episodes) == len(ma_times):
                    axes[1, 1].plot(ma_episodes, ma_times, "red", linewidth=2, label=f"MA({window})")
                    axes[1, 1].legend()
        axes[1, 1].set_title("Episode Time")
        axes[1, 1].set_xlabel("Episodes")
        axes[1, 1].set_ylabel("Time (s)")
        axes[1, 1].grid(True)
    
    # 7. 每100回合的胜率变化 - 使用新的hundred_win_rates字段
    if "hundred_win_rates" in history and history["hundred_win_rates"]:
        hundred_episodes = [(i+1)*100 for i in range(len(history["hundred_win_rates"]))]
        axes[1, 2].plot(hundred_episodes, history["hundred_win_rates"], "cyan", marker="o", alpha=0.7)
        axes[1, 2].set_title("Win Rate (per 100 episodes)")
        axes[1, 2].set_xlabel("Episodes")
        axes[1, 2].set_ylabel("Win Rate")
        axes[1, 2].grid(True)
    
    # 8. 每100回合的平均步长 - 使用新的hundred_average_steps字段
    if "hundred_average_steps" in history and history["hundred_average_steps"]:
        hundred_episodes = [(i+1)*100 for i in range(len(history["hundred_average_steps"]))]
        axes[1, 3].plot(hundred_episodes, history["hundred_average_steps"], "brown", marker="s", alpha=0.7)
        axes[1, 3].set_title("Average Steps (per 100 episodes)")
        axes[1, 3].set_xlabel("Episodes")
        axes[1, 3].set_ylabel("Average Steps")
        axes[1, 3].grid(True)
    
    # 第三行 - 损失和累积统计
    # 9. 训练损失随时间变化 - 使用losses_data字段并处理异常值
    if "losses_data" in history and history["losses_data"]:
        valid_losses = [loss for loss in history["losses_data"] if loss is not None]
        if valid_losses:
            if len(valid_losses) > total_episodes:
                loss_x = np.linspace(episodes[0], episodes[-1], len(valid_losses))
            else:
                loss_x = episodes[:len(valid_losses)]
            
            # 计算95%分位数作为上限，过滤极端异常值
            loss_upper_limit = np.percentile(valid_losses, 95)
            
            # 过滤掉极端异常值以便更好地可视化
            filtered_losses = [min(loss, loss_upper_limit) for loss in valid_losses]
            
            axes[2, 0].plot(loss_x, filtered_losses, "orange", alpha=0.7)
            axes[2, 0].set_title("Training Loss (95th percentile cap)")
            axes[2, 0].set_xlabel("Episodes")
            axes[2, 0].set_ylabel("Loss")
            axes[2, 0].grid(True)
    
    # 10. 游戏结果累积图 - 使用新的字段名
    wins = history.get("wins_game", [])
    draws = history.get("draws_game", [])
    losses_game = history.get("losses_game", [])
    
    # 确保数据长度一致
    min_len = min(len(wins), len(losses_game), len(draws))
    if min_len > 0:
        wins = wins[:min_len]
        losses_game = losses_game[:min_len]
        draws = draws[:min_len]
        ep = episodes[:min_len]
        
        # 累积结果
        wins_cum = np.cumsum(wins)
        losses_cum = np.cumsum(losses_game)
        draws_cum = np.cumsum(draws)
        
        axes[2, 1].plot(ep, wins_cum, "g-", label="Wins", alpha=0.7)
        axes[2, 1].plot(ep, losses_cum, "r-", label="Losses", alpha=0.7)
        axes[2, 1].plot(ep, draws_cum, "y-", label="Draws", alpha=0.7)
        axes[2, 1].set_title("Cumulative Game Results")
        axes[2, 1].set_xlabel("Episodes")
        axes[2, 1].set_ylabel("Count")
        axes[2, 1].legend()
        axes[2, 1].grid(True)
    
    # 11. 步数分布直方图
    if "steps" in history and history["steps"]:
        steps = history["steps"]
        axes[2, 2].hist(steps, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        axes[2, 2].set_title("Steps Distribution")
        axes[2, 2].set_xlabel("Steps per Episode")
        axes[2, 2].set_ylabel("Frequency")
        axes[2, 2].grid(True)
        
        # 添加统计信息
        mean_steps = np.mean(steps)
        std_steps = np.std(steps)
        axes[2, 2].axvline(mean_steps, color="red", linestyle="--", 
                        label=f"Mean: {mean_steps:.1f}±{std_steps:.1f}")
        axes[2, 2].legend()
    
    # 12. 回合时间分布直方图
    if "episode_times" in history and history["episode_times"]:
        episode_times = history["episode_times"]
        axes[2, 3].hist(episode_times, bins=30, alpha=0.7, color="lightgreen", edgecolor="black")
        axes[2, 3].set_title("Episode Time Distribution")
        axes[2, 3].set_xlabel("Time per Episode (s)")
        axes[2, 3].set_ylabel("Frequency")
        axes[2, 3].grid(True)
        
        # 添加统计信息
        mean_time = np.mean(episode_times)
        std_time = np.std(episode_times)
        axes[2, 3].axvline(mean_time, color="red", linestyle="--", 
                        label=f"Mean: {mean_time:.2f}±{std_time:.2f}s")
        axes[2, 3].legend()
    
    # 调整布局并保存
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为标题留出空间
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"训练历史图表已保存到: {output_path}")
    return output_path

if __name__ == "main":
    data_magager = TrainingDataManager()
    history = data_magager.load_session("model_data/training_history_D3QNAgent.json")
    
    # 绘制数据
    plot_saved_training_data(
        history=history["training_history"],
        session_name="Loaded Training Data"
    )