import numpy as np
import time
import os
import argparse
import copy
from datetime import datetime
from typing import Dict, Any, Callable, Type
from base import Player, BaseTrainer, Board
from training_data_manager import TrainingDataManager

def get_agent_specific_info(agent, print_q_loss: bool = True) -> str:
    """获取智能体特定的信息字符串"""
    info_parts = []
    
    # Q值信息 (AQ 专有)
    if hasattr(agent, 'q_values_history') and agent.q_values_history and print_q_loss:
        recent_q = np.mean(agent.q_values_history[-20:])
        info_parts.append(f"Q值 = {recent_q:.3f}")
    
    # 损失信息 (两者都有)
    if hasattr(agent, 'losses') and agent.losses and print_q_loss:
        recent_loss = np.mean(agent.losses[-10:])
        info_parts.append(f"损失 = {recent_loss:.4f}")
    
    return ", " + ", ".join(info_parts) if info_parts else ""

def get_agent_hyperparams_info(agent) -> str:
    """获取智能体超参数信息"""
    info_parts = []
    
    # Epsilon
    if hasattr(agent, 'epsilon'):
        info_parts.append(f"ε = {agent.epsilon:.3f}")
    
    # 学习率
    if hasattr(agent, 'get_learning_rate'):
        lr = agent.get_learning_rate()
        if lr is not None:
            info_parts.append(f"lr = {lr:.6f}")
    elif hasattr(agent, 'learning_rate'):
        info_parts.append(f"lr = {agent.learning_rate:.6f}")
    
    # 折扣因子 (AQ 专有)
    if hasattr(agent, 'discount_factor'):
        info_parts.append(f"γ = {agent.discount_factor:.3f}")
    elif hasattr(agent, 'gamma'):
        info_parts.append(f"γ = {agent.gamma:.3f}")
    
    return ", " + ", ".join(info_parts) if info_parts else ""

def update_agent_hyperparams(agent, win_rate: float, episode: int, phase: str):
    """统一更新智能体超参数"""
    # 学习率调整频率根据阶段不同
    lr_freq_map = {
        'phase1': 100,
        'phase2': 150, 
        'phase3': 200
    }
    freq = lr_freq_map.get(phase, 100)
    
    if episode > 0 and episode % freq == 0:
        if hasattr(agent, 'update_learning_rate'):
            agent.update_learning_rate(win_rate)
    
    # Epsilon 衰减 (仅在阶段2和阶段3)
    if phase in ['phase2', 'phase3'] and hasattr(agent, 'decay_epsilon'):
        agent.decay_epsilon()
    
    # 折扣因子更新 (AQ 专有，仅在阶段2和阶段3)
    if phase in ['phase2', 'phase3'] and hasattr(agent, 'update_discount_factor'):
        agent.update_discount_factor()

def calculate_recent_win_rate(data_manager, start_episode: int, episode: int, window: int) -> float:
    """计算最近窗口内的胜率"""
    if not data_manager:
        return 0.0
    
    recent_wins = 0
    recent_episodes = min(window, episode + 1)
    for i in range(max(0, start_episode + episode - recent_episodes + 1), 
                   start_episode + episode + 1):
        if (len(data_manager.current_session['training_history']['wins']) > i and 
            data_manager.current_session['training_history']['wins'][i] == 1):
            recent_wins += 1
    
    return recent_wins / recent_episodes if recent_episodes > 0 else 0.0

def train_with_curriculum_generic(
    agent: Player, 
    opponent: Player, 
    trainer_class: Type[BaseTrainer],
    episodes: int = 2000, 
    data_manager: TrainingDataManager = None, 
    print_interval: int = 50,
    **trainer_kwargs
):
    """
    通用课程式训练函数 - 支持 DQN 和 ApproximateQ
    
    Args:
        agent: 智能体实例 (DQNAgent 或 ApproximateQAgent)
        opponent: 对手智能体
        trainer_class: 训练器类 (DQNTrainer 或 ApproximateQTrainer)
        episodes: 总训练回合数
        data_manager: 数据管理器
        print_interval: 打印进度的间隔
        **trainer_kwargs: 传递给训练器的额外参数
    """
    trainer = trainer_class(agent, opponent, **trainer_kwargs)
    
    total_episodes = episodes
    agent_type = agent.__class__.__name__
    
    # 添加计时记录
    training_start_time = time.time()
    phase_times = {}
    
    print(f"开始 {agent_type} 课程式训练...")
    
    # 保存原始参数
    original_epsilon_decay = getattr(agent, 'epsilon_decay', None)
    original_discount_growth = getattr(agent, 'discount_growth', None)
    
    # 阶段1: 快速探索 (30%)
    phase1_episodes = int(total_episodes * 0.3)
    print(f"阶段1: 快速探索学习 ({phase1_episodes} episodes) - 每{print_interval}回合输出进度")
    print(f"epsilon固定在: {agent.epsilon:.3f}")
    if hasattr(agent, 'get_learning_rate'):
        print(f"初始学习率: {agent.get_learning_rate():.6f}")
    if hasattr(agent, 'discount_factor'):
        print(f"初始折扣因子: {agent.discount_factor:.3f}")
    
    # 阶段1特殊设置
    if hasattr(agent, 'epsilon_decay'):
        agent.epsilon_decay = 1.0  # 禁用epsilon衰减
    if hasattr(agent, 'discount_growth'):
        agent.discount_growth = agent.discount_growth * 0.5  # 减缓折扣因子增长
    
    # DQN 专有设置
    if hasattr(trainer, 'target_update_freq'):
        trainer.target_update_freq = 30
    
    phase1_start_time = time.time()
    phase1_stats = run_training_phase(
        trainer, agent, phase1_episodes, 'phase1', 0,
        data_manager, print_interval, phase1_start_time
    )
    phase1_end_time = time.time()
    phase_times['phase1'] = phase1_end_time - phase1_start_time
    
    print(f"阶段1完成! 当前状态:{get_agent_hyperparams_info(agent)}")
    print(f"阶段1总耗时: {phase_times['phase1']:.1f}秒, 平均: {phase_times['phase1']/phase1_episodes:.2f}秒/回合")
    
    # 阶段2: 平衡学习 (50%)
    phase2_episodes = int(total_episodes * 0.5)
    print(f"\n阶段2: 平衡学习 ({phase2_episodes} episodes)")
    print(f"启用缓慢epsilon衰减，当前epsilon: {agent.epsilon:.3f}")
    
    # 阶段2设置
    if hasattr(agent, 'epsilon_decay'):
        agent.epsilon_decay = 0.9995  # 启用缓慢epsilon衰减
    if hasattr(agent, 'discount_growth') and original_discount_growth:
        agent.discount_growth = original_discount_growth  # 恢复正常折扣因子增长
    
    # DQN 专有设置
    if hasattr(trainer, 'target_update_freq'):
        trainer.target_update_freq = 50
    
    phase2_start_time = time.time()
    phase2_stats = run_training_phase(
        trainer, agent, phase2_episodes, 'phase2', phase1_episodes,
        data_manager, print_interval, phase2_start_time
    )
    phase2_end_time = time.time()
    phase_times['phase2'] = phase2_end_time - phase2_start_time
    
    print(f"阶段2完成! 当前状态:{get_agent_hyperparams_info(agent)}")
    print(f"阶段2总耗时: {phase_times['phase2']:.1f}秒, 平均: {phase_times['phase2']/phase2_episodes:.2f}秒/回合")
    
    # 阶段3: 策略精炼 (20%)
    phase3_episodes = total_episodes - phase1_episodes - phase2_episodes
    print(f"\n阶段3: 策略精炼 ({phase3_episodes} episodes)")
    print(f"恢复正常epsilon衰减，当前epsilon: {agent.epsilon:.3f}")
    
    # 阶段3设置
    if hasattr(agent, 'epsilon_decay') and original_epsilon_decay:
        agent.epsilon_decay = original_epsilon_decay  # 恢复正常epsilon衰减
    
    # DQN 专有设置
    if hasattr(trainer, 'target_update_freq'):
        trainer.target_update_freq = 100
    
    phase3_start_time = time.time()
    phase3_stats = run_training_phase(
        trainer, agent, phase3_episodes, 'phase3', phase1_episodes + phase2_episodes,
        data_manager, print_interval, phase3_start_time
    )
    phase3_end_time = time.time()
    phase_times['phase3'] = phase3_end_time - phase3_start_time
    
    # 总结训练时间
    total_training_time = phase3_end_time - training_start_time
    
    print(f"\n{agent_type} 课程训练完成!")
    print(f"最终状态:{get_agent_hyperparams_info(agent)}")
    if hasattr(agent, 'get_stats'):
        print(f"最终胜率: {agent.get_stats()['win_rate']:.3f}")
    
    print(f"\n=== 训练耗时统计 ===")
    print(f"阶段1 ({phase1_episodes} episodes): {phase_times['phase1']:.1f}秒 (平均 {phase_times['phase1']/phase1_episodes:.2f}秒/回合)")
    print(f"阶段2 ({phase2_episodes} episodes): {phase_times['phase2']:.1f}秒 (平均 {phase_times['phase2']/phase2_episodes:.2f}秒/回合)")
    print(f"阶段3 ({phase3_episodes} episodes): {phase_times['phase3']:.1f}秒 (平均 {phase_times['phase3']/phase3_episodes:.2f}秒/回合)")
    print(f"总训练时间: {total_training_time:.1f}秒 ({total_training_time/60:.1f}分钟)")
    print(f"总体平均: {total_training_time/total_episodes:.2f}秒/回合")
    
    # 在训练结束后打印详细统计
    if data_manager:
        summary = data_manager.get_summary_stats()
        print(f"\n=== 详细训练统计 ===")
        if 'average_steps' in summary:
            print(f"平均步数: {summary['average_steps']:.1f} ± {summary['steps_std']:.1f}")
            print(f"步数范围: {summary['min_steps']} - {summary['max_steps']}")
        if 'average_episode_time' in summary:
            print(f"平均每回合时间: {summary['average_episode_time']:.3f} ± {summary['episode_time_std']:.3f} 秒")
        print(f"总胜率: {summary['final_win_rate']:.3f}")
        print(f"平均奖励: {summary['average_reward']:.3f} ± {summary['reward_std']:.3f}")
        
        # 智能体特有统计
        if hasattr(agent, 'q_values_history') and agent.q_values_history:
            print(f"平均Q值: {np.mean(agent.q_values_history):.3f} ± {np.std(agent.q_values_history):.3f}")
        if hasattr(agent, 'losses') and agent.losses:
            print(f"平均损失: {np.mean(agent.losses):.4f} ± {np.std(agent.losses):.4f}")
    
    return data_manager.current_session['training_history'] if data_manager else {}

def run_training_phase(
    trainer: BaseTrainer, 
    agent: Player, 
    num_episodes: int, 
    phase_name: str, 
    episode_offset: int,
    data_manager: TrainingDataManager, 
    print_interval: int,
    phase_start_time: float
):
    """运行单个训练阶段"""
    batch_wins = 0
    batch_loses = 0
    batch_draws = 0
    batch_steps = []
    batch_start_episode = 0
    
    for episode in range(num_episodes):
        episode_start_time = time.time()
        total_reward, steps, result = trainer.train_episode()
        episode_end_time = time.time()
        episode_time = episode_end_time - episode_start_time
        
        # 记录批次数据
        batch_steps.append(steps)
        if result == 0:  # 智能体胜利
            batch_wins += 1
        elif result == 1:  # 对手胜利
            batch_loses += 1
        else:  # 平局
            batch_draws += 1
        
        # 定期输出进度
        if episode % print_interval == 0 or episode == num_episodes - 1:
            batch_episodes = episode - batch_start_episode + 1
            batch_win_rate = (batch_wins + batch_draws / 2) / batch_episodes
            avg_steps = sum(batch_steps) / len(batch_steps) if batch_steps else 0
            
            # 计算耗时
            current_time = time.time()
            if episode == 0:
                batch_time = current_time - phase_start_time
            else:
                phase_elapsed = current_time - phase_start_time
                if episode >= print_interval:
                    avg_time_per_episode = phase_elapsed / (episode + 1)
                    batch_time = batch_episodes * avg_time_per_episode
                else:
                    batch_time = phase_elapsed
            
            avg_time_per_episode = batch_time / batch_episodes if batch_episodes > 0 else 0
            
            # 组装输出信息
            param_info = get_agent_hyperparams_info(agent)
            agent_info = get_agent_specific_info(agent, print_q_loss=True)
            time_info = f", 用时 = {batch_time:.1f}s, 平均 = {avg_time_per_episode:.2f}s/ep"
            
            print(f"{phase_name.upper()} - 回合 {episode}: 奖励 = {total_reward:.2f}, 步数 = {steps}, "
                  f"胜 = {batch_wins}, 负 = {batch_loses}, 平 = {batch_draws}, "
                  f"批次胜率 = {batch_win_rate:.3f}, 平均步长 = {avg_steps:.1f}{param_info}{agent_info}{time_info}")
            
            # 记录批次统计到数据管理器
            if data_manager:
                data_manager.log_batch_stats(
                    batch_wins, batch_loses, batch_draws, 
                    batch_win_rate, avg_steps
                )
            
            # 重置批次统计
            batch_wins = 0
            batch_loses = 0
            batch_draws = 0
            batch_steps = []
            batch_start_episode = episode + 1
        
        # 更新超参数
        recent_win_rate = calculate_recent_win_rate(
            data_manager, episode_offset, episode, 
            100 if phase_name == 'phase1' else 150 if phase_name == 'phase2' else 200
        )
        update_agent_hyperparams(agent, recent_win_rate, episode, phase_name)
        
        # 记录每个episode的详细数据
        if data_manager:
            data_manager.log_episode(
                episode=episode_offset + episode,
                reward=total_reward,
                result=result,
                learning_rate=agent.get_learning_rate() if hasattr(agent, 'get_learning_rate') else getattr(agent, 'learning_rate', None),
                epsilon=getattr(agent, 'epsilon', None),
                loss=agent.losses[-1] if hasattr(agent, 'losses') and agent.losses else None,
                phase=f'{phase_name}_{"exploration" if phase_name == "phase1" else "balance" if phase_name == "phase2" else "refinement"}',
                steps=steps,
                episode_time=episode_time,
                discount_factor=getattr(agent, 'discount_factor', None)
            )
    
    return {
        'wins': batch_wins,
        'losses': batch_loses,
        'draws': batch_draws
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
    **trainer_kwargs
):
    """
    通用的训练或加载模型函数
    
    Args:
        agent_class: 智能体类 (DQNAgent 或 ApproximateQAgent)
        trainer_class: 训练器类 (DQNTrainer 或 ApproximateQTrainer)
        agent_config: 智能体初始化配置
        model_name: 模型名称
        force_retrain: 是否强制重新训练
        episodes: 训练回合数
        lr_strategy: 学习率策略
        print_interval: 打印间隔
        **trainer_kwargs: 传递给训练器的额外参数
    """
    from AgentFight import RandomPlayer
    
    # 创建智能体实例
    agent = agent_class(**agent_config)
    random_opponent = RandomPlayer(player_id=1)
    
    # 设置学习率策略
    if hasattr(agent, 'enable_adaptive_lr') and hasattr(agent, 'disable_adaptive_lr'):
        if lr_strategy == "adaptive":
            agent.enable_adaptive_lr()
            print("使用自适应学习率策略")
        elif lr_strategy == "fixed":
            agent.disable_adaptive_lr()
            print("使用固定学习率策略")
        else:  # hybrid for DQN
            print(f"使用{lr_strategy}学习率策略")
    
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
        
        # 创建数据管理器
        data_manager = TrainingDataManager()
        session_name = f"{agent.__class__.__name__}_{lr_strategy}_{episodes}eps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        data_manager.start_session(agent, session_name)
        
        # 使用通用课程式训练
        combined_history = train_with_curriculum_generic(
            agent, random_opponent, trainer_class, episodes, 
            data_manager, print_interval, **trainer_kwargs
        )
        
        # 结束数据记录会话
        final_stats = {
            'training_episodes': episodes,
            'lr_strategy': lr_strategy,
            'final_epsilon': getattr(agent, 'epsilon', None),
            'final_learning_rate': agent.get_learning_rate() if hasattr(agent, 'get_learning_rate') else getattr(agent, 'learning_rate', None),
            'final_discount_factor': getattr(agent, 'discount_factor', None)
        }
        data_manager.end_session(agent, final_stats)
        
        # 绘制训练历史
        data_manager.plot_training_history()
        
        # 保存模型
        agent.save_model(model_name)
        
        print(f"训练完成! 最终epsilon: {getattr(agent, 'epsilon', 'N/A')}")
        if hasattr(agent, 'get_stats'):
            print(f"最终胜率: {agent.get_stats()['win_rate']:.3f}")
        
        # 设置为测试模式
        agent.epsilon = 0.0
        agent.ai_type = f"{agent.__class__.__name__} (Trained)"
    
    return agent, random_opponent

def test_agent_generic(agent: Player, opponent: Player, num_games: int = 100, show_individual: bool = False, display: bool = False):
    """通用的智能体测试函数"""
    from base import Game
    
    # 设置为测试模式
    if hasattr(agent, 'set_training_mode'):
        agent.set_training_mode(False)
    
    print(f"\n开始测试 {num_games} 场游戏...")
    print(f"测试模式 - epsilon: {getattr(agent, 'epsilon', 'N/A')}")
    
    wins = 0
    draws = 0
    
    for i in range(num_games):
        game = Game(agent, opponent, display=display, delay=0.1)
        result = game.run()
        
        if result == 0:  # Agent wins
            wins += 1
        elif result == 2:  # Draw
            draws += 1
            
        if show_individual and num_games <= 10:
            print(f"游戏 {i+1}: {'胜利' if result == 0 else '失败' if result == 1 else '平局'}")
    
    win_rate = wins / num_games
    draw_rate = draws / num_games
    
    print(f"\n测试结果:")
    print(f"胜利: {wins}/{num_games} ({win_rate:.3f})")
    print(f"平局: {draws}/{num_games} ({draw_rate:.3f})")
    print(f"失败: {num_games - wins - draws}/{num_games}")
    print(f"有效胜率 (胜+0.5*平): {win_rate + 0.5 * draw_rate:.3f}")
    
    return {
        'wins': wins,
        'draws': draws,
        'losses': num_games - wins - draws,
        'win_rate': win_rate,
        'draw_rate': draw_rate,
        'effective_win_rate': win_rate + 0.5 * draw_rate
    }

# 智能体配置映射
AGENT_CONFIGS = {
    'dqn': {
        'agent_class': None,  # 将在运行时导入
        'trainer_class': None,
        'agent_config': {
            'player_id': 0,
            'learning_rate': 5e-4,
            'epsilon': 0.9,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.995,
            'batch_size': 128,
            'memory_size': 50000,
            'use_dueling': True,
            'use_double': True
        },
        'model_name': 'final_D3QNAgent',
        'trainer_kwargs': {}
    },
    'aq': {
        'agent_class': None,  # 将在运行时导入
        'trainer_class': None,
        'agent_config': {
            'player_id': 0,
            'learning_rate': 0.02,
            'epsilon': 0.9,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.995,
            'discount_factor': 0.3,
            'discount_max': 0.95,
            'discount_growth': 0.002
        },
        'model_name': 'final_ApproximateQAgent',
        'trainer_kwargs': {}
    }
}

def get_agent_imports(agent_type: str):
    """动态导入智能体类"""
    if agent_type == 'dqn':
        from DQN import DQNAgent, DQNTrainer
        return DQNAgent, DQNTrainer
    elif agent_type == 'aq':
        from ApproximateQAgent import ApproximateQAgent, ApproximateQTrainer
        return ApproximateQAgent, ApproximateQTrainer
    else:
        raise ValueError(f"不支持的智能体类型: {agent_type}")

def main():
    """主函数 - 支持多种智能体的训练和测试"""
    parser = argparse.ArgumentParser(description='通用智能体训练和测试系统')
    parser.add_argument('--agent', choices=['dqn', 'aq'], required=True, 
                       help='选择智能体类型: dqn (DQN Agent) 或 aq (Approximate Q Agent)')
    parser.add_argument('--retrain', action='store_true', help='强制重新训练模型')
    parser.add_argument('--episodes', type=int, default=2000, help='训练回合数')
    parser.add_argument('--test-games', type=int, default=100, help='测试游戏数量')
    parser.add_argument('--no-display', action='store_true', help='不显示游戏界面')
    parser.add_argument('--lr-strategy', choices=['adaptive', 'fixed', 'hybrid'], 
                       default='adaptive', help='学习率调整策略')
    parser.add_argument('--test-only', action='store_true', help='仅测试，不训练')
    parser.add_argument('--print-interval', type=int, default=50, help='训练进度输出间隔')
    
    args = parser.parse_args()
    
    # 获取智能体配置
    config = AGENT_CONFIGS[args.agent].copy()
    agent_class, trainer_class = get_agent_imports(args.agent)
    config['agent_class'] = agent_class
    config['trainer_class'] = trainer_class
    
    print(f"选择的智能体: {agent_class.__name__}")
    
    # 训练或加载模型
    if not args.test_only:
        agent, opponent = train_or_load_model_generic(
            agent_class=config['agent_class'],
            trainer_class=config['trainer_class'],
            agent_config=config['agent_config'],
            model_name=config['model_name'],
            force_retrain=args.retrain,
            episodes=args.episodes,
            lr_strategy=args.lr_strategy,
            print_interval=args.print_interval,
            **config['trainer_kwargs']
        )
    else:
        # 仅测试模式：直接加载模型
        print("仅测试模式，加载已训练模型...")
        agent = agent_class(player_id=0)
        if not agent.load_model(config['model_name']):
            print("无法加载模型，请先训练!")
            return
        from AgentFight import RandomPlayer
        opponent = RandomPlayer(player_id=1)
    
    # 测试
    test_results = test_agent_generic(
        agent, opponent, 
        num_games=args.test_games, 
        show_individual=(args.test_games <= 10),
        display=not args.no_display
    )
    
    print(f"\n{agent.__class__.__name__} 测试完成!")

if __name__ == "__main__":
    main()

# 运行示例:
# python train_and_record.py --agent dqn --retrain --episodes 5000 --lr-strategy hybrid --test-games 200 --no-display --print-interval 100
# python train_and_record.py --agent aq --retrain --episodes 5000 --lr-strategy adaptive --test-games 200 --no-display --print-interval 100
# python train_and_record.py --agent dqn --test-only --test-games 100 --no-display