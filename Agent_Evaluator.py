import AgentFight as AG
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

# 这个我用来批量测试，计算胜率的

def compare_agents_performance(agent1=None, agent2=None, num_games: int = 1000, 
                             agent1_name: str = "Agent1", agent2_name: str = "Agent2") -> Dict[str, Any]:
    """
    比较两个智能体对战的性能
    
    Args:
        agent1: 第一个智能体 (默认为RandomPlayer)
        agent2: 第二个智能体 (默认为RandomPlayer)  
        num_games: 对战场次数
        agent1_name: 第一个智能体的名称
        agent2_name: 第二个智能体的名称
    
    Returns:
        包含详细统计结果的字典
    """
    
    # 如果没有提供智能体，使用随机玩家
    if agent1 is None:
        agent1 = AG.RandomPlayer(player_id=0)
        agent1_name = "Random"
    else:
        agent1.player_id = 0  # 确保player_id正确
        
    if agent2 is None:
        agent2 = AG.RandomPlayer(player_id=1)
        agent2_name = "Random"
    else:
        agent2.player_id = 1  # 确保player_id正确
    
    print(f"开始 {agent1_name} vs {agent2_name} 对战测试...")
    print(f"总共进行 {num_games} 场游戏")
    
    # 统计变量
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    for game_num in range(num_games):
        # 创建新的棋盘
        board = AG.Board()
        current_player = 0  # 0: agent1, 1: agent2
        game_over = False
        max_turns = 200  # 防止无限循环
        turn_count = 0
        
        while not game_over and turn_count < max_turns:
            turn_count += 1
            
            if current_player == 0:
                # agent1 回合
                if agent1.take_turn(board):
                    current_player = 1
                else:
                    # 无法移动，游戏结束
                    game_over = True
            else:
                # agent2 回合
                if agent2.take_turn(board):
                    current_player = 0
                else:
                    # 无法移动，游戏结束
                    game_over = True
            
            # 检查游戏结束条件
            red_pieces = board.get_player_pieces(0)  # agent1
            blue_pieces = board.get_player_pieces(1)  # agent2
            
            if not red_pieces:
                # agent1 失败
                agent2_wins += 1
                game_over = True
            elif not blue_pieces:
                # agent2 失败
                agent1_wins += 1
                game_over = True
            elif len(red_pieces) == 1 and len(blue_pieces) == 1:
                # 检查是否为和棋
                rr, rc = red_pieces[0]
                br, bc = blue_pieces[0]
                red_piece = board.get_piece(rr, rc)
                blue_piece = board.get_piece(br, bc)
                
                if red_piece.revealed and blue_piece.revealed:
                    can_red_attack = (red_piece.strength > blue_piece.strength) or \
                                   (red_piece.strength == 1 and blue_piece.strength == 8)
                    can_blue_attack = (blue_piece.strength > red_piece.strength) or \
                                    (blue_piece.strength == 1 and red_piece.strength == 8)
                    
                    if board.is_adjacent((rr, rc), (br, bc)):
                        if not can_red_attack and not can_blue_attack:
                            draws += 1
                            game_over = True
                    elif not can_red_attack and not can_blue_attack:
                        draws += 1
                        game_over = True
            
            # 超时判断为和棋
            if turn_count >= max_turns:
                draws += 1
                game_over = True
        
        # 每100场游戏显示进度
        if (game_num + 1) % 100 == 0:
            current_agent1_winrate = agent1_wins / (game_num + 1) * 100
            print(f"已完成 {game_num + 1} 场游戏，当前 {agent1_name} 胜率: {current_agent1_winrate:.2f}%")
    
    # 计算最终统计
    agent1_winrate = agent1_wins / num_games * 100
    agent2_winrate = agent2_wins / num_games * 100
    draw_rate = draws / num_games * 100
    
    # 显示结果
    print("\n" + "="*60)
    print(f"{agent1_name} vs {agent2_name} 对战结果:")
    print(f"总游戏数: {num_games}")
    print(f"{agent1_name} 获胜: {agent1_wins} 场 ({agent1_winrate:.2f}%)")
    print(f"{agent2_name} 获胜: {agent2_wins} 场 ({agent2_winrate:.2f}%)")
    print(f"平局: {draws} 场 ({draw_rate:.2f}%)")
    print("="*60)
    
    # 生成可视化图表
    agents = [agent1_name, agent2_name, "平局"]
    results = [agent1_winrate, agent2_winrate, draw_rate]
    colors = ['blue', 'red', 'gray']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(agents, results, color=colors)
    plt.title(f'{agent1_name} vs {agent2_name} 对战结果')
    plt.ylabel('比例 (%)')
    plt.ylim(0, 100)
    
    # 在柱状图上添加数值标签
    for bar, result in zip(bars, results):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{result:.1f}%', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    filename = f"{agent1_name}_vs_{agent2_name}_comparison.png"
    plt.savefig(filename)
    print(f"对战结果图表已保存为: {filename}")
    plt.show()
    
    # 返回详细结果
    return {
        'total_games': num_games,
        'agent1_name': agent1_name,
        'agent2_name': agent2_name,
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'draws': draws,
        'agent1_winrate': agent1_winrate,
        'agent2_winrate': agent2_winrate,
        'draw_rate': draw_rate
    }

def compare_multiple_agents(agents_dict: Dict[str, Any], num_games: int = 1000) -> Dict[str, Dict]:
    """
    比较多个智能体之间的性能
    
    Args:
        agents_dict: 智能体字典，格式为 {'名称': agent对象}
        num_games: 每对智能体对战的场次数
    
    Returns:
        包含所有对战结果的字典
    """
    results = {}
    agent_names = list(agents_dict.keys())
    
    print(f"开始多智能体比较，共有 {len(agent_names)} 个智能体")
    print(f"智能体列表: {', '.join(agent_names)}")
    
    # 两两对战
    for i in range(len(agent_names)):
        for j in range(i + 1, len(agent_names)):
            agent1_name = agent_names[i]
            agent2_name = agent_names[j]
            agent1 = agents_dict[agent1_name]
            agent2 = agents_dict[agent2_name]
            
            print(f"\n--- {agent1_name} vs {agent2_name} ---")
            result = compare_agents_performance(
                agent1, agent2, num_games, agent1_name, agent2_name
            )
            results[f"{agent1_name}_vs_{agent2_name}"] = result
    
    # 生成总结报告
    print("\n" + "="*80)
    print("多智能体比较总结:")
    for match_name, result in results.items():
        agent1_name = result['agent1_name']
        agent2_name = result['agent2_name']
        agent1_winrate = result['agent1_winrate']
        agent2_winrate = result['agent2_winrate']
        print(f"{agent1_name} vs {agent2_name}: {agent1_winrate:.1f}% - {agent2_winrate:.1f}%")
    print("="*80)
    
    return results

# 使用示例
if __name__ == "__main__":
    # 示例1: 两个随机玩家对战
    print("示例1: 两个随机玩家对战")
    result = compare_agents_performance(num_games=100)
    
    # 示例2: 比较多个智能体（需要实际的智能体对象）
    # agents = {
    #     'Random': AG.RandomPlayer(0),
    #     'Minimax': AG.MinimaxPlayer(0),
    #     # 'Q-Learning': trained_q_agent,
    #     # 'Approximate Q': trained_aq_agent
    # }
    # results = compare_multiple_agents(agents, num_games=500)