import AgentFight as AG
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

def check_game_over(board: AG.Board) -> int:
    """检查游戏是否结束
    Returns:
        -1: 游戏继续
        0: 红方(player_id=0)获胜
        1: 蓝方(player_id=1)获胜  
        2: 平局
    """
    red_pieces = board.get_player_pieces(0)
    blue_pieces = board.get_player_pieces(1)

    if not red_pieces:
        return 1  # Blue wins
    elif not blue_pieces:
        return 0  # Red wins
    elif len(red_pieces) == 1 and len(blue_pieces) == 1:
        rr, rc = red_pieces[0]
        br, bc = blue_pieces[0]
        red_piece = board.get_piece(rr, rc)
        blue_piece = board.get_piece(br, bc)

        # 只有当两个棋子都已翻开时，才能判断是否能互相捕获
        if red_piece.revealed and blue_piece.revealed:
            can_red_attack = (red_piece.strength > blue_piece.strength) or \
                            (red_piece.strength == 1 and blue_piece.strength == 8)
            can_blue_attack = (blue_piece.strength > red_piece.strength) or \
                            (blue_piece.strength == 1 and red_piece.strength == 8)

            # 如果它们相邻，且双方都无法捕获对方，则为和棋
            if board.is_adjacent((rr, rc), (br, bc)):
                if not can_red_attack and not can_blue_attack:
                    return 2  # Draw
            else: # 如果不相邻，也无法捕获，则为和棋
                if not can_red_attack and not can_blue_attack:
                    return 2  # Draw
        # 如果有一方或双方未翻开，游戏继续 (因为信息不完全，未来可能仍有变化)
    return -1  # Game continues

def compare_agents_performance(agent1=None, agent2=None, num_games: int = 1000, 
                             agent1_name: str = "Agent1", agent2_name: str = "Agent2") -> Dict[str, Any]:
    """比较两个智能体对战的性能"""
    
    # 如果没有提供智能体，使用随机玩家
    if agent1 is None:
        agent1 = AG.RandomPlayer(player_id=0)
        agent1_name = "Random"
    else:
        agent1.player_id = 0
        
    if agent2 is None:
        agent2 = AG.RandomPlayer(player_id=1)
        agent2_name = "Random"
    else:
        agent2.player_id = 1
    
    print(f"开始 {agent1_name} vs {agent2_name} 对战测试...")
    print(f"总共进行 {num_games} 场游戏")
    
    # 统计变量
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    for game_num in range(num_games):
        board = AG.Board()
        current_player = 0  # 0: agent1, 1: agent2
        turn_count = 0
        max_turns = 1000  # 防止无限循环
        
        while turn_count < max_turns:
            turn_count += 1
            
            # 当前玩家行动
            if current_player == 0:
                success = agent1.take_turn(board)
            else:
                success = agent2.take_turn(board)
            
            if not success:
                # 无法移动，检查游戏结果
                result = check_game_over(board)
                if result != -1:
                    if result == 0:
                        agent1_wins += 1
                    elif result == 1:
                        agent2_wins += 1
                    else:
                        draws += 1
                break
            
            # 切换玩家
            current_player = 1 - current_player
            
            # 检查游戏结果
            result = check_game_over(board)
            if result != -1:
                if result == 0:
                    agent1_wins += 1
                elif result == 1:
                    agent2_wins += 1
                else:
                    draws += 1
                break
            
        # 超时判为平局
        if turn_count >= max_turns:
            draws += 1
        
        # 显示进度
        if (game_num + 1) % 10 == 0:
            current_agent1_winrate = (agent1_wins + 0.5 * draws) / (game_num + 1) * 100
            print(f"已完成 {game_num + 1} 场游戏，当前 {agent1_name} 胜率: {current_agent1_winrate:.2f}%")
    
    # 计算统计结果
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
    
    for bar, result in zip(bars, results):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{result:.1f}%', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"{agent1_name}_vs_{agent2_name}_comparison.png"
    plt.savefig(filename)
    print(f"对战结果图表已保存为: {filename}")
    plt.show()
    
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
    """比较多个智能体之间的性能"""
    results = {}
    agent_names = list(agents_dict.keys())
    
    print(f"开始多智能体比较，共有 {len(agent_names)} 个智能体")
    print(f"智能体列表: {', '.join(agent_names)}")
    
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

if __name__ == "__main__":
    # 示例1: 两个随机玩家对战
    print("示例1: 两个随机玩家对战")
    result = compare_agents_performance(num_games=100)