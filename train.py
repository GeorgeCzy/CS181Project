from QlearningAgent import QLearningAgent, QLearningTrainer
from ApproxiamateQAgent import ApproximateQAgent, ApproximateQTrainer
from DQN import DQNAgent, DQNTrainer
# from new_sim import RandomPlayer, Game
import new_sim as HM
import AgentFight as AG
from Agent_Evaluator import compare_agents_performance, compare_multiple_agents
import matplotlib.pyplot as plt


def train_rl_agent():
    rl_agent = QLearningAgent(player_id=0, learning_rate=0.1, epsilon=0.1)
    random_opponent = AG.RandomPlayer(player_id=1)
    trainer = QLearningTrainer(rl_agent, random_opponent)
    trainer.train(episodes=2000)
    return rl_agent

def train_approximate_q_agent():
    approx_agent = ApproximateQAgent(player_id=0, learning_rate=0.1, epsilon=0.1)
    random_opponent = AG.RandomPlayer(player_id=1)
    trainer = ApproximateQTrainer(approx_agent, random_opponent)
    trainer.train(episodes=2000, save_interval=100)
    return approx_agent

def train_dqn_agent(): # 这里用 id=1 训练了，记得保持一致
    dqn_agent = DQNAgent(
        player_id=1, 
        learning_rate=1e-4, 
        epsilon=0.9,
        epsilon_decay=0.995,
        batch_size=64
    )
    random_opponent = AG.RandomPlayer(player_id=0)
    trainer = DQNTrainer(dqn_agent, random_opponent)
    trainer.train(episodes=2000, save_interval=1000)    
    return dqn_agent

def play_with_trained_agent():
    rl_agent = QLearningAgent(player_id=1, epsilon=0.0)  # 测试时不探索
    rl_agent.load_q_table("model_data/final_q_table.pkl")
    
    game = HM.Game(rl_agent)
    game.run()
    
def play_with_approximate_agent():
    approx_agent = ApproximateQAgent(player_id=1, epsilon=0.0)  # 测试时不探索
    approx_agent.load_model("model_data/final_aq_model.pth")
    
    # 修改游戏设置
    game = HM.Game(approx_agent)
    game.run()
    
def play_with_dqn_agent():
    """与训练好的DQN智能体对战"""
    print("加载训练好的DQN智能体...")
    dqn_agent = DQNAgent(player_id=1, epsilon=0.0)  # 测试时不探索
    dqn_agent.load_model("final_dqn_model.pth")
    
    # 修改游戏设置
    game = HM.Game(dqn_agent)
    game.run()
    
def test_rl_agent():
    rl_agent = QLearningAgent(player_id=1, epsilon=0.0)  # 测试时不探索
    rl_agent.load_q_table("model_data/final_q_table.pkl")
    
    game = AG.Game(rl_agent)
    game.run()
    
def test_approximate_agent():
    approx_agent = ApproximateQAgent(player_id=1, epsilon=0.0)  # 测试时不探索
    approx_agent.load_model("model_data/final_aq_model.pth")
    
    game = AG.Game(approx_agent)
    game.run()
    
def evaluate_trained_agents():
    """评估训练好的智能体"""
    
    # 加载训练好的Q-Learning智能体
    rl_agent = QLearningAgent(player_id=0, epsilon=0.0)
    try:
        rl_agent.load_q_table("model_data/final_q_table.pkl")
        print("成功加载Q-Learning模型")
    except:
        print("Q-Learning模型加载失败, 使用未训练版本")
    
    # 加载训练好的Approximate Q智能体
    approx_agent = ApproximateQAgent(player_id=1, epsilon=0.0) # 训练的时候就记得用 id=1 训练
    try:
        approx_agent.load_model("model_data/final_aq_model.pkl")
        print("成功加载Approximate Q模型")
    except:
        print("Approximate Q模型加载失败, 使用未训练版本")
    
    # 创建随机对手
    random_agent_0 = AG.RandomPlayer(player_id=0)
    random_agent_1 = AG.RandomPlayer(player_id=1)
    
    # 1. Q-Learning vs Random
    print("\n1. Q-Learning vs Random:")
    compare_agents_performance(rl_agent, random_agent_1, 1000, "Q-Learning", "Random")
    
    # 2. Approximate Q vs Random
    print("\n2. Approximate Q vs Random:")
    compare_agents_performance(random_agent_0, approx_agent, 1000, "Approximate Q", "Random")
    
    # 3. Q-Learning vs Approximate Q
    print("\n3. Q-Learning vs Approximate Q:")
    compare_agents_performance(rl_agent, approx_agent, 1000, "Q-Learning", "Approximate Q")
    

if __name__ == "__main__":
    # train_rl_agent()
    train_approximate_q_agent()
    # train_dqn_agent()
    
    # play_with_trained_agent()
    
    # test_rl_agent()
    # test_approximate_agent()
    # test_dqn_agent()
    