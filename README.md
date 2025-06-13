# CS181 Project - Simplified Jungle Chess AI Agents

## Introduction

This project implements a simplified version of Jungle Chess (Chinese: 斗兽棋) game environment and develops various reinforcement learning agents to learn game strategies. The project includes implementations of Q-Learning, Approximate Q-Learning, and Deep Q-Network (DQN) algorithms, supporting agent-vs-agent battles and human-AI interactions.

### Game Rules
- 7×8 board without traditional traps, rivers, and dens
- 8 pieces per player with strength from 1 (Mouse) to 8 (Elephant)
- Pieces start face-down and must be revealed through reveal actions
- Victory condition: Capture all opponent pieces

## Project Structure

```
CS181Project/
├── model_data/                     # Trained model files
│   ├── final_D3QNAgent.pkl        # DQN agent model
│   ├── final_ApproximateQAgent.pkl # Approximate Q-Learning model
│   └── final_QLearningAgent.pkl   # Q-Learning model
│
├── training_logs/                  # Training logs and history
│   ├── training_plots/            # Training progress charts
│   └── *.json, *.pkl              # Training data files
│
├── base.py                        # Game framework base classes
│   ├── Board                      # Board class
│   ├── Piece                      # Piece class
│   ├── Player                     # Player base class
│   ├── Game                       # Main game loop
│   └── BaseTrainer               # Trainer base class
│
├── utils.py                       # Utility functions and components
│   ├── RewardFunction            # Reward function
│   ├── FeatureExtractor          # Feature extractor
│   ├── PrioritizedReplayBuffer   # Prioritized experience replay
│   └── Data save/load functions
│
├── QlearningAgent.py             # Tabular Q-Learning agent
├── ApproximateQAgent.py          # Approximate Q-Learning agent
├── DQN.py                        # Deep Q-Network agent
│   ├── DQN/DoubleDQN/DuelingDQN  # Multiple network architectures
│   └── DQNTrainer                # DQN trainer
│
├── train_and_record.py           # Main training program
│   ├── Preset training configs
│   ├── Curriculum learning strategy
│   └── Command line interface
│
├── training_data_manager.py      # Training data manager
├── AgentFight.py                 # AI vs AI battle program
├── new_sim.py                    # Human vs AI interaction program
├── requirements.txt              # Dependencies list
└── README.md                     # Project documentation
```

## Agent Architectures

### 1. Q-Learning Agent (`QlearningAgent.py`)
- Table-based Q-Learning algorithm
- Supports phased training strategies
- Adaptive learning rate and ε-greedy policy

### 2. Approximate Q-Learning Agent (`ApproximateQAgent.py`)
- Uses SGD regressor for function approximation
- Feature engineering for game state representation
- Dynamic discount factor adjustment

### 3. Deep Q-Network Agent (`DQN.py`)
- Supports DQN, Double DQN, Dueling DQN variants
- Prioritized experience replay mechanism
- Guided exploration strategy optimization

## Usage

### 1. Training Agents

Use `train_and_record.py` for training:

```bash
# Train DQN agent (recommended configuration)
python train_and_record.py --agent dqn --retrain --episodes 4000 --opponent progressive --exploration guided --lr-strategy hybrid

# Train Q-Learning agent
python train_and_record.py --agent ql --retrain --episodes 3000 --opponent progressive --lr-strategy adaptive

# Train Approximate Q-Learning agent
python train_and_record.py --agent aq --retrain --episodes 2000 --opponent progressive --lr-strategy adaptive

# Use preset configuration for training (recommended)
python train_and_record.py --use-preset --preset-id 3
```

### 2. AI vs AI Battle Testing

Use `AgentFight.py` for AI battles:

```python
# Modify agent configuration in AgentFight.py
game = Game(
    agent=DQNAgent,      # First agent
    base_agent=RandomPlayer,  # Second agent
    display=True,        # Show game interface
    delay=1.0           # Action interval
)
game.run()
```

### 3. Human vs AI Interaction

Use `new_sim.py` for human-AI battles:

```python
# Run human vs AI game
python new_sim.py

# Or configure agent type in code
game = Game(
    agent=DQNAgent,     # AI agent
    test_mode=0         # 0: Human vs AI, 1: AI vs AI
)
game.run()
```

### 4. Test-Only Mode

```bash
# Test trained models
python train_and_record.py --agent dqn --test-only --test-games 100
```

You can set `DISPLAY_GAME = True` in `main` to watch the test process.

## Key Features

### 1. Curriculum Learning Training
- Phased training strategy: Basic Learning → Advanced Learning → Strategy Refinement
- Dynamic opponent difficulty adjustment: Random → Greedy → Minimax
- Adaptive hyperparameter tuning

### 2. Guided Exploration Strategy
- Heuristic-based exploration strategy
- Significantly improves learning efficiency compared to pure random exploration
- Avoids ineffective "random vs random" learning scenarios

### 3. Comprehensive Training Monitoring
- Real-time training progress visualization
- Detailed training log recording
- Multi-dimensional performance metrics tracking

### 4. Modular Design
- Unified agent and trainer interfaces
- Extensible reward function design
- Flexible opponent configuration system

## Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Pygame

## Training Recommendations

1. **DQN Agent**: Use guided exploration + curriculum learning, train for 4000+ episodes
2. **Q-Learning**: Good for quick validation, train for 3000 episodes
3. **Approximate Q-Learning**: Balances performance and speed, train for 3000 episodes

For optimal performance, we recommend using preset configuration ID=3 DQN training scheme.

## Training Methodology

Our DQN agent employs a two-phase curriculum learning approach:

**Phase 1 - Guided Exploration (Episodes 1-1,600)**: Uses heuristic-guided ε-greedy exploration with ε decaying from 0.9 to 0.02, providing structured learning experiences.

**Phase 2 - Random Exploration (Episodes 1,601-4,000)**: Switches to pure random exploration with ε decaying from 0.5 to 0.02, allowing discovery of novel strategies while maintaining learned coherence.

This approach addresses the fundamental challenge that DQN agents struggle to learn effective strategies when trained exclusively against random opponents, where both players make unpredictable moves resulting in noisy reward signals.