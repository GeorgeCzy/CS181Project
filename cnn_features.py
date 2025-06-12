import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Tuple, Optional
from base import Board

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChessFeatureNet(nn.Module):
    """优化的斗兽棋CNN特征提取网络"""
    
    def __init__(self, output_size: int = 64):
        super(ChessFeatureNet, self).__init__()
        
        # 多通道输入处理 - 参考我之前的8通道设计
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 使用预训练的ResNet18 - 保持您的设计
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # 移除最大池化，保持尺寸
        
        # 修改最后的全连接层
        self.resnet.fc = nn.Linear(512, output_size)
        
        # 位置价值评估头 - 添加空间理解能力
        self.position_head = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),  # 从ResNet的feature map
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 全局特征提取
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, output_size),
            nn.ReLU()
        )
    
    def forward(self, board_tensor: torch.Tensor, extra_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            board_tensor: [batch_size, 8, 7, 8] 多通道棋盘状态
            extra_features: [batch_size, 2] 额外特征
        Returns:
            global_features: [batch_size, 64] 全局特征
            position_values: [batch_size, 1, 7, 8] 位置价值图
        """
        # 输入卷积处理
        x = self.input_conv(board_tensor)
        
        # ResNet特征提取，但获取中间feature map
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        feature_map = self.resnet.layer4(x)  # [batch, 512, H, W]
        
        # 全局特征
        global_features = self.global_head(feature_map)
        
        # 位置价值
        position_values = self.position_head(feature_map)
        
        return global_features, position_values

class CNNEnhancedFeatureExtractor:
    """增强的CNN特征提取器 - 优化版本"""
    
    def __init__(self, model_path: str = None):
        self.feature_net = ChessFeatureNet().to(device)
        
        if model_path:
            self.load_models(model_path)
        
        self.feature_net.eval()
        
        # 棋子价值映射
        self.piece_values = {1: 1.8, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5, 6: 3.0, 7: 3.5, 8: 4.0}
    
    def board_to_tensor(self, board: Board, player_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """将棋盘转换为多通道张量 - 使用更丰富的表示"""
        # 8通道表示
        channels = np.zeros((8, 7, 8), dtype=np.float32)
        
        for r in range(7):
            for c in range(8):
                piece = board.get_piece(r, c)
                if piece:
                    # 通道0-1: 玩家0和玩家1的棋子位置
                    channels[piece.player, r, c] = 1.0
                    
                    # 通道2: 棋子强度（归一化）
                    channels[2, r, c] = piece.strength / 8.0
                    
                    # 通道3: 是否已翻开
                    channels[3, r, c] = 1.0 if piece.revealed else 0.0
                    
                    # 通道4: 己方棋子标记
                    if piece.player == player_id:
                        channels[4, r, c] = 1.0
                    
                    # 通道5: 对方棋子标记
                    if piece.player != player_id:
                        channels[5, r, c] = 1.0
                    
                    # 通道6: 棋子价值权重
                    if piece.revealed:
                        channels[6, r, c] = self.piece_values[piece.strength] / 4.0
                    
                    # 通道7: 威胁/机会标记
                    if piece.revealed and piece.player == player_id:
                        # 检查是否受威胁或有攻击机会
                        threat_level = self._assess_position_threat(board, r, c, player_id)
                        channels[7, r, c] = threat_level
        
        board_tensor = torch.FloatTensor(channels).unsqueeze(0).to(device)
        
        # 额外特征
        my_pieces = len(board.get_player_pieces(player_id))
        enemy_pieces = len(board.get_player_pieces(1 - player_id))
        game_phase = my_pieces / 16.0  # 游戏阶段
        
        extra_features = torch.FloatTensor([[player_id, game_phase]]).to(device)
        
        return board_tensor, extra_features
    
    def _assess_position_threat(self, board: Board, r: int, c: int, player_id: int) -> float:
        """评估位置的威胁/机会水平"""
        piece = board.get_piece(r, c)
        if not piece or not piece.revealed:
            return 0.0
        
        threat_score = 0.0
        opportunity_score = 0.0
        
        # 检查四个方向的邻居
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 7 and 0 <= nc < 8:
                neighbor = board.get_piece(nr, nc)
                if neighbor and neighbor.revealed and neighbor.player != player_id:
                    compare = piece.compare_strength(neighbor)
                    if compare == -1:  # 受威胁
                        threat_score -= 0.5
                    elif compare == 1:  # 有机会
                        opportunity_score += 0.5
        
        return threat_score + opportunity_score
    
    def extract_cnn_features(self, board: Board, player_id: int) -> np.ndarray:
        """提取CNN特征 - 返回64维特征向量"""
        with torch.no_grad():
            board_tensor, extra_features = self.board_to_tensor(board, player_id)
            global_features, position_values = self.feature_net(board_tensor, extra_features)
            return global_features.cpu().numpy().flatten()
    
    def evaluate_position(self, board: Board, player_id: int) -> float:
        """使用CNN评估位置价值"""
        with torch.no_grad():
            board_tensor, extra_features = self.board_to_tensor(board, player_id)
            global_features, position_values = self.feature_net(board_tensor, extra_features)
            
            # 计算己方棋子位置的平均价值
            my_pieces = board.get_player_pieces(player_id)
            total_value = 0.0
            count = 0
            
            for r, c in my_pieces:
                piece = board.get_piece(r, c)
                if piece:
                    pos_value = position_values[0, 0, r, c].item()
                    if piece.revealed:
                        piece_weight = self.piece_values[piece.strength]
                        total_value += pos_value * piece_weight
                        count += 1
                    else:
                        total_value += pos_value * 2.5  # 平均价值
                        count += 1
            
            if count > 0:
                return total_value / count
            else:
                return 0.0
    
    def save_models(self, path: str):
        """保存模型"""
        torch.save({
            'feature_net': self.feature_net.state_dict(),
        }, path)
        print(f"CNN模型已保存: {path}")
    
    def load_models(self, path: str):
        """加载模型"""
        try:
            checkpoint = torch.load(path, map_location=device)
            self.feature_net.load_state_dict(checkpoint['feature_net'])
            print(f"CNN模型加载成功: {path}")
        except Exception as e:
            print(f"CNN模型加载失败: {path}，使用随机初始化 - {e}")