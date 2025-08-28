import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, GatedGraphConv, global_mean_pool
from torch_geometric.data import Batch

class GCN(nn.Module):
    def __init__(self, dropout, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        # 图级别预测：使用全局平均池化
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
            
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, dropout, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=1, dropout=dropout)
        self.classifier = nn.Linear(hidden_channels, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        # 图级别预测：使用全局平均池化
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
            
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class GIN(nn.Module):
    def __init__(self, dropout, in_channels, hidden_channels):
        super().__init__()
        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), 
            nn.ReLU(), 
            nn.Linear(hidden_channels, hidden_channels)
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), 
            nn.ReLU(), 
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.classifier = nn.Linear(hidden_channels, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        # 图级别预测：使用全局平均池化
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
            
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class GraphSAGE(nn.Module):
    def __init__(self, dropout, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        # 图级别预测：使用全局平均池化
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
            
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class GGNN(nn.Module):
    def __init__(self, dropout, in_channels, out_channels, num_layers=3):
        super().__init__()
        # 添加输入投影层，将输入特征映射到GGNN的隐藏维度
        self.input_projection = nn.Linear(in_channels, out_channels)
        self.ggnn = GatedGraphConv(out_channels, num_layers)
        self.classifier = nn.Linear(out_channels, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        # 投影输入特征
        x = self.input_projection(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GGNN层
        x = self.ggnn(x, edge_index)
        x = self.dropout(x)
        
        # 图级别预测：使用全局平均池化
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
            
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)