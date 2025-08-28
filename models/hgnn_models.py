import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, global_mean_pool, global_max_pool
from torch_geometric.utils import dropout_adj


class HGCN(nn.Module):
    def __init__(self, dropout, in_channels=768, hidden_channels=512, num_classes=2, 
                 num_layers=3, use_residual=True, use_attention=True, use_layer_norm=True):
        super(HGCN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.use_layer_norm = use_layer_norm
        
        # Input projection layer
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Hypergraph convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(HypergraphConv(hidden_channels, hidden_channels, use_attention=False))
        
        # Normalization layers
        if use_layer_norm:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        else:
            self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        
        # Multi-scale pooling
        self.pool_layers = nn.ModuleList([
            nn.Linear(hidden_channels * 2, hidden_channels),  # mean + max pooling
            nn.Linear(hidden_channels, hidden_channels // 2)
        ])
        
        # Enhanced classifier with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_channels // 4, num_classes)
        )
        
        # Edge dropout for regularization
        self.edge_dropout = 0.1
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout * 0.2, training=self.training)
        
        # Store initial representation for skip connection
        x_initial = x
        
        # Apply edge dropout during training
        if self.training:
            edge_index, _ = dropout_adj(edge_index, p=self.edge_dropout, 
                                      force_undirected=False, num_nodes=x.size(0))
        
        # Apply hypergraph convolutions with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x
            
            # Hypergraph convolution - provide hyperedge_attr as None when not available
            try:
                x = conv(x, edge_index)
            except AssertionError:
                # If hyperedge_attr is required but not available, create dummy attr
                num_edges = edge_index.max().item() + 1 if edge_index.numel() > 0 else 0
                hyperedge_attr = torch.ones(num_edges, 1, device=x.device)
                x = conv(x, edge_index, hyperedge_attr)
            
            # Normalization
            if self.use_layer_norm:
                x = norm(x)
            else:
                x = norm(x)
            
            # Activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (skip every other layer for deeper networks)
            if self.use_residual and i > 0:
                x = x + x_res
        
        # Add final skip connection from input
        if self.use_residual:
            x = x + x_initial
        
        # Multi-scale global pooling
        if batch is not None:
            # Combine mean and max pooling for richer representation
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
            
            # Pool projection
            x = self.pool_layers[0](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout * 0.3, training=self.training)
            x = self.pool_layers[1](x)
        else:
            # Node-level prediction
            x = self.pool_layers[1](self.pool_layers[0](torch.cat([x, x], dim=1)))
        
        # Final classification
        x = self.classifier(x)
        return x
