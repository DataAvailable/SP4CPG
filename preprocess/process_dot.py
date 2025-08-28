import os
import re
import pydot
import torch
import html
from torch_geometric.data import Data
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

NODE_TYPE_MAP = {
    "METHOD": 0, "BLOCK": 1, "IDENTIFIER": 2, "LITERAL": 3,
    "CALL": 4, "METHOD_RETURN": 5, "CLASS": 6, "FILE": 7,
    "PARAM": 8, "LOCAL": 9, "FIELD_IDENTIFIER": 10, "MODIFIER": 11,
    "<operator>.assignment": 12, "<operator>.indirectFieldAccess": 13
}
EDGE_TYPE_MAP = {
    "AST": 0, "CFG": 1, "DDG": 2, "CDG": 3
}

class CodeGraphEncoder(nn.Module):
    """
    结合CodeBERT和图神经网络的编码器
    """
    def __init__(self, 
                 codebert_model_name='microsoft/codebert-base',
                 node_type_embed_dim=64,
                 edge_type_embed_dim=32,
                 hidden_dim=256):
        super().__init__()
        
        # CodeBERT用于编码代码文本
        self.tokenizer = AutoTokenizer.from_pretrained(codebert_model_name)
        self.codebert = AutoModel.from_pretrained(codebert_model_name)
        
        # 节点类型嵌入
        self.node_type_embedding = nn.Embedding(
            len(NODE_TYPE_MAP) + 1, node_type_embed_dim
        )
        
        # 边类型嵌入
        self.edge_type_embedding = nn.Embedding(
            len(EDGE_TYPE_MAP) + 1, edge_type_embed_dim
        )
        
        # 特征融合层
        codebert_dim = self.codebert.config.hidden_size  # 通常是768
        self.node_fusion = nn.Linear(
            codebert_dim + node_type_embed_dim, hidden_dim
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def encode_code_text(self, code_texts, max_length=128):
        """
        使用CodeBERT编码代码文本
        """
        if not code_texts:
            return torch.zeros(1, self.codebert.config.hidden_size)
            
        # 处理批量文本
        inputs = self.tokenizer(
            code_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # print(inputs)

        with torch.no_grad():
            outputs = self.codebert(**inputs)
            # 使用[CLS]token的表示
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return embeddings
    
    def forward(self, data):
        """
        前向传播
        """
        # 编码节点特征
        node_type_embeds = self.node_type_embedding(data.node_types)
        code_embeds = data.x  # 修改：直接使用x作为代码嵌入
        
        # 融合节点类型和代码嵌入
        node_features = torch.cat([node_type_embeds, code_embeds], dim=1)
        node_features = self.node_fusion(node_features)
        node_features = F.relu(node_features)
        
        # 编码边特征
        if data.edge_attr.numel() > 0:
            edge_embeds = self.edge_type_embedding(data.edge_attr.squeeze())
        else:
            edge_embeds = torch.empty(0, self.edge_type_embedding.embedding_dim)
            
        return node_features, edge_embeds

def parse_node_label(raw_label):
    """
    解析节点标签，提取节点类型和节点值
    """
    if not raw_label or len(raw_label) < 2:
        return "UNKNOWN", ""
        
    try:
        # 去掉最外层的引号和尖括号
        inner_content = raw_label.strip('"<>')
        
        # 按<BR/>分割
        parts = inner_content.split('<BR/>')
        if len(parts) >= 2:
            # 处理第一部分获取节点类型
            first_part = parts[0]
            node_type = first_part.split(',')[0].strip()
            
            # 处理第二部分获取的节点类型和节点值并还原转义字符
            node_type = html.unescape(node_type)
            node_value = html.unescape(parts[1])
            return node_type, node_value
        else:
            # 如果没有<BR/>，尝试其他格式
            if ',' in inner_content:
                node_type = inner_content.split(',')[0].strip()
                return node_type, inner_content
            else:
                return "UNKNOWN", inner_content
                
    except Exception as e:
        print(f"Error parsing label {raw_label}: {e}")
        return "UNKNOWN", ""

 
def parse_dot_file_enhanced(graph, label, encoder_model):
    """
    解析DOT文件并生成图数据
    """
    try:
        # graphs = pydot.graph_from_dot_file(dot_path)
        # if not graphs:
        #     raise ValueError(f"Cannot parse DOT file: {dot_path}")
        # graph = graphs[0]
        
        nodes = graph.get_nodes()
        edges = graph.get_edges()

        node_types = []
        code_texts = []
        node_id_map = {}

        # 解析节点
        for idx, node in enumerate(nodes):
            node_id = node.get_name().strip('"')
            attrs = node.get_attributes()
            raw_label = attrs.get("label", "")
            
            if not raw_label:
                node_type, code_text = "UNKNOWN", ""
            else:
                node_type, code_text = parse_node_label(raw_label)
            
            # print(f"Node {idx}: ID={node_id}, Type={node_type}, Code={code_text}")
            
            # 映射节点类型到数字
            type_id = NODE_TYPE_MAP.get(node_type, len(NODE_TYPE_MAP))
            node_types.append(type_id)
            code_texts.append(code_text if code_text else "")
            node_id_map[node_id] = idx

        # 如果没有节点，创建空图
        if not node_types:
            return Data(
                x=torch.zeros(1, 768),  # 空图的默认特征
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, 1, dtype=torch.long),
                y=torch.tensor([label], dtype=torch.long)
            )

        # 使用CodeBERT编码代码文本
        code_embeddings = encoder_model.encode_code_text(code_texts)

        # 解析边
        edge_index = []
        edge_attr = []

        for edge in edges:
            src = edge.get_source().strip('"')
            dst = edge.get_destination().strip('"')
            raw_edge_label = edge.get_attributes().get("label", "AST")
            
            # 提取边类型
            edge_type_str = raw_edge_label.split(":")[0].strip().strip('"')
            edge_type = EDGE_TYPE_MAP.get(edge_type_str, len(EDGE_TYPE_MAP))
            
            if src in node_id_map and dst in node_id_map:
                edge_index.append([node_id_map[src], node_id_map[dst]])
                edge_attr.append([edge_type])

        # 创建Data对象
        data = Data(
            x=code_embeddings,  # 修改：直接使用x存储代码嵌入
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty(2, 0, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.long) if edge_attr else torch.empty(0, 1, dtype=torch.long),
            y=torch.tensor([label], dtype=torch.long)
        )
        
        # 添加节点类型作为额外属性
        data.node_types = torch.tensor(node_types, dtype=torch.long)

        return data
        
    except Exception as e:
        print(f"Error parsing DOT file: {e}")
        # 返回空图
        return Data(
            x=torch.zeros(1, 768),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_attr=torch.empty(0, 1, dtype=torch.long),
            y=torch.tensor([label], dtype=torch.long)
        )

if __name__ == "__main__":
    # 初始化编码器
    print("Loading CodeBERT model...")
    encoder = CodeGraphEncoder()
    
    # 示例用法
    dot_file_path = "./dataset/8-cpg-1.dot"
    label = 1
    
    graphs = pydot.graph_from_dot_file(dot_file_path)
    if graphs is None:
        print(f"Failed to load graph from {dot_file_path}")
        exit(1)
    graph = graphs[0]
    
    print("Parsing DOT file...")
    data = parse_dot_file_enhanced(graph, label, encoder)
    print("Data structure:")
    print(f"Number of nodes: {data.x.size(0)}")
    print(f"Node features shape: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Edge attributes shape: {data.edge_attr.shape}")
    print(f"Label: {data.y}")
    print("Data parsed successfully!")