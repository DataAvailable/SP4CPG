from torch_geometric.data import Dataset
from process_dot import parse_dot_file_enhanced, CodeGraphEncoder
import os
import torch
import re
import pydot
from typing import List, Set
from tqdm.auto import tqdm

class CPGDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.dot_dir = os.path.join("dots")
        
        # 检查目录是否存在
        if not os.path.exists(self.dot_dir):
            raise ValueError(f"Dataset directory not found: {self.dot_dir}")
        
        dot_files = os.listdir(self.dot_dir)
        self.dot_files = sorted(dot_files, key=lambda x: int(re.search(r'(\d+)', x).group()))

        self.encoder = CodeGraphEncoder()

        # 解析标签
        self.labels = []
        for file_path in self.dot_files:
            # 获取文件名
            file_name = os.path.basename(file_path)
            # 从文件名中提取标签, 文件名格式为: "序号-cpg-标签.dot"
            try:
                # 去掉扩展名
                name_without_ext = os.path.splitext(file_name)[0]
                
                # 按"-"分割并获取最后一部分作为标签
                parts = name_without_ext.split("-")
                if len(parts) >= 3:  # 确保格式正确
                    label = int(parts[-1])  # 转换为整数
                    self.labels.append(label)
                else:
                    print(f"警告: 文件名格式不符合预期 - {file_name}")
            except ValueError:
                print(f"警告: 无法从文件名中提取标签 - {file_name}")
                
        # 初始化编码器（只初始化一次）
        # print("Loading CodeBERT model...")
        # print("CodeBERT model loaded successfully!")


    def encode_dot_to_feature(self):
        """
        解析DOT文件并转换为图数据
        :param dot_path: DOT文件路径
        :param label: 标签
        :return: 图数据对象
        """
        dataset = []  # 用于存储解析后的图数据
        # 遍历所有DOT文件
        for dot_file_path, dot_file_label in zip(self.dot_files, self.labels):
            path = os.path.join(self.dot_dir, dot_file_path)
            label = dot_file_label
            print("--------------------------------------------------")
            print(f"Processing file: {path} - Label: {label}")
            
            # 读取DOT文件为图数据对象
            graphs = pydot.graph_from_dot_file(path)
            if not graphs:
                print(f"警告: 无法读取DOT文件 {path}，跳过该文件。")
                continue
            graph = graphs[0]  # 取第一个图（通常只有一个图）
            # print(f"初始节点数: {len(graph.get_nodes())}")
            # print(f"初始边数: {len(graph.get_edges())}")
            
            # 1.中间节点裁剪
            graph = remove_immediate_assignment_and_print_nodes(graph)
            # 2.叶子节点裁剪
            graph = remove_and_merge_ast_leaf_nodes(graph)
            
            # 解析DOT文件
            data = parse_dot_file_enhanced(graph, label, self.encoder)
            dataset.append(data)

        return dataset

def is_ast_leaf_node(graph: pydot.Dot, node_name: str, node) -> bool:
    """
    判断一个节点是否是叶子节点, 且与其父节点之间仅存在 AST 边
    
    Args:
        graph: pydot.Dot 对象
        node_name: 节点名（字符串）
    
    Returns:
        bool: 满足条件返回 True, 否则 False
    """
    
    # 检查是否为叶子节点（没有出边）
    out_edges = []
    for edge in graph.get_edges():
        if edge.get_source() == node_name:
            out_edges.append(edge)
    
    if len(out_edges) > 0:
        return False
    
    # 获取所有入边
    in_edges = []
    for edge in graph.get_edges():
        if edge.get_destination() == node_name:
            in_edges.append(edge)
    
    # 如果没有入边，也不是AST叶子节点
    if len(in_edges) == 0:
        return False
    
    # 检查所有入边是否都是 AST 类型
    for edge in in_edges:
        label = edge.get_label()
        if not label or "AST: " not in str(label):
            return False
    
    return True

def remove_and_merge_ast_leaf_nodes(graph: pydot.Dot) -> pydot.Dot:
    """
    1.删除图中所有"与父节点仅存在 AST 边, 节点类型为 LITERAL 或 IDENTIFIER, 且自身为叶子节点"的节点和相关边 (规则3)
    2.(1)合并所有节点类型为 LOCAL/PARAM, 节点内容为同数据类型定义(如 char *ptr: char* 和 char *leak: char*)的节点, 并将这些节点的入边指向新合并的节点 (规则4) 
      (2)如果存在多个相同类型(如 char* )的 TYPE_REF/METHOD_RETURN 节点, 则仅保留一个节点, 删除的节点的入边指向保留的节点 (规则4)
    Args:
        graph: pydot.Dot 对象（将被修改）
    
    Returns:
        pydot.Dot: 修改后的图对象
    """
    
    # 收集要删除的节点
    nodes_to_delete = []
    
    # 收集要合并的节点集
    local_nodes_to_merge = []
    param_nodes_to_merge = []
    typeref_nodes_to_merge = []
    mreturn_nodes_to_merge = []
    
    for node in graph.get_nodes():
        node_name = node.get_name()
        if node_name == 'node':  # 跳过默认属性节点
            continue

        if is_ast_leaf_node(graph, node_name, node):
            # 检查节点类型是否为 LITERAL 或 IDENTIFIER
            label = node.get_label()
            node_type = str(label).split(",")[0].strip('<') # 获取节点类型
            if (node_type.startswith("LITERAL") or node_type.startswith("IDENTIFIER")):
            # print(f"Checking node_type: {node_type}")
                nodes_to_delete.append(node_name)
            elif node_type.startswith("LOCAL"):
                local_nodes_to_merge.append(node_name)
            elif node_type.startswith("PARAM"):
                param_nodes_to_merge.append(node_name)
            elif node_type.startswith("TYPE_REF"):
                typeref_nodes_to_merge.append(node_name)
            elif node_type.startswith("METHOD_RETURN"):
                mreturn_nodes_to_merge.append(node_name)
    
    # print(f"找到 {len(nodes_to_delete)} 个满足裁剪要求的AST叶子节点")
    # print(f"满足要求的节点: {nodes_to_delete}")
    
    # 删除叶子节点和相关边
    for node_name in nodes_to_delete:
        # print(f"删除节点: {node_name}")
        
        # 删除相关的边
        edges_to_remove = []
        for edge in graph.get_edges():
            if (edge.get_source() == node_name or 
                edge.get_destination() == node_name):
                edges_to_remove.append(edge)
        
        for edge in edges_to_remove:
            graph.del_edge(edge.get_source(), edge.get_destination())
        
        # 删除节点
        for node in graph.get_nodes():
            if node.get_name() == node_name:
                graph.del_node(node.get_name())
                break
    
    # 辅助函数：提取数据类型定义
    def extract_data_type(label_str):
        """从节点标签中提取数据类型定义"""
        # 示例: "LOCAL,char *ptr" -> "char*"
        # 示例: "PARAM,int value" -> "int"
        if ',' in label_str:
            parts = label_str.split(',', 1)
            if len(parts) > 1:
                var_def = parts[1].strip()
                # 提取类型部分 (变量名之前的部分)
                # 例如: "char *ptr" -> "char*", "int value" -> "int"
                tokens = var_def.split()
                if len(tokens) >= 2:
                    # 假设最后一个token是变量名，前面的是类型
                    type_tokens = tokens[:-1]
                    return ''.join(type_tokens).replace(' ', '')
                elif len(tokens) == 1:
                    # 只有一个token，可能是简单类型
                    return tokens[0]
        return None
    
    # 辅助函数：合并同类型节点
    def merge_nodes_by_type(nodes_list, node_type_prefix):
        """合并具有相同数据类型的节点"""
        if not nodes_list:
            return
            
        # 按数据类型分组
        type_groups = {}
        node_info = {}  # 存储节点信息
        
        for node_name in nodes_list:
            # 获取节点信息
            for node in graph.get_nodes():
                if node.get_name() == node_name:
                    label = str(node.get_label()).strip('"')
                    node_info[node_name] = (node, label)
                    
                    data_type = extract_data_type(label)
                    if data_type:
                        if data_type not in type_groups:
                            type_groups[data_type] = []
                        type_groups[data_type].append(node_name)
                    break
        
        # 对每个类型组进行合并
        for data_type, nodes_in_group in type_groups.items():
            if len(nodes_in_group) <= 1:
                continue  # 只有一个节点，无需合并
                
            # 保留第一个节点作为代表节点
            representative_node = nodes_in_group[0]
            nodes_to_merge = nodes_in_group[1:]
            
            # 将被合并节点的所有入边重定向到代表节点
            for node_to_merge in nodes_to_merge:
                # 收集入边
                incoming_edges = []
                for edge in graph.get_edges():
                    if edge.get_destination() == node_to_merge:
                        incoming_edges.append(edge)
                
                # 重定向入边到代表节点
                for edge in incoming_edges:
                    source = edge.get_source()
                    edge_label = edge.get_label()
                    edge_attrs = edge.get_attributes()
                    
                    # 删除原边
                    graph.del_edge(edge.get_source(), edge.get_destination())
                    
                    # 创建新边指向代表节点
                    new_edge = pydot.Edge(source, representative_node)
                    if edge_label:
                        new_edge.set_label(edge_label)
                    for attr, value in edge_attrs.items():
                        new_edge.set(attr, value)
                    graph.add_edge(new_edge)
                
                # 删除被合并的节点
                graph.del_node(node_to_merge)
    
    # 辅助函数：合并相同类型的TYPE_REF和METHOD_RETURN节点
    def merge_same_type_nodes(nodes_list):
        """合并相同类型的节点（用于TYPE_REF和METHOD_RETURN）"""
        if not nodes_list:
            return
            
        # 按节点标签内容分组
        label_groups = {}
        
        for node_name in nodes_list:
            for node in graph.get_nodes():
                if node.get_name() == node_name:
                    label = str(node.get_label()).strip('"')
                    if label not in label_groups:
                        label_groups[label] = []
                    label_groups[label].append(node_name)
                    break
        
        # 对每个标签组进行合并
        for label_content, nodes_in_group in label_groups.items():
            if len(nodes_in_group) <= 1:
                continue  # 只有一个节点，无需合并
                
            # 保留第一个节点作为代表节点
            representative_node = nodes_in_group[0]
            nodes_to_merge = nodes_in_group[1:]
            
            # 将被合并节点的所有入边重定向到代表节点
            for node_to_merge in nodes_to_merge:
                # 收集入边
                incoming_edges = []
                for edge in graph.get_edges():
                    if edge.get_destination() == node_to_merge:
                        incoming_edges.append(edge)
                
                # 重定向入边到代表节点
                for edge in incoming_edges:
                    source = edge.get_source()
                    edge_label = edge.get_label()
                    edge_attrs = edge.get_attributes()
                    
                    # 删除原边
                    graph.del_edge(edge.get_source(), edge.get_destination())
                    
                    # 创建新边指向代表节点
                    new_edge = pydot.Edge(source, representative_node)
                    if edge_label:
                        new_edge.set_label(edge_label)
                    for attr, value in edge_attrs.items():
                        new_edge.set(attr, value)
                    graph.add_edge(new_edge)
                
                # 删除被合并的节点
                graph.del_node(node_to_merge)
    
    # 合并所有具备相同数据类型定义的 LOCAL 节点
    merge_nodes_by_type(local_nodes_to_merge, "LOCAL")
    
    # 合并所有具备相同数据类型定义的 PARAM 节点
    merge_nodes_by_type(param_nodes_to_merge, "PARAM")
    
    # 存在多个相同类型的 TYPE_REF 节点, 保留一个
    merge_same_type_nodes(typeref_nodes_to_merge)
    
    # 存在多个相同类型的 METHOD_RETURN 节点, 保留一个
    merge_same_type_nodes(mreturn_nodes_to_merge)

    # print(f"删除了 {len(nodes_to_delete)} 个AST叶子节点")
    return graph

def remove_immediate_assignment_and_print_nodes(graph: pydot.Dot) -> pydot.Dot:
    """
    1.删除节点内容同时在其父子节点中出现的中间赋值节点
    2.删除中间节点中的常量 print 节点
    
    Args:
        graph: pydot.Dot 对象（将被修改）
    
    Returns:
        pydot.Dot: 修改后的图对象
    """
    # 收集要删除的节点
    nodes_to_remove = []

    for node in graph.get_nodes():  # node: "111669149696" [label = <METHOD, 1<BR/>&lt;global&gt;> ]
        node_name = node.get_name() # 获取节点序号名: node_name: "111669149696"
        label = node.get_label() # 获取节点label中的值 label: <METHOD, 1<BR/>&lt;global&gt;>
        node_type = str(label).split(",")[0].strip('<') # 获取节点类型 node_type: METHOD
        if node_type.startswith("&lt;operator&gt;.assignment"):
            #print(f"Processing node: {node.get_name()}")
            parent_nodes = get_parent_nodes(graph, node_name) # 获取父节点 列表
            # print(f"Parent nodes: {parent_nodes}")
            child_nodes = get_child_nodes(graph, node_name) # 获取子节点 列表
            # print(f"Child nodes: {child_nodes}")
            # 如果父子节点中都包含该节点，则删除该节点
            if len(parent_nodes) > 0 and len(child_nodes) > 0:
                for parent in parent_nodes:
                    parent_node = graph.get_node(parent)[0]  # 获取父节点对象
                    # print(f"Parent node: {parent_node}")
                    if str(parent_node.get_label()).split(",")[0].strip('<').startswith("BLOCK"):   # 判断父节点类型是否为 BLOCK
                        for child in child_nodes:
                            child_node = graph.get_node(child)[0] # 获取子节点对象
                            if str(child_node.get_label()).split(",")[0].strip('<').startswith("&lt;operator&gt;.alloc"): # 判断子节点类型是否为 alloc
                                nodes_to_remove.append(node)
        elif node_type.startswith("printf") or node_type.startswith("sprintf") or node_type.startswith("fprintf") or node_type.startswith("snprintf") or node_type.startswith("vprintf") or node_type.startswith("vsnprintf"):
            nodes_to_remove.append(node)  # 收集 printf 节点
                                
    # 执行节点删除操作
    for node in nodes_to_remove:
        node_name = node.get_name()
        node_type = str(node.get_label()).split(",")[0].strip('<')  # 获取节点类型
        # print(f"Removing node: {node_name}")
        parent_nodes = get_parent_nodes(graph, node_name) # 获取目标节点的 父节点 列表
        child_nodes = get_child_nodes(graph, node_name) # 获取目标节点的 子节点 列表

        # 删除该节点
        graph.del_node(node_name)
        # print(f"Removing node: {node_name}, {node}")
        
        # 移除涉及该节点的所有边
        edges_to_remove = []
        for edge in graph.get_edges():
            if (edge.get_source() == node_name or 
                edge.get_destination() == node_name):
                edges_to_remove.append(edge)
        for edge in edges_to_remove:
            graph.del_edge(edge.get_source(), edge.get_destination())
            # print(f"Removing edge: {edge.get_source()} -> {edge.get_destination()}")
        
        # 创建新边连接父节点和子节点
        if node_type.startswith("&lt;operator&gt;.assignment"):
            for parent in parent_nodes:
                parent_node = graph.get_node(parent)[0]  # 获取父节点对象
                for child in child_nodes:
                    child_node = graph.get_node(child)[0] # 获取子节点对象
                    if str(child_node.get_label()).split(",")[0].strip('<').startswith("&lt;operator&gt;.alloc"): # 判断子节点类型是否为 alloc
                        graph.add_edge(pydot.Edge(parent, child, label="AST: "))
                        # print(f"Adding new edge: {parent} -> {child}")
                    else:
                        if str(parent_node.get_label()).split(",")[0].strip('<').startswith("BLOCK"):
                            graph.add_edge(pydot.Edge(parent, child, label="AST: "))
                            # print(f"Adding new edge: {parent} -> {child}")

        # printf类型的节点和相连的边直接删除，无需重建边

    return graph

def get_parent_nodes(graph: pydot.Dot, node_name: str):
        """获取节点的所有父节点"""
        parents = []
        for edge in graph.get_edges():
            if edge.get_destination() == node_name:
                parents.append(edge.get_source())
        return parents

def get_child_nodes(graph: pydot.Dot, node_name: str):
        """获取节点的所有子节点"""
        children = []
        for edge in graph.get_edges():
            if edge.get_source() == node_name:
                children.append(edge.get_destination())
        return children

if __name__ == "__main__":
    # 加载 dot 文件数据集
    print("Loading CPG dataset...")
    CPGDataset_obj = CPGDataset()
    dataset = CPGDataset_obj.encode_dot_to_feature()
    torch.save(dataset, 'pruned_cpg_dataset.pkl')
    print("CPG dataset saved to 'pruned_cpg_dataset.pkl'")