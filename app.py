import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
import json
import faiss
import os
import requests
import re
from pyvis.network import Network
import time
import pickle

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 耗时: {end_time - start_time:.2f} 秒")
        return result
    return wrapper

# 定义全局client
client = OpenAI(
    api_key="sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn",
    base_url="https://api.chatanywhere.tech/v1"
)

# 下载和准备数据
def prepare_data():
    index_root = os.path.join(os.getcwd(), 'graphrag_index')
    os.makedirs(os.path.join(index_root, 'input'), exist_ok=True)

    url = "https://www.gutenberg.org/cache/epub/7785/pg7785.txt"
    file_path = os.path.join(index_root, 'input', 'davinci.txt')

    response = requests.get(url, verify=True)
    if response.status_code == 200:
        text = response.text
        cleaned_text = clean_text(text)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)
        st.success(f"数据已下载、清理并准备完毕。文件保存在: {file_path}")
    else:
        st.error(f"下载失败，状态码: {response.status_code}")
        return None

    return file_path

def clean_text(text):
    # 移除版权信息和元数据
    text = re.sub(r'Project Gutenberg.*?\n\n', '', text, flags=re.DOTALL)
    text = re.sub(r'\*\*\*.*?\*\*\*', '', text, flags=re.DOTALL)
    # 移除空行和多余的空格
    text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
    return text

# 初始化模型和Faiss索引
@st.cache_resource
def init_models():
    dimension = 1536  # OpenAI's text-embedding-ada-002 model outputs 1536-dimensional vectors
    index = faiss.IndexFlatL2(dimension)
    return index

# 删除这些导入和函数
# import spacy
# @st.cache_resource
# def load_nlp_model():
#     return spacy.load("en_core_web_sm")
# nlp = load_nlp_model()

# 替换 extract_entities 函数
def extract_entities(text):
    # 这是一个非常简单的实体提取方法，仅作为示例
    return [word for word in text.split() if word[0].isupper()]

# 替换 extract_relation 函数
def extract_relation(sentence):
    words = sentence.split()
    if len(words) >= 3:
        return words[0], words[1], words[-1]
    return None, None, None

@time_function
def create_knowledge_graph(file_path):
    G = nx.Graph()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    sentences = text.split('.')
    for sentence in sentences:
        subject, relation, object = extract_relation(sentence)
        if subject and relation and object:
            G.add_edge(subject, object, relation=relation)
    
    # 打印一些图的信息，用于调试
    st.write(f"图中的节点数量: {G.number_of_nodes()}")
    st.write(f"图中的边数量: {G.number_of_edges()}")
    st.write("图中的一些边示例:")
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        if i >= 5:  # 只打印前5条边
            break
        st.write(f"Edge {i+1}: {u} -- {data.get('relation', 'is connected to')} --> {v}")
    
    return G

@time_function
def visualize_graph_interactive(G):
    net = Network(notebook=True, width="100%", height="500px", cdn_resources="remote")
    net.from_nx(G)
    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, width=700, height=500)

@time_function
def query_graph(G, query, index, nodes):
    query_embedding = create_embeddings([query])[0]
    _, I = index.search(np.array([query_embedding]).astype('float32'), k=10)  # 增加到10个最相关的节点
    
    relevant_nodes = []
    context = []
    for i in I[0]:
        if 0 <= i < len(nodes):
            node = nodes[i]
            relevant_nodes.append(node)
            neighbors = list(G.neighbors(node))
            edges = G.edges(node, data=True)
            relations = []
            for edge in edges:
                if len(edge) == 3:
                    neighbor, data = edge[1], edge[2]
                    relations.append(f"{node} {data.get('relation', 'is connected to')} {neighbor}")
                else:
                    neighbor = edge[1]
                    relations.append(f"{node} is connected to {neighbor}")
            context.append(f"节点 '{node}' 的关系: {'; '.join(relations)}")  # 保留所有关系
    
    # 返回最相关的5个节点的信息
    return "\n".join(context[:5])

@time_function
def generate_answer(query, context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手，专门回答关于莱昂纳多·达芬奇的问题。请根据给定的上下文，用中文提供全面的答案，涵盖达芬奇在艺术、科学和工程方面的成就。"},
            {"role": "user", "content": f"上下文: {context}\n\n问题: {query}\n\n请用中文提供一个详细的答案，涵盖莱昂纳多·达芬奇生平和工作的多个方面:"}
        ],
        max_tokens=1000  # 保持1000以获得完整的回答
    )
    return response.choices[0].message.content.strip()

# 新增：纯向量RAG方法
def vector_rag(text, query, index, sentences):
    query_embedding = create_embeddings([query])[0]
    _, I = index.search(np.array([query_embedding]).astype('float32'), k=5)
    
    context = [sentences[i] for i in I[0] if 0 <= i < len(sentences)]
    context_str = "\n".join(context)
    
    # 使用OpenAI模型进行翻译
    translation = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个翻译助手，请将给定的英文文本翻译成中文。"},
            {"role": "user", "content": f"请将以下文本翻译成中文：\n\n{context_str}"}
        ],
        max_tokens=1000
    )
    return translation.choices[0].message.content.strip()

def create_embeddings(texts, batch_size=100):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(input=batch, model="text-embedding-ada-002")
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
        except Exception as e:
            st.error(f"创建嵌入时出错: {str(e)}")
            return None
    return all_embeddings

def save_graph(G, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(G, f)

def load_graph(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_embeddings(embeddings, file_path):
    np.save(file_path, embeddings)

def load_embeddings(file_path):
    return np.load(file_path)

def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)

def load_faiss_index(file_path):
    return faiss.read_index(file_path)

# 在文件中添加这个新函数
def compare_methods(query, graph_context, graph_answer, vector_context, vector_answer):
    comparison_prompt = f"""
    请对以下两种RAG（检索增强生成）方法进行详细对比分析，并给出结论：

    查询: {query}

    方法1: GraphRAG
    上下文: {graph_context}
    生成的答案: {graph_answer}

    方法2: 纯向量RAG
    上下文: {vector_context}
    生成的答案: {vector_answer}

    请从以下几个方面进行分析：
    1. 上下文质量对比：评估两种方法提供的上下文的相关性、结构性和信息量。
    2. 答案质量对比：比较两种方法生成的答案的相关性、准确性、全面性和连贯性。
    3. 性能对比：讨论两种方法在处理这类查询时可能的优缺点。
    4. 总体评估：综合以上因素，给出哪种方法更适合处理这类查询的结论。

    请提供详细的分析和具体的例子来支持你的观点。
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个专门分析和比较不同RAG方法的AI助手。请基于给定的信息提供客观、详细的分析。"},
            {"role": "user", "content": comparison_prompt}
        ],
        max_tokens=1500
    )
    return response.choices[0].message.content.strip()

def main():
    st.title("GraphRAG vs 纯向量RAG 对比研究")
    
    # 准备数据
    file_path = prepare_data()
    if file_path is None:
        st.error("无法准备数据，请检查网络连接或URL。")
        return

    graph_file = 'knowledge_graph.gpickle'
    graph_embeddings_file = 'graph_embeddings.npy'
    graph_index_file = 'graph_index.faiss'
    vector_embeddings_file = 'vector_embeddings.npy'
    vector_index_file = 'vector_index.faiss'
    
    # 创建或加载知识图谱
    if os.path.exists(graph_file):
        G = load_graph(graph_file)
        st.success("已加载现有知识图谱")
    else:
        G = create_knowledge_graph(file_path)
        save_graph(G, graph_file)
        st.success("已创建并保存新的知识图谱")

    # 可视化知识图谱
    st.header("知识图谱")
    visualize_graph_interactive(G)

    # 准备或加载图嵌入和索引
    nodes = list(G.nodes())
    if os.path.exists(graph_embeddings_file) and os.path.exists(graph_index_file):
        graph_embeddings = load_embeddings(graph_embeddings_file)
        graph_index = load_faiss_index(graph_index_file)
        st.success("已加载图嵌入和索引")
    else:
        graph_embeddings = create_embeddings(nodes)
        if graph_embeddings is None:
            st.error("无法创建图节点的嵌入。")
            return
        graph_index = faiss.IndexFlatL2(len(graph_embeddings[0]))
        graph_index.add(np.array(graph_embeddings))
        save_embeddings(graph_embeddings, graph_embeddings_file)
        save_faiss_index(graph_index, graph_index_file)
        st.success("已创建并保存图嵌入和索引")

    # 准备或加载向量嵌入和索引
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if os.path.exists(vector_embeddings_file) and os.path.exists(vector_index_file):
        vector_embeddings = load_embeddings(vector_embeddings_file)
        vector_index = load_faiss_index(vector_index_file)
        st.success("已加载向量嵌入和索引")
    else:
        vector_embeddings = create_embeddings(sentences)
        if vector_embeddings is None:
            st.error("无法创建句子的嵌入。")
            return
        vector_index = faiss.IndexFlatL2(len(vector_embeddings[0]))
        vector_index.add(np.array(vector_embeddings))
        save_embeddings(vector_embeddings, vector_embeddings_file)
        save_faiss_index(vector_index, vector_index_file)
        st.success("已创建并保存向量嵌入和索引")

    # 查询部分
    st.header("RAG 查询")
    query = st.text_input("输入您的查询:")
    if query:
        # GraphRAG 方法
        graph_context = query_graph(G, query, graph_index, nodes)
        st.subheader("GraphRAG 结果")
        st.write("相关上下文:")
        st.write(graph_context)
        if graph_context != "无相关上下文":
            graph_answer = generate_answer(query, graph_context)
            st.write("生成的答案:", graph_answer)
        else:
            st.warning("GraphRAG: 由于没有找到相关上下文，无法生成答案。")
            graph_answer = "无法生成答案"
        
        # 纯向量RAG方法
        vector_context = vector_rag(text, query, vector_index, sentences)
        st.subheader("纯向量RAG 结果")
        st.write("相关上下文:")
        st.write(vector_context)
        vector_answer = generate_answer(query, vector_context)
        st.write("生成的答案:", vector_answer)
        
        # 方法对比
        st.header("方法对比")
        comparison_result = compare_methods(query, graph_context, graph_answer, vector_context, vector_answer)
        st.write(comparison_result)

if __name__ == "__main__":
    main()
