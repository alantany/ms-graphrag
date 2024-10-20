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
import PyPDF2
import io
import jieba
import string

# 定义全局client
client = OpenAI(
    api_key="sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn",
    base_url="https://api.chatanywhere.tech/v1"
)

@st.cache_data
def prepare_data():
    index_root = os.path.join(os.getcwd(), 'graphrag_index')
    file_path = os.path.join(index_root, 'input', 'davinci.txt')

    if os.path.exists(file_path):
        st.success(f"使用已存在的文件: {file_path}")
        return file_path

    os.makedirs(os.path.join(index_root, 'input'), exist_ok=True)

    url = "https://www.gutenberg.org/cache/epub/7785/pg7785.txt"

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
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

@st.cache_resource
def init_models():
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    dimension = embedding_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)
    return embedding_model, index

def create_knowledge_graph(file_path):
    G = nx.Graph()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    sentences = re.split(r'[.!?]', text)
    for sentence in sentences:
        # 移除括号内的内容
        sentence = re.sub(r'\([^()]*\)', '', sentence)
        # 移除所有标点符号
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        words = sentence.strip().split()
        if len(words) > 2:
            # 只使用字母数字的单词作为节点，并且长度大于1
            clean_words = [word for word in words if word.isalnum() and len(word) > 1]
            if len(clean_words) > 2:
                # 使用前两个单词作为节点，整个句子作为关系
                G.add_edge(clean_words[0], clean_words[1], relation=' '.join(clean_words))
    
    # 确保图中至少有一个节点
    if len(G) == 0:
        G.add_node("Empty")  # 添加一个默认节点
    
    return G

def visualize_graph_interactive(G):
    net = Network(notebook=True, width="100%", height="500px", cdn_resources="remote")
    net.from_nx(G)
    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, width=700, height=500)

def query_graph(G, query, embedding_model, index, nodes):
    query_embedding = embedding_model.encode([query])[0]
    _, I = index.search(np.array([query_embedding]).astype('float32'), k=5)
    
    relevant_nodes = [nodes[i] for i in I[0] if 0 <= i < len(nodes)]
    context = []
    for node in relevant_nodes:
        neighbors = list(G.neighbors(node))
        edges = G.edges(node, data=True)
        relations = []
        for edge in edges:
            if len(edge) == 3:  # 如果边包含数据
                neighbor, data = edge[1], edge[2]
                if neighbor != node:  # 避免自环
                    relations.append(f"{node} 与 {neighbor} 的关系: {data.get('relation', '相关')}")
        if relations:
            context.append(f"节点 '{node}' 的关系:\n" + "\n".join(relations))
    
    return "\n\n".join(context)

def generate_answer(query, context):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        max_tokens=300  # 增加到300或更多
    )
    return response.choices[0].message.content.strip()

def vector_rag(text, query, embedding_model, index, sentences):
    query_embedding = embedding_model.encode([query])[0]
    _, I = index.search(np.array([query_embedding]).astype('float32'), k=5)
    
    context = [sentences[i] for i in I[0] if 0 <= i < len(sentences)]
    return "\n".join(context)

def process_query(query, G, embedding_model, index, nodes, sentences, text):
    graph_context = query_graph(G, query, embedding_model, index, nodes)
    vector_context = vector_rag(text, query, embedding_model, index, sentences)
    
    graph_answer = generate_answer(query, graph_context)
    vector_answer = generate_answer(query, vector_context)
    
    comparison = compare_results(query, graph_context, graph_answer, vector_context, vector_answer)
    
    return graph_context, graph_answer, vector_context, vector_answer, comparison

def on_query_submit():
    query = st.session_state.query_input
    if query:
        graph_context, graph_answer, vector_context, vector_answer, comparison = process_query(
            query, st.session_state.G, st.session_state.embedding_model, 
            st.session_state.index, st.session_state.nodes, 
            st.session_state.sentences, st.session_state.text
        )
        st.session_state.last_query = query
        st.session_state.graph_context = graph_context
        st.session_state.graph_answer = graph_answer
        st.session_state.vector_context = vector_context
        st.session_state.vector_answer = vector_answer

def compare_results(query, graph_context, graph_answer, vector_context, vector_answer):
    comparison_prompt = f"""
    请比较以下两种RAG方法的结果，并给出评价：

    查询: {query}

    GraphRAG 结果:
    上下文: {graph_context}
    答案: {graph_answer}

    纯向量RAG 结果:
    上下文: {vector_context}
    答案: {vector_answer}

    请从以下几个方面进行分析：
    1. 上下文相关性：哪种方法提供的上下文更相关？
    2. 答案质量：哪种方法的答案更准确、全面？
    3. 信息丰富度：哪种方法提供了更多有用的信息？
    4. 整体表现：综合考虑，哪种方法在回答这个问题时表现更好？

    请给出详细的分析和解释。
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个专门分析和比较不同RAG方法的AI助手。"},
            {"role": "user", "content": comparison_prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

def main():
    st.title("GraphRAG vs 纯向量RAG 对比研究")
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    tab1, tab2 = st.tabs(["PDF分析", "演示数据"])
    
    with tab1:
        st.header("电子病历PDF分析")
        st.info("PDF分析功能暂未实现")
        
        # PDF分析的示例问题
        with st.sidebar:
            st.header("PDF分析示例问题")
            with st.expander("点击展开示例问题"):
                pdf_questions = [
                    "患者的主要症状是什么？",
                    "患者的诊断结果是什么？",
                    "医生建议的治疗方案是什么？",
                    "患者的病史中有哪些重要信息？",
                    "患者的用药情况如何？"
                ]
                for q in pdf_questions:
                    st.markdown(f"- {q}")
    
    with tab2:
        st.header("演示数据分析")
        
        # 演示数据的示例问题
        with st.sidebar:
            st.header("达芬奇示例问题")
            with st.expander("点击展开示例问题"):
                davinci_questions = [
                    "Who was Leonardo da Vinci and what was he known for? (莱昂纳多·达芬奇是谁，他因什么而闻名？)",
                    "What were some of Leonardo da Vinci's most famous artworks? (莱昂纳多·达芬奇最著名的艺术作品有哪些？)",
                    "How did Leonardo da Vinci contribute to the field of science? (莱昂纳多·达芬奇对科学领域有哪些贡献？)",
                    "What was Leonardo da Vinci's relationship with his patrons? (莱昂纳多·达芬奇与他的赞助人之间的关系如何？)",
                    "Can you describe Leonardo da Vinci's approach to art and science? (你能描述一下莱昂纳多·达芬奇对艺术和科学的方法吗？)",
                    "What was Leonardo da Vinci's early life and training like? (莱昂纳多·达芬奇的早期生活和训练是怎样的？)",
                    "How did Leonardo da Vinci's work influence later generations? (莱昂纳多·达芬奇的作品如何影响了后代？)"
                ]
                for q in davinci_questions:
                    st.markdown(f"- {q}")
        
        load_button = st.empty()
        
        if not st.session_state.data_loaded:
            if load_button.button("加载演示数据"):
                file_path = prepare_data()
                if file_path is None:
                    st.error("无法准备数据，请检查网络连接或URL。")
                else:
                    st.success("演示数据已加载")
                    st.session_state.data_loaded = True
                    
                    embedding_model, index = init_models()
                    
                    G = create_knowledge_graph(file_path)
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                    sentences = [s.strip() for s in text.split('.') if s.strip()]
                    
                    nodes = list(G.nodes())
                    node_embeddings = embedding_model.encode(nodes)
                    index.add(np.array(node_embeddings))
                    
                    st.session_state.G = G
                    st.session_state.embedding_model = embedding_model
                    st.session_state.index = index
                    st.session_state.nodes = nodes
                    st.session_state.sentences = sentences
                    st.session_state.text = text
                    
                    st.rerun()
        
        if st.session_state.data_loaded:
            load_button.empty()
            
            # 固定的知识图谱区域
            st.subheader("知识图谱")
            graph_container = st.container()
            with graph_container:
                visualize_graph_interactive(st.session_state.G)
            
            # 查询输入区域
            st.subheader("RAG 查询")
            query_input = st.text_input("输入您的查询:", key="query_input")
            
            # 结果显示区域
            results_container = st.container()
            
            if query_input:
                graph_context, graph_answer, vector_context, vector_answer, comparison = process_query(
                    query_input, st.session_state.G, st.session_state.embedding_model, 
                    st.session_state.index, st.session_state.nodes, 
                    st.session_state.sentences, st.session_state.text
                )
                
                with results_container:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("GraphRAG 结果:")
                        st.write(graph_context)
                        st.write("生成的答案:", graph_answer)
                    
                    with col2:
                        st.write("纯向量RAG 结果:")
                        st.write(vector_context)
                        st.write("生成的答案:", vector_answer)
                    
                    st.subheader("结果比较")
                    st.write(comparison)

if __name__ == "__main__":
    main()
