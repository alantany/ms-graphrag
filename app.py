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
from pyvis.options import Options
import time
import pickle
import PyPDF2
import io
import jieba
import string
import logging
import threading
from transformers import BertTokenizer, BertModel
import torch
import jieba.posseg as pseg

# 配置日志
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='app.log',
                    filemode='w')  # 将 'a' 改为 'w'

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
        st.success(f"使用已存在的文: {file_path}")
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
    st.write("正在加载 SentenceTransformer 模型...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.write("SentenceTransformer 模型加载完成")
    
    st.write("正在初始化 FAISS 索引...")
    dimension = embedding_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)
    st.write("FAISS 索引初始化完成")
    
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
    
    for node in G.nodes():
        net.add_node(node, label=node, title=node)
    
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], title=edge[2].get('relation', ''))
    
    net.save_graph("temp_graph.html")
    with open("temp_graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, width=700, height=500)

def query_graph(_G, query, _embedding_model, index, nodes):
    logging.info(f"开始查询图: 节点数 {_G.number_of_nodes()}, 边数 {_G.number_of_edges()}")
    logging.info(f"查询: {query}")
    
    if not nodes:
        logging.warning("图中没有节点可供查询")
        return "图中没有节点可供查询。"
    
    # 使用jieba进行中文分词
    keywords = [word for word in jieba.lcut(query) if len(word) > 1]
    
    context = []
    
    # 直接在图中搜索这些关键词
    for keyword in keywords:
        matched_nodes = [node for node in _G.nodes() if keyword in str(node)]
        for node in matched_nodes:
            neighbors = list(_G.neighbors(node))
            edges = _G.edges(node, data=True)
            relations = []
            for edge in edges:
                if edge[1] != node:  # edge[1] 是邻居节点
                    relation = edge[2].get('relation', '相关')
                    relations.append(f"{node} 与 {edge[1]} 的关系: {relation}")
            if relations:
                context.append(f"关键词 '{keyword}' 相关的节点 '{node}' 的关系:\n" + "\n".join(relations))
    
    # 添加日志输出
    logging.info(f"提取的关键词: {keywords}")
    logging.info(f"查询结果: {context[:500]}...")  # 只记录前500个字符
    return "\n\n".join(context) if context else "未找到相关信息。"

def generate_answer(query, context, max_tokens=3000):
    logging.info(f"生成答案的查询: {query}")
    logging.info(f"生成答案的上下文长度: {len(context)}")
    
    # 截断上下文以适应模型的最大输入长度
    truncated_context = context[:max_tokens]
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个助手，根据给定的上下文信息回答问题。"},
                {"role": "user", "content": f"上下文信息：\n{truncated_context}\n\n问题：{query}\n\n请根据上述信息回答问题，如果信息不足，请说明并尝试推断可能的答案。"}
            ],
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()
        logging.info(f"生成的答案: {answer}")
        return answer
    except Exception as e:
        logging.error(f"生成答案时出错: {str(e)}")
        return "无法生成答案。"

def vector_rag(text, query, _embedding_model, sentence_index, sentences):
    if not sentences:
        return "没有可供查询的句子。"
    query_embedding = encode_text(query, _embedding_model)
    try:
        _, I = sentence_index.search(np.array([query_embedding]), k=min(5, len(sentences)))
        context = [sentences[i] for i in I[0] if 0 <= i < len(sentences)]
        return "\n".join(context)
    except Exception as e:
        logging.error(f"在vector_rag中搜索相似句子时出错: {str(e)}", exc_info=True)
        return "在处理查询出现误。"

@st.cache_data
def process_query(query, _G, _embedding_model, _index, nodes, sentences, text, _sentence_index):
    logging.info(f"处理查询: {query}")
    try:
        start_time = time.time()

        graph_start = time.time()
        graph_context = query_graph(_G, query, _embedding_model, _index, nodes)
        graph_time = time.time() - graph_start
        logging.info(f"图谱查询耗时: {graph_time:.2f}秒")

        vector_start = time.time()
        vector_context = vector_rag(text, query, _embedding_model, _sentence_index, sentences)
        vector_time = time.time() - vector_start
        logging.info(f"向量检索耗时: {vector_time:.2f}秒")

        graph_answer_start = time.time()
        graph_answer = generate_answer(query, graph_context)
        graph_answer_time = time.time() - graph_answer_start
        logging.info(f"图谱答案生成耗时: {graph_answer_time:.2f}秒")

        vector_answer_start = time.time()
        vector_answer = generate_answer(query, vector_context)
        vector_answer_time = time.time() - vector_answer_start
        logging.info(f"向量答案生成耗时: {vector_answer_time:.2f}秒")

        comparison_start = time.time()
        comparison = compare_results(query, graph_context, graph_answer, vector_context, vector_answer)
        comparison_time = time.time() - comparison_start
        logging.info(f"结果比较耗时: {comparison_time:.2f}秒")

        total_time = time.time() - start_time
        logging.info(f"总查询处理时间: {total_time:.2f}秒")
        
        return graph_context, graph_answer, vector_context, vector_answer, comparison
    except Exception as e:
        logging.error(f"处理查询时出错: {str(e)}")
        return "处理查询时出错", "无法生成答案", "处理查询时出错", "无法生成答案", "无法比较结果"

def on_query_submit():
    query = st.session_state.query_input
    if query:
        graph_context, graph_answer, vector_context, vector_answer, comparison = process_query(
            query, st.session_state.G, st.session_state.embedding_model, 
            st.session_state.index, st.session_state.nodes, 
            st.session_state.sentences, st.session_state.text,
            st.session_state.sentence_index  # 添加这个参数
        )
        st.session_state.last_query = query
        st.session_state.graph_context = graph_context
        st.session_state.graph_answer = graph_answer
        st.session_state.vector_context = vector_context
        st.session_state.vector_answer = vector_answer

def compare_results(query, graph_context, graph_answer, vector_context, vector_answer, max_tokens=2000):
    # 限制上下文和答案的长度
    graph_context = graph_context[:max_tokens] + ("..." if len(graph_context) > max_tokens else "")
    vector_context = vector_context[:max_tokens] + ("..." if len(vector_context) > max_tokens else "")
    graph_answer = graph_answer[:max_tokens] + ("..." if len(graph_answer) > max_tokens else "")
    vector_answer = vector_answer[:max_tokens] + ("..." if len(vector_answer) > max_tokens else "")

    comparison_prompt = f"""
    请比较下两种RAG方法的结果，并给出评价：

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
    3. 信息丰富度：哪种方法提供了更多有用信息？
    4. 整体表现：综合来看，哪种方法在回答这个问题时表现更好？

    请给出详细分析和解释。
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个专门分析和比较不同RAG方法的AI助手。"},
                {"role": "user", "content": comparison_prompt}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"生成比较结果时出错: {str(e)}")
        return "无法生成比较结果。"

@st.cache_data
def upload_and_process_pdf(uploaded_file):
    if uploaded_file is not None:
        logging.info(f"PDF文件上传成功: {uploaded_file.name}")
        # 读取PDF内容
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        logging.info(f"PDF文本提取完成，长度: {len(text)} 字符")
        
        # 保存文本文件
        file_path = os.path.join("uploaded_pdfs", uploaded_file.name.replace('.pdf', '.txt'))
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)
        
        logging.info(f"PDF文本已保存到: {file_path}")
        return file_path
    return None

def create_knowledge_graph_chinese(file_path):
    G = nx.Graph()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    logging.info(f"文件内容预览: {text[:200]}...")
    
    extracted_data = extract_entities_and_relations(text)
    
    logging.info(f"提取的数据: {extracted_data}")
    
    if not extracted_data or 'entities' not in extracted_data or 'relations' not in extracted_data:
        logging.warning("无法从文本中提取实体和关系。使用备用方法创建图。")
        return create_fallback_graph(text)
    
    # 创建一字典来存储患者
    patient_names = set()
    
    for entity in extracted_data['entities']:
        if entity['type'] == '患者姓名':
            patient_names.add(entity['name'])
        G.add_node(entity['name'], type=entity['type'])
    
    for relation in extracted_data['relations']:
        # 如果关系的一端是"患者"，我们将其替换为实际的患者姓名
        if relation['from'] == '患者' and patient_names:
            relation['from'] = list(patient_names)[0]
        if relation['to'] == '患者' and patient_names:
            relation['to'] = list(patient_names)[0]
        
        G.add_edge(relation['from'], relation['to'], type=relation['type'])
    
    logging.info(f"创建的图节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    
    if G.number_of_nodes() == 0:
        logging.warning("创建的图没有节点。使用备用方法创建图。")
        return create_fallback_graph(text)
    
    return G

def create_fallback_graph(text):
    G = nx.Graph()
    sentences = re.split(r'[。！？]', text)
    for sentence in sentences:
        words = list(jieba.cut(sentence.strip()))
        if len(words) > 2:
            clean_words = [word for word in words if len(word) > 1 and not word.isdigit()]
            if len(clean_words) > 2:
                G.add_edge(clean_words[0], clean_words[1], relation=' '.join(clean_words))
    
    logging.info(f"备用方法创建的图节点数: {G.number_of_nodes()}, 边: {G.number_of_edges()}")
    return G

def fix_json(json_string):
    # 移除可能导致题的Unicode字符
    json_string = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', json_string)
    
    # 确保JSON对象正确闭合
    json_string = json_string.strip()
    if not json_string.endswith('}'):
        json_string += '}'
    if not json_string.startswith('{'):
        json_string = '{' + json_string
    
    # 修复常见的JSON格式错误
    json_string = re.sub(r',\s*}', '}', json_string)  # 移除对象末尾多余的逗号
    json_string = re.sub(r',\s*]', ']', json_string)  # 移数组末尾多余的逗号
    
    return json_string

def extract_entities_and_relations(text):
    logging.debug("开始提取实体和关系")
    
    # 首先提取患者姓名
    patient_name_prompt = f"""
    请从以下电子病历文本中提取患者姓名。
    姓名格式必须是"姓氏+某某"，例如"张某某"或"李某某"。
    如果文本中没有明确的姓名，请返回"患者某某"。
    只需返回一个姓名，不需要其他解释。

    电子病历文本：
    {text[:1000]}  # 只使用1000个字符来提取姓名
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是专门提取者的AI手。请返回一个符合'姓氏+某某'格式的姓名。"},
                {"role": "user", "content": patient_name_prompt}
            ],
            max_tokens=50,
            temperature=0.3
        )
        patient_name = response.choices[0].message.content.strip()
        if not re.match(r'^[\u4e00-\u9fa5]某某$', patient_name):
            patient_name = "患者某某"
        logging.info(f"提取的患者姓名: {patient_name}")
    except Exception as e:
        logging.error(f"提取患者姓名时发生错误: {str(e)}")
        patient_name = "患者某某"

    # 然后提取其他实体和关系
    text_parts = [text[i:i+1500] for i in range(0, len(text), 1500)]
    all_entities = [{"name": patient_name, "type": "患者姓名"}]
    all_relations = []

    for part in text_parts:
        prompt = f"""
        请仔细分析以下电子病历文本，并提取重要的医疗实体和它们与患者"{patient_name}"之间的关系。

        实体类型可能包括但不限于：
        - 症状
        - 诊断
        - 治疗方法
        - 药物
        - 检查结果
        - 医疗设备
        - 医疗程序
        - 检查指标（如钙、葡萄糖、转氨酶等）

        关系类型可能包括但不限于：
        - "患有"（患者姓名与疾病或诊断）
        - "表现"（患者姓名与症状）
        - "接受"（患者姓名与治疗/检查）
        - "使用"（患者姓名与药物/备）
        - "检结果"（患者姓名与检查指标）
        - "相关"（患者姓名与其他所有实体）

        特注意：
        1. 所有实体都必须与患者"{patient_name}"建立直接关系。如果无法确定具体关系类型，请使用"相关"。
        2. 确保每个检查指标、症状、诊断等都作为独立的实体，并与患者建立直接关系。
        3. 不要遗漏任何可能的实体，即使它们看起来是次要的或不确定的。
        4. 不要创建新的患者姓名或更改已给定的患者姓名。

        请以JSON格式输出，格式如下：
        {{
            "entities": [
                {{"name": "实体名称", "type": "实体类型"}},
                ...
            ],
            "relations": [
                {{"from": "{patient_name}", "to": "实体名称", "type": "关系型"}},
                ...
            ]
        }}

        电子病历文本：
        {part}
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专门分析电子病历的AI助手，擅长提取医疗实体和关系。请严格按照指定的JSON格式输出结果。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            logging.info(f"GPT API 响应内容：{content[:200]}...")  # 记录前200个字符

            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                json_content = json_match.group()
                fixed_json = fix_json(json_content)
                parsed_content = json.loads(fixed_json)

                entities = parsed_content.get('entities', [])
                relations = parsed_content.get('relations', [])

                all_entities.extend(entities)
                all_relations.extend(relations)
            else:
                logging.error("无法从响应中提取 JSON 内容")

        except Exception as e:
            logging.exception(f"处理文本部分时发生异常: {str(e)}")

    # 去除重复的实体和关系
    unique_entities = list({entity['name']: entity for entity in all_entities}.values())
    unique_relations = list({(r['from'], r['to'], r['type']): r for r in all_relations}.values())

    return {"entities": unique_entities, "relations": unique_relations}

def init_models_with_timeout():
    result = []
    def target():
        result.append(init_models())
    
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=60)  # 60秒超时
    
    if thread.is_alive():
        st.error("模型初始化超时，请检查网络连接或重试")
        return None, None
    elif result:
        return result[0]
    else:
        st.error("模型初始化失败")
        return None, None

def display_graph_info(G):
    try:
        st.subheader("知识谱息")
        st.write(f"节点数量: {G.number_of_nodes()}")
        st.write(f"数量: {G.number_of_edges()}")
        
        st.write("节点示例:")
        for node in list(G.nodes())[:10]:  # 显示10个节点
            st.write(f"- {node}")
        
        st.write("边示例:")
        for edge in list(G.edges(data=True))[:10]:  # 显示前10条边
            st.write(f"- {edge[0]} -> {edge[1]}: {edge[2].get('relation', '未知关系')}")
    except Exception as e:
        logging.error(f"显示图信息时出错: {str(e)}")
        st.error("无法显示图信息")

def save_graph(G, filename):
    with open(filename, 'wb') as f:
        pickle.dump(G, f)

def load_graph(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_faiss_index(index, filename):
    faiss.write_index(index, filename)

def load_faiss_index(filename):
    return faiss.read_index(filename)

def save_embeddings(embeddings, filename):
    np.save(filename, embeddings)

def load_embeddings(filename):
    return np.load(filename)

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def process_and_save_data(file_path, prefix):
    logging.info(f"开始处理文件: {file_path}")
    logging.info(f"使用前缀: {prefix}")

    # 创建知识图谱
    G = create_knowledge_graph_chinese(file_path)
    
    # 始化型
    embedding_model, index = init_models()
    
    # 读取文本
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # 分割句子
    sentences = [s.strip() for s in re.split(r'[。！？]', text) if s.strip()]
    
    # 获取节点
    nodes = list(G.nodes())
    
    # 创建嵌入
    if nodes:
        node_embeddings = np.array([encode_text(node, embedding_model) for node in nodes])
        index.add(node_embeddings)
    else:
        node_embeddings = np.array([])
    
    # 创建句子嵌入
    sentence_embeddings = np.array([encode_text(sentence, embedding_model) for sentence in sentences])
    sentence_index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    sentence_index.add(sentence_embeddings)
    
    # 保存数据
    save_graph(G, f"{prefix}_graph.gpickle")
    save_faiss_index(index, f"{prefix}_index.faiss")
    save_embeddings(node_embeddings, f"{prefix}_node_embeddings.npy")
    save_embeddings(sentence_embeddings, f"{prefix}_sentence_embeddings.npy")
    save_data(nodes, f"{prefix}_nodes.pkl")
    save_data(sentences, f"{prefix}_sentences.pkl")
    save_faiss_index(sentence_index, f"{prefix}_sentence_index.faiss")

    logging.info(f"文件 {prefix} 的数据处理和保存完成")
    logging.info(f"图中节点数: {len(nodes)}, 句子数: {len(sentences)}")

    return G, index, node_embeddings, sentence_embeddings, nodes, sentences, sentence_index

def load_processed_data(prefix):
    G = load_graph(f"{prefix}_graph.gpickle")
    index = load_faiss_index(f"{prefix}_index.faiss")
    node_embeddings = load_embeddings(f"{prefix}_node_embeddings.npy")
    sentence_embeddings = load_embeddings(f"{prefix}_sentence_embeddings.npy")
    nodes = load_data(f"{prefix}_nodes.pkl")
    sentences = load_data(f"{prefix}_sentences.pkl")

    return G, index, node_embeddings, sentence_embeddings, nodes, sentences

def get_processed_files():
    processed_files = []
    for filename in os.listdir('uploaded_pdfs'):
        if filename.endswith('.txt'):
            base_name = os.path.splitext(filename)[0]
            has_vector = os.path.exists(f"{base_name}_index.faiss")
            has_graph = os.path.exists(f"{base_name}_graph.gpickle")
            processed_files.append({
                'name': filename,
                'has_vector': has_vector,
                'has_graph': has_graph
            })
    return processed_files

def load_file_data(file_name):
    base_name = os.path.splitext(file_name)[0]
    graph_file = f"{base_name}_graph.gpickle"
    index_file = f"{base_name}_index.faiss"
    sentence_index_file = f"{base_name}_sentence_index.faiss"
    
    logging.info(f"正在加载文件: {file_name}")
    logging.info(f"图文件: {graph_file}")
    logging.info(f"索引文件: {index_file}")
    logging.info(f"句子索引文件: {sentence_index_file}")

    if not all(os.path.exists(f) for f in [graph_file, index_file, sentence_index_file]):
        logging.error(f"缺少必要的文件: {file_name}")
        return None, None, None, None, None, None, None

    G = load_graph(graph_file)
    index = load_faiss_index(index_file)
    with open(f"uploaded_pdfs/{file_name}", 'r', encoding='utf-8') as file:
        text = file.read()
    sentences = [s.strip() for s in re.split(r'[。！？]', text) if s.strip()]
    nodes = list(G.nodes())
    sentence_index = load_faiss_index(sentence_index_file)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    logging.info(f"成功加载文件 {file_name} 的数据")
    logging.info(f"图中节点数: {len(nodes)}, 句子数: {len(sentences)}")
    
    return G, index, nodes, sentences, text, sentence_index, embedding_model

def encode_text(text, embedding_model):
    return embedding_model.encode(text)

def format_graph_results(context):
    if context == "未找到相关信息。":
        return context

    G = nx.Graph()
    lines = context.split('\n')
    current_entity = ""
    for line in lines:
        if line.startswith("相关节点") or line.startswith("实体"):
            try:
                current_entity = line.split("'")[1]
                G.add_node(current_entity)
            except IndexError:
                logging.warning(f"无法从行中提取实体: {line}")
                continue
        elif "的关系:" in line:
            try:
                parts = line.split("的关系:")
                if len(parts) != 2:
                    logging.warning(f"关系行格式不正确: {line}")
                    continue
                relation = parts[1].strip()
                entity_parts = parts[0].split("与")
                if len(entity_parts) != 2:
                    logging.warning(f"实体关系格式不确: {line}")
                    continue
                related_entity = entity_parts[1].strip()
                G.add_node(related_entity)
                G.add_edge(current_entity, related_entity, relation=relation)
            except Exception as e:
                logging.warning(f"处理关系行时出错: {line}. 错误: {str(e)}")
                continue

    if len(G) == 0:
        return "无法生成图形表示。"

    net = Network(notebook=True, width="100%", height="400px", bgcolor="#222222", font_color="white")
    
    for node in G.nodes():
        net.add_node(node, label=node, color="#00ff1e")
    
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], title=edge[2].get('relation', '未知关系'), color="#ffffff")

    net.save_graph("temp_graph.html")
    
    with open("temp_graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    
    return html

def reprocess_all_documents():
    uploaded_dir = 'uploaded_pdfs'
    if not os.path.exists(uploaded_dir):
        st.warning("没有找到上传的文档目录。")
        return

    files = [f for f in os.listdir(uploaded_dir) if f.endswith('.txt')]
    if not files:
        st.warning("没有找到可以处理的文本文件。")
        return

    for file in files:
        file_path = os.path.join(uploaded_dir, file)
        prefix = os.path.splitext(file)[0]
        with st.spinner(f"正在处理文件 {file}..."):
            process_and_save_data(file_path, prefix)
    
    st.success("所有文档已重新处完成。")

def prepare_demo_data():
    index_root = os.path.join(os.getcwd(), 'demo_data')
    file_path = os.path.join(index_root, 'input', 'demo_text.txt')

    if os.path.exists(file_path):
        st.success(f"使用已存在的文件: {file_path}")
        return file_path

    os.makedirs(os.path.join(index_root, 'input'), exist_ok=True)

    # 这里可以放置下载演示数据的代码
    # 例如，可以使用一个在线的中文文本作为演示数据
    url = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/lunyu/lunyu.json"

    response = requests.get(url, verify=True)
    if response.status_code == 200:
        data = response.json()
        text = "\n".join([item['chapter'] + ": " + item['content'] for item in data])
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)
        st.success(f"演示数据已下载并保存在: {file_path}")
    else:
        st.error(f"下载失败，状态码: {response.status_code}")
        return None

    return file_path

def init_demo_models_with_timeout():
    result = []
    def target():
        result.append(init_models())
    
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=60)  # 60秒超时
    
    if thread.is_alive():
        st.error("模型初始化超时，请检查网络连接或重试")
        return None, None
    elif result:
        return result[0]
    else:
        st.error("模型初始化失败")
        return None, None

def create_demo_knowledge_graph(file_path):
    G = nx.Graph()
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    logging.info(f"文件内容预览: {text[:200]}...")
    
    # 使用正则表达式分割句子，考虑中文标点
    sentences = re.split(r'[。！？!?]', text)
    for sentence in sentences:
        # 移除括号内的内容
        sentence = re.sub(r'\([^()]*\)', '', sentence)
        # 使用jieba进行中文分词
        words = jieba.lcut(sentence.strip())
        # 移除停用词和标点符号
        words = [word for word in words if word not in string.punctuation and len(word) > 1]
        
        if len(words) > 2:
            # 使用所有词对作为边，整个句子作为关系
            for i in range(len(words) - 1):
                for j in range(i + 1, len(words)):
                    G.add_edge(words[i], words[j], relation=' '.join(words))
    
    logging.info(f"创建的知识图谱节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    logging.info(f"图中的一些节点示例: {list(G.nodes())[:10]}")
    logging.info(f"图中的一些边示例: {list(G.edges(data=True))[:5]}")
    
    # 确保图中至少有一个节点
    if len(G) == 0:
        G.add_node("Empty")
        logging.warning("创建了一个空的知识图谱，添加了一个默认节点")
    
    return G

def demo_query_graph(_G, query, _embedding_model, index, nodes):
    logging.info(f"开始查询图: 节点数 {_G.number_of_nodes()}, 边数 {_G.number_of_edges()}")
    logging.info(f"查询: {query}")
    
    if not nodes:
        logging.warning("图中没有节点可供查询")
        return "图中没有节点可供查询。"
    
    # 使用jieba行中文分词
    keywords = [word for word in jieba.lcut(query) if len(word) > 1]
    
    context = []
    
    # 直接在图中搜索这些关键词
    for keyword in keywords:
        matched_nodes = [node for node in _G.nodes() if keyword in str(node)]
        for node in matched_nodes:
            neighbors = list(_G.neighbors(node))
            edges = _G.edges(node, data=True)
            relations = []
            for edge in edges:
                if edge[1] != node:  # edge[1] 是邻居节
                    relation = edge[2].get('relation', '相关')
                    relations.append(f"{node} 与 {edge[1]} 的关系: {relation}")
            if relations:
                context.append(f"关键词 '{keyword}' 相关的节点 '{node}' 的关系:\n" + "\n".join(relations))
    
    # 添加日志输出
    logging.info(f"提取的关键词: {keywords}")
    logging.info(f"查询结果: {context[:500]}...")  # 只记录前500个字符
    return "\n\n".join(context) if context else "未找到相关信息。"

def handle_demo_data():
    st.header("演示数据分析")
    
    # 初始化 session state 变量
    if 'demo_file_selected' not in st.session_state:
        st.session_state.demo_file_selected = False
    if 'demo_graph_html' not in st.session_state:
        st.session_state.demo_graph_html = None
    if 'demo_current_file' not in st.session_state:
        st.session_state.demo_current_file = None
    
    # 显示已处的文件列表
    processed_files = get_demo_processed_files()
    st.subheader("已处理的演示文件")
    st.write(f"处理文件数量: {len(processed_files)}")
    
    for file in processed_files:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            if st.button(f"{file['name']}", key=f"demo_select_{file['name']}"):
                st.session_state.demo_current_file = file['name']
                st.session_state.demo_file_selected = True
                st.session_state.demo_graph_html = None  # 清除之前的图谱
                st.rerun()
        with col2:
            if st.button("重新处理", key=f"demo_reprocess_{file['name']}"):
                with st.spinner(f"正在重新处理文件 {file['name']}..."):
                    file_path = os.path.join('demo_data', file['name'])
                    process_and_save_demo_data(file_path, os.path.splitext(file['name'])[0])
                st.success(f"文件 {file['name']} 已重新处理")
                st.session_state.demo_file_reprocessed = True
                st.session_state.demo_graph_html = None  # 清除之前的图谱
                st.rerun()
        with col3:
            if st.button("删除", key=f"demo_delete_{file['name']}"):
                delete_demo_file(file['name'])
                st.success(f"文件 {file['name']} 及其相关处理文件已删除")
                st.session_state.demo_file_deleted = True
                st.session_state.demo_graph_html = None  # 清除之前的图谱
                st.rerun()
    
    uploaded_file = st.file_uploader("上传新的演示文本文件", type="txt")
    if uploaded_file:
        file_path = upload_and_process_demo_file(uploaded_file)
        if file_path:
            st.success(f"演示文件已上传并处理: {file_path}")
            st.session_state.demo_current_file = os.path.basename(file_path)
            st.session_state.demo_file_uploaded = True
            st.rerun()
    
    # 添加信息
    st.write(f"当前文件: {st.session_state.demo_current_file if 'demo_current_file' in st.session_state else '无'}")
    
    if st.session_state.demo_current_file:
        # 加载当前选择的文件数据
        file_path = os.path.join('demo_data', st.session_state.demo_current_file)
        prefix = os.path.splitext(st.session_state.demo_current_file)[0]
        
        if 'demo_G' not in st.session_state or st.session_state.demo_file_selected:
            st.session_state.demo_G = load_graph(f"{prefix}_demo_graph.gpickle")
            st.session_state.demo_index = load_faiss_index(f"{prefix}_demo_index.faiss")
            st.session_state.demo_nodes = load_data(f"{prefix}_demo_nodes.pkl")
            st.session_state.demo_sentences = load_data(f"{prefix}_demo_sentences.pkl")
            with open(file_path, 'r', encoding='utf-8') as file:
                st.session_state.demo_text = file.read()
            st.session_state.demo_sentence_index = load_faiss_index(f"{prefix}_demo_sentence_index.faiss")
            st.session_state.demo_embedding_model, _ = init_models()
            
            st.session_state.demo_file_selected = False
        
        st.write(f"图中节点数: {len(st.session_state.demo_nodes)}")
        st.write(f"句子数: {len(st.session_state.demo_sentences)}")
        
        # 显示知识图谱
        st.subheader("知识图谱 (制显示)")
        if st.session_state.demo_graph_html is None or st.session_state.demo_file_selected:
            with st.spinner("正在生成知识图谱..."):
                st.session_state.demo_graph_html = get_graph_html(st.session_state.demo_G, max_nodes=100, max_edges=200)
                st.session_state.demo_file_selected = False
        st.components.v1.html(st.session_state.demo_graph_html, height=500)
        
        st.info("注意：为了提高加载速度，知识图谱仅显示了最重要的100个节点和200条边。")
        
        # 查询输入区域
        st.subheader("RAG 查询")
        query_input = st.text_input("输入您的查询:", key="demo_query_input")
        
        # 结果显示区域
        results_container = st.container()
        
        if query_input:
            if st.button("提交查询"):
                with st.spinner("正在处理查询..."):
                    graph_context, graph_answer, vector_context, vector_answer, comparison = process_demo_query(
                        query_input, st.session_state.demo_G, st.session_state.demo_embedding_model, 
                        st.session_state.demo_index, st.session_state.demo_nodes, 
                        st.session_state.demo_sentences, st.session_state.demo_text,
                        st.session_state.demo_sentence_index
                    )
                
                with results_container:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**GraphRAG 结果:**")
                        graph_html = format_graph_results(graph_context)
                        st.components.v1.html(graph_html, height=400)
                        st.markdown("**生成的答案:**")
                        st.write(graph_answer)
                    
                    with col2:
                        st.markdown("**纯向量RAG 结果:**")
                        st.write(vector_context)
                        st.markdown("**生成的答案:**")
                        st.write(vector_answer)
                    
                    st.subheader("结果比较")
                    st.write(comparison)
    else:
        st.info("请上传或选择一个演示文件进行分析。")
    
    # 检查是否需要重新运行
    if 'demo_file_reprocessed' in st.session_state or 'demo_file_uploaded' in st.session_state or 'demo_file_deleted' in st.session_state:
        if 'demo_file_reprocessed' in st.session_state:
            del st.session_state.demo_file_reprocessed
        if 'demo_file_uploaded' in st.session_state:
            del st.session_state.demo_file_uploaded
        if 'demo_file_deleted' in st.session_state:
            del st.session_state.demo_file_deleted
        st.rerun()

def get_demo_processed_files():
    processed_files = []
    demo_dir = 'demo_data'
    if os.path.exists(demo_dir):
        for filename in os.listdir(demo_dir):
            if filename.endswith('.txt'):
                processed_files.append({'name': filename})
    return processed_files

def upload_and_process_demo_file(uploaded_file):
    if uploaded_file is not None:
        file_path = os.path.join("demo_data", uploaded_file.name)
        os.makedirs("demo_data", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        process_and_save_demo_data(file_path, os.path.splitext(uploaded_file.name)[0])
        return file_path
    return None

def process_and_save_demo_data(file_path, prefix):
    G = create_demo_knowledge_graph(file_path)
    embedding_model, index = init_models()
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    nodes = list(G.nodes())
    if nodes:
        node_embeddings = embedding_model.encode(nodes)
        index.add(np.array(node_embeddings))
    
    sentence_embeddings = embedding_model.encode(sentences)
    sentence_index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    sentence_index.add(np.array(sentence_embeddings))
    
    save_graph(G, f"{prefix}_demo_graph.gpickle")
    save_faiss_index(index, f"{prefix}_demo_index.faiss")
    save_embeddings(node_embeddings, f"{prefix}_demo_node_embeddings.npy")
    save_embeddings(sentence_embeddings, f"{prefix}_demo_sentence_embeddings.npy")
    save_data(nodes, f"{prefix}_demo_nodes.pkl")
    save_data(sentences, f"{prefix}_demo_sentences.pkl")
    save_faiss_index(sentence_index, f"{prefix}_demo_sentence_index.faiss")

def process_demo_query(query, _G, _embedding_model, _index, nodes, sentences, text, _sentence_index):
    logging.info(f"处理查询: {query}")
    try:
        graph_start = time.time()
        graph_context = query_graph(_G, query, _embedding_model, _index, nodes)
        graph_time = time.time() - graph_start
        logging.info(f"图谱查询耗时: {graph_time:.2f}秒")

        vector_start = time.time()
        vector_context = vector_rag(text, query, _embedding_model, _sentence_index, sentences)
        vector_time = time.time() - vector_start
        logging.info(f"向量检索耗时: {vector_time:.2f}秒")

        graph_answer_start = time.time()
        graph_answer = generate_answer(query, graph_context[:3000])  # 限制上下文长度
        graph_answer_time = time.time() - graph_answer_start
        logging.info(f"图谱答案生成耗时: {graph_answer_time:.2f}秒")

        vector_answer_start = time.time()
        vector_answer = generate_answer(query, vector_context[:3000])  # 限制上下文长度
        vector_answer_time = time.time() - vector_answer_start
        logging.info(f"向量答案生成耗时: {vector_answer_time:.2f}秒")

        comparison_start = time.time()
        comparison = compare_results(query, graph_context[:1500], graph_answer, vector_context[:1500], vector_answer)
        comparison_time = time.time() - comparison_start
        logging.info(f"结果比较耗时: {comparison_time:.2f}秒")

        total_time = time.time() - graph_start
        logging.info(f"总查询处理时间: {total_time:.2f}秒")
        
        return graph_context, graph_answer, vector_context, vector_answer, comparison
    except Exception as e:
        logging.error(f"处理查询时出错: {str(e)}", exc_info=True)
        error_message = f"处理查询时出错: {str(e)}"
        return error_message, error_message, error_message, error_message, error_message

def get_graph_html(G, max_nodes=100, max_edges=200):
    # 选择最重要的节点（基于度数）
    important_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)[:max_nodes]
    subgraph = G.subgraph(important_nodes)

    # 如果边太多，进一步筛选
    if subgraph.number_of_edges() > max_edges:
        edges = sorted(subgraph.edges(data=True), key=lambda x: len(x[2].get('relation', '')), reverse=True)[:max_edges]
        subgraph = nx.Graph(edges)

    net = Network(notebook=True, width="100%", height="500px", cdn_resources="remote")
    
    # 设置物理模拟参数
    physics_options = {
        "forceAtlas2Based": {
            "gravitationalConstant": -50,
            "centralGravity": 0.01,
            "springLength": 100,
            "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based",
        "stabilization": {
            "enabled": True,
            "iterations": 1000,
            "updateInterval": 100
        }
    }
    
    net.options = Options()
    net.options.physics = physics_options

    for node in subgraph.nodes():
        net.add_node(node, label=node, title=node)
    
    for edge in subgraph.edges(data=True):
        net.add_edge(edge[0], edge[1], title=edge[2].get('relation', '')[:50])  # 限制关系文本长度
    
    # 使用固定布局
    options = {"physics": physics_options}
    net.set_options(json.dumps(options))
    
    net.save_graph("temp_graph.html")
    with open("temp_graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    return html

def delete_demo_file(file_name):
    base_name = os.path.splitext(file_name)[0]
    files_to_delete = [
        os.path.join('demo_data', file_name),
        f"{base_name}_demo_graph.gpickle",
        f"{base_name}_demo_index.faiss",
        f"{base_name}_demo_node_embeddings.npy",
        f"{base_name}_demo_sentence_embeddings.npy",
        f"{base_name}_demo_nodes.pkl",
        f"{base_name}_demo_sentences.pkl",
        f"{base_name}_demo_sentence_index.faiss"
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"已删除文件: {file_path}")
    
    # 清除相关的会话状态
    keys_to_clear = ['demo_G', 'demo_index', 'demo_nodes', 'demo_sentences', 'demo_text', 
                     'demo_sentence_index', 'demo_embedding_model', 'demo_graph_html']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    if 'demo_current_file' in st.session_state and st.session_state.demo_current_file == file_name:
        del st.session_state.demo_current_file
    
    logging.info(f"已删除文件 {file_name} 及其相关处理文件")

def main():
    st.title("GraphRAG vs 纯向量RAG 对比研究")

    # 添加侧边栏说明
    with st.sidebar:
        st.header("GraphRAG vs 纯向量RAG")
        st.markdown("""
        ### 数据处理机制对比
        
        #### GraphRAG:
        1. 构建知识图谱：从文本中提取实体和关系。
        2. 图嵌入：将图中的节点转换为向量。
        3. 查询处理：
           - 在图中搜索相关节点和关系。
           - 使用图结构进行推理。
        4. 结果生成：基于图的上下文生成答案。

        #### 纯向量RAG:
        1. 文本分割：将文档分割成小段落或句子。
        2. 向量化：将每个文本片段转换为向量。
        3. 查询处理：
           - 将查询转换为向量。
           - 搜索最相似的文本片段。
        4. 结果生成：基于检索到的文本片段生成答案。

        ### 主要区别
        - GraphRAG 利用结构化信息和关系推理。
        - 纯向量RAG 依赖于文本相似性匹配。
        - GraphRAG 可能更适合复杂查询和推理任务。
        - 纯向量RAG 通常更简单，处理速度可能更快。
        """)

    # 初始化所有可能用到的 session state 变量
    session_state_vars = [
        'current_file', 'demo_data_loaded', 'pdf_embedding_model',
        'G', 'embedding_model', 'index', 'nodes', 'sentences', 'text',
        'sentence_index', 'pdf_G', 'pdf_index', 'pdf_nodes', 'pdf_sentences', 'pdf_text'
    ]
    
    for var in session_state_vars:
        if var not in st.session_state:
            st.session_state[var] = None

    tab1, tab2 = st.tabs(["PDF分析", "演示数据"])
    
    with tab1:
        st.header("电子病历PDF分析")
        
        # 添加重新处理所有文档的按钮
        if st.button("重新处理所有上传的文档"):
            reprocess_all_documents()
        
        # 使用 expander 来显示会话状态变量
        with st.expander("查看会话状态变量", expanded=False):
            st.write(st.session_state)
        
        # 显示已处理的文件列表
        processed_files = get_processed_files()
        st.subheader("已处理的文件")
        st.write(f"处理文件数量: {len(processed_files)}")
        for file in processed_files:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                if st.button(f"{file['name']} (向量: {'是' if file['has_vector'] else '否'}, 图: {'是' if file['has_graph'] else '否'})", key=f"select_{file['name']}"):
                    # 清除之前的会话状态
                    for key in ['G', 'index', 'nodes', 'sentences', 'text', 'sentence_index', 'embedding_model']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.session_state.current_file = file['name']
                    st.rerun()  # 重新运行应用以更新显示
            with col2:
                if st.button("重新处理", key=f"reprocess_{file['name']}"):
                    with st.spinner(f"正在重新处理文件 {file['name']}..."):
                        file_path = os.path.join('uploaded_pdfs', file['name'])
                        process_and_save_data(file_path, os.path.splitext(file['name'])[0])
                    st.success(f"文件 {file['name']} 已重新处理")
                    st.rerun()
        
        uploaded_file = st.file_uploader("上传新的PDF文件", type="pdf")
        if uploaded_file:
            file_path = upload_and_process_pdf(uploaded_file)
            if file_path:
                st.success(f"PDF已上传并换为文本文件: {file_path}")
                st.session_state.current_file = os.path.basename(file_path)
                st.rerun()  # 重新运行应用以更新显示
        
        # 添加试信息
        st.write(f"当前文件: {st.session_state.current_file}")
        st.write(f"图中节点数: {len(st.session_state.nodes) if st.session_state.nodes else 0}")
        st.write(f"句子数: {len(st.session_state.sentences) if st.session_state.sentences else 0}")
        
        # 始终显示重新处理按钮，但根据条件禁用它
        reprocess_button = st.button(
            "重新处理当前文件",
            key="reprocess_button",
            disabled=(st.session_state.current_file is None)
        )
        
        if reprocess_button and st.session_state.current_file:
            st.write("正在重新处理文件...")  # 添加这行来确认按钮被点击
            file_path = os.path.join('uploaded_pdfs', st.session_state.current_file)
            with st.spinner("正在重新处理文件..."):
                G, index, node_embeddings, sentence_embeddings, nodes, sentences, sentence_index = process_and_save_data(file_path, os.path.splitext(st.session_state.current_file)[0])
            st.success("文件已重新处理")
            st.rerun()
        
        if st.session_state.current_file:
            # 加载文件数
            G, index, nodes, sentences, text, sentence_index, embedding_model = load_file_data(st.session_state.current_file)
            
            # 更新会话状态
            st.session_state.G = G
            st.session_state.index = index
            st.session_state.nodes = nodes
            st.session_state.sentences = sentences
            st.session_state.text = text
            st.session_state.sentence_index = sentence_index
            st.session_state.embedding_model = embedding_model
            
            # 显知识图谱
            st.subheader("知识图谱")
            logging.info(f"正在显示文件 {st.session_state.current_file} 的知识图谱")
            st.write(f"当前显示的是 {st.session_state.current_file} 的知识图谱")
            graph_container = st.container()
            with graph_container:
                visualize_graph_interactive(G)
            
            # 显示知识图谱信息
            display_graph_info(G)
            
            # 查询输入区域
            st.subheader("RAG 查询")
            query_input = st.text_input("输入您的查询:", key="pdf_query_input")
            
            # 结果显示区域
            results_container = st.container()
            
            if query_input:
                try:
                    graph_context, graph_answer, vector_context, vector_answer, comparison = process_query(
                        query_input, G, embedding_model, 
                        index, nodes, sentences, text, sentence_index
                    )
                    
                    with results_container:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**GraphRAG 结果:**")
                            graph_html = format_graph_results(graph_context)
                            st.components.v1.html(graph_html, height=400)
                            st.markdown("**生成的答案:**")
                            st.write(graph_answer)
                        
                        with col2:
                            st.markdown("**纯向量RAG 结果:**")
                            st.write(vector_context)
                            st.markdown("**生成的答案:**")
                            st.write(vector_answer)
                        
                        st.subheader("结果比较")
                        st.write(comparison)
                except Exception as e:
                    logging.error(f"处理查询时出错: {str(e)}")
                    st.error("处理查询时出现错误，请检查日志获取更多信息。")
        else:
            st.info("请选择一个文件或上传新的PDF件进行分析。")
    
    with tab2:
        handle_demo_data()

if __name__ == "__main__":
    logging.info("=" * 50)
    logging.info("新的应用程序会话开始")
    main()

