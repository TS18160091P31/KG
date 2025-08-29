import torch
import re
import ollama
from config import settings

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def split_text_into_sentences(text):
    """
    将多段中文文本按句子拆分，返回 vault_content 格式的列表。
    每句以中文标点（。！？）为分隔。 
    """
    # 去除换行、清理空格
    text = text.strip().replace('\n', '')
    # 使用正则表达式断句：匹配中文句号、叹号、问号
    # 保留标点符号作为句子结尾
    sentences = re.findall(r'[^。！？]*[。！？]', text)
    # 去除首尾空格
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def split_text_into_paragraphs(text: str) -> list[str]:
    """
    将文本按段落（换行符）切分，去除空段、首尾空格，自动兼容 \r\n/\r。
    同时去掉重复段落（保留顺序）。
    """
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 切分并去掉首尾空白
    raw_paragraphs = [p.strip() for p in text.strip().split("\n") if p.strip()]
    
    # 去重但保持顺序
    paragraphs = list(dict.fromkeys(raw_paragraphs))
    
    return paragraphs

# Function to generate embeddings using Ollama
def generate_embeddings(content, model='mxbai-embed-large'):
    """
    使用提供的模型和URL生成内容的嵌入。
    :param content: 输入的文本内容
    :param model: 模型标识符，默认为 'mxbai-embed-large'
    :param url: API URL，默认为配置文件中的 OLLAMA_BASE + '/v1'
    :return: 嵌入结果
    """
    url = settings.EMBED_BASE
    client = ollama.Client(host=url)
    # ollama.api_url = url  # 替换为内网 IP 地址
    response = client.embeddings(model=model, prompt=content)
    return response

# Function to convert embeddings to tensor
def embeddings_to_tensor(embeddings_list):
    """
    将嵌入向量列表转换为 Tensor 格式。
    """
    return torch.tensor(embeddings_list)

# Function to get embeddings for the entire text content
def get_embeddings(content):
    vault_content = split_text_into_sentences(content)
    # print(vault_content)
    # 生成 vault_content 的嵌入
    vault_embeddings = []
    for content1 in vault_content:
        embedding = generate_embeddings(content1)  # 使用 Ollama 生成 embedding
        vault_embeddings.append(embedding["embedding"])
    # 将嵌入向量转换为 tensor

    return vault_embeddings

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if len(vault_embeddings) == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    vault_embeddings_tensor = embeddings_to_tensor(vault_embeddings)
    input_embedding = generate_embeddings(rewritten_input)["embedding"]  # 获取用户输入的 embedding
    # 将用户输入的 embedding 转换为 tensor
    input_embedding_tensor = torch.tensor(input_embedding).unsqueeze(0)  # 获取用户输入的 embedding
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(input_embedding_tensor, vault_embeddings_tensor)
    
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def get_relevant_answer(text: str, question: str, top_k: int = 20):
    """
    通过输入原始文本和问题，返回最相关的上下文段落或句子。
    
    参数：
    - text (str): 原始文本
    - question (str): 问题
    - top_k (int): 返回的最相关的上下文数量
    
    返回：
    - relevant (List[str]): 最相关的上下文段落或句子
    """
    # Step 1: 将文本拆分为句子
    sentences = split_text_into_sentences(text)
    
    # Step 2: 获取文本的嵌入向量
    vault_embeddings = get_embeddings(text)
    
    # Step 3: 获取最相关的上下文
    relevant = get_relevant_context(question, vault_embeddings, sentences, top_k)
    
    return relevant

if __name__ == "__main__":
    # 读取文件内容
    text = open_file("QA_part/vault.txt")
    sentences=split_text_into_sentences(text)
    # 问题输入
    question = "里根号在哪里"
    # 获取嵌入向量
    vault_embeddings = get_embeddings(text)
    # 获取最相关的上下文
    relevant = get_relevant_context(question, vault_embeddings, sentences, top_k=20)
    
    if relevant:
        context_str = "\n".join(relevant)
    print(context_str, "\n", vault_embeddings)
