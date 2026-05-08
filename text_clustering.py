import os
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ⚠️ 注意保持你原来的路径
folder_path = r"C:\Users\wxcyr\Desktop\数学建模赛题和论文模板\B题\B题数据集\数据集1：历史真实文件数据"

def load_and_clean_texts(path):
    print("老葛正在帮你清洗并分词 txt 文件 (V2.0启动)...")
    documents = []
    
    # 老葛特调：公文常见无意义高频词（停用词表）
    stop_words = {'情况', '地区', '发展', '工作', '建设', '项目', '管理', '企业', '相关', '推进', '组织', '实施', '要求'}
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='gbk') as f:
                            text = f.read()
                    except Exception:
                        continue 
                
                if len(text.strip()) > 10:
                    # 1. 使用正则把所有数字替换为空格
                    text_no_digits = re.sub(r'\d+', ' ', text)
                    
                    # 2. 分词
                    words = jieba.lcut(text_no_digits)
                    
                    # 3. 核心清洗：只要长度大于1的词，并且不能在停用词表里
                    cleaned_words = [w for w in words if len(w) > 1 and w not in stop_words]
                    
                    documents.append(" ".join(cleaned_words))
                    
    print(f"成功清洗了 {len(documents)} 个有效 txt 文件！")
    return documents

def cluster_texts(documents, num_clusters=5):
    print(f"\n开始深度特征提取与聚类（设定聚成 {num_clusters} 类）...")
    
    # 提高了 min_df，过滤掉那些只在极个别文档里出现的生僻词
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=10)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(tfidf_matrix)
    
    print("\n--- V2.0 聚类结果与核心主题词揭晓 ---")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    for i in range(num_clusters):
        print(f"【类别 {i+1}】的核心词汇:")
        top_words = [terms[ind] for ind in order_centroids[i, :10]]
        print(" | ".join(top_words))
        print("-" * 40)

if __name__ == "__main__":
    docs = load_and_clean_texts(folder_path)
    if docs:
        cluster_texts(docs, num_clusters=5)

