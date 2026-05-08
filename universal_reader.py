import os
import re
import jieba
import docx
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ⚠️ 数据集路径（保持你的真实路径不变）
folder_path = r"C:\Users\wxcyr\Desktop\数学建模赛题和论文模板\B题\B题数据集\数据集1：历史真实文件数据"

def extract_text_from_file(file_path, ext):
    """根据不同的后缀名，调用不同的解析库提取文本（自带防崩容错）"""
    text = ""
    try:
        if ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='gbk') as f:
                    text = f.read()
                    
        elif ext == '.docx':
            doc = docx.Document(file_path)
            text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
            
        elif ext == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                # 仅提取前20页，防止超长无意义扫描件卡死内存
                max_pages = min(len(pdf.pages), 20)
                pages_text = [pdf.pages[i].extract_text() for i in range(max_pages) if pdf.pages[i].extract_text()]
                text = "\n".join(pages_text)
                
        elif ext == '.xlsx':
            # 智能提取：只读前100行，兼顾办公表单的完整性和大数据的表头提取
            df_dict = pd.read_excel(file_path, sheet_name=None, nrows=100)
            for sheet_name, df in df_dict.items():
                headers = " ".join(df.columns.astype(str))
                content = df.astype(str).apply(lambda x: ' '.join(x), axis=1)
                content_str = " ".join(content)
                text += f"工作表名:{sheet_name} 表头:{headers} 内容:{content_str}\n"
                
    except Exception as e:
        # 静默跳过损坏文件，保证整体程序不挂
        pass
        
    return text

def load_and_clean_all_documents(path):
    print("🚀 老葛超级读取器 (V4.0 终极版) 启动中...")
    
    # 用于保存结果的列表
    valid_documents = []
    file_names = []
    file_types = []
    
    # 终极去噪停用词表
    base_stop_words = {'情况', '地区', '发展', '工作', '建设', '项目', '管理', '企业', '相关', '推进', '组织', '实施', '要求', '附件', '说明', '报告', '关于', '通知', '进行', '通过', '单位', '以上'}
    noise_words = {
        'the', 'and', 'of', 'in', 'to', 'for', 'is', 'we', 'on', 'that', 'with', 'as', 'by', 'this', 'from', 'are', 'an', 
        'nan', 'unnamed', 'none', 'null', 
        'url', 'html', 'jpg', 'png', 'ndsj', 'stats', 'gov', 'ch', 'sj', '下载', '图片', '点击'
    }
    stop_words = base_stop_words | noise_words
    supported_exts = ['.txt', '.docx', '.pdf', '.xlsx']
    
    # 获取总文件数用于进度条
    all_files = [os.path.join(root, f) for root, dirs, files in os.walk(path) for f in files if os.path.splitext(f)[1].lower() in supported_exts]
    total_files = len(all_files)
    print(f"总计发现 {total_files} 个支持解析的文档。开始硬核吞吐...\n")

    for index, file_path in enumerate(all_files):
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        
        if (index + 1) % 50 == 0:
            print(f"[{index+1}/{total_files}] 正在处理中，当前文件: {file_name}")

        raw_text = extract_text_from_file(file_path, ext)
        
        if len(raw_text.strip()) > 10:
            # 数据清洗流水线
            text_no_urls = re.sub(r'http\S+|www.\S+', ' ', raw_text) 
            text_no_digits = re.sub(r'\d+', ' ', text_no_urls)
            words = jieba.lcut(text_no_digits.lower())
            cleaned_words = [w for w in words if len(w) > 1 and w not in stop_words]
            
            if cleaned_words:
                valid_documents.append(" ".join(cleaned_words))
                file_names.append(file_name)
                file_types.append(ext)
                        
    print(f"\n✅ 兵马集结完毕！成功清洗了 {len(valid_documents)} 个有效文件！")
    return valid_documents, file_names, file_types

def cluster_and_save(documents, file_names, file_types, num_clusters=6):
    print(f"\n🧠 开始 TF-IDF 特征提取与 K-Means 聚类 (K={num_clusters})...")
    
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=15)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(tfidf_matrix)
    
    print("\n🏆 最终聚类主题与核心词：")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    # 打印每个类别的核心词
    for i in range(num_clusters):
        print(f"【类别 {i}】的核心词汇:")
        top_words = [terms[ind] for ind in order_centroids[i, :12]]
        print(" | ".join(top_words))
        print("-" * 40)
        
    # 保存结果到 CSV，作为第二问的有监督学习训练集
    print("\n💾 正在将分类结果落盘保存为 CSV 文件...")
    df = pd.DataFrame({
        '文件名': file_names,
        '文件格式': file_types,
        '聚类标签': labels,
        '清洗后文本特征': documents
    })
    output_path = "problem1_clustering_results.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"🎉 结果已保存至同目录下的 {output_path}。第一问圆满闭环！")

if __name__ == "__main__":
    docs, names, types = load_and_clean_all_documents(folder_path)
    if docs:
        cluster_and_save(docs, names, types, num_clusters=6)