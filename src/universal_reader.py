import warnings
# 🤫 屏蔽讨厌的 PyTorch MPS 警告，让终端保持清爽
warnings.filterwarnings("ignore", category=UserWarning)

import os
import re
import jieba
import docx
import pdfplumber
import pandas as pd
import platform
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ==========================================
# 📍 动态项目根目录定位
# ==========================================
# 因为当前脚本在 src 文件夹下，.parent 就是 src，再 .parent 就是项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================
# 👁️ 视觉模块初始化 (OCR)
# ==========================================
try:
    import easyocr
    print("✅ 检测到 EasyOCR，图像识别模块已就绪！")
    # 初始化 OCR 模型（放在全局，避免每次读文件都重新加载）
    ocr_reader = easyocr.Reader(['ch_sim', 'en'])
except ImportError:
    ocr_reader = None
    print("⚠️ 未检测到 EasyOCR，图片文件将被跳过。如需处理图片，请执行 'pip install easyocr'")

# ==========================================
# 🚀 跨平台智能相对路径配置
# ==========================================
def get_dataset_path():
    # 结合咱们标准化的项目结构，直接使用基于 BASE_DIR 的相对路径
    dataset_path = BASE_DIR / "data" / "数据集1：历史真实文件数据"
    
    current_os = platform.system()
    if current_os == "Darwin": 
        print(f"🍎 已加载 Mac 路径 (基于项目根目录: {dataset_path})")
    elif current_os == "Windows":
        print(f"🪟 已加载 Windows 路径 (基于项目根目录: {dataset_path})")
    else:
        print(f"🐧 已加载其他系统路径: {dataset_path}")
        
    return dataset_path

folder_path = get_dataset_path()

# ==========================================
# 🛠️ 核心功能模块 (已加入图片解析)
# ==========================================
def extract_text_from_file(file_path, ext):
    text = ""
    file_path_str = str(file_path) 
    
    try:
        if ext == '.txt':
            try:
                with open(file_path_str, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(file_path_str, 'r', encoding='gbk') as f:
                    text = f.read()
                    
        elif ext == '.docx':
            doc = docx.Document(file_path_str)
            text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
            
        elif ext == '.pdf':
            with pdfplumber.open(file_path_str) as pdf:
                max_pages = min(len(pdf.pages), 20)
                pages_text = [pdf.pages[i].extract_text() for i in range(max_pages) if pdf.pages[i].extract_text()]
                text = "\n".join(pages_text)
                
        elif ext == '.xlsx':
            df_dict = pd.read_excel(file_path_str, sheet_name=None, nrows=100)
            for sheet_name, df in df_dict.items():
                headers = " ".join(df.columns.astype(str))
                content = df.astype(str).apply(lambda x: ' '.join(x), axis=1)
                content_str = " ".join(content)
                text += f"工作表名:{sheet_name} 表头:{headers} 内容:{content_str}\n"
                
        # 🚀 图像处理分支
        elif ext in ['.jpg', '.jpeg', '.png']:
            if ocr_reader is not None:
                # detail=0 表示只提取纯文本列表，丢弃坐标框等冗余数据
                result = ocr_reader.readtext(file_path_str, detail=0)
                text = " ".join(result)
                
    except Exception as e:
        pass
        
    return text

def load_and_clean_all_documents(path):
    print("\n🚀 老葛超级读取器 (V5.2 终极架构版) 启动中...\n")
    
    if not path.exists():
        print(f"❌ 严重错误：找不到文件夹路径 {path}\n请确保 '数据集1：历史真实文件数据' 文件夹已放入项目的 'data' 目录下！")
        return [], [], []

    valid_documents = []
    file_names = []
    file_types = []
    
    base_stop_words = {'情况', '地区', '发展', '工作', '建设', '项目', '管理', '企业', '相关', '推进', '组织', '实施', '要求', '附件', '说明', '报告', '关于', '通知', '进行', '通过', '单位', '以上', '根据'}
    noise_words = {
        'the', 'and', 'of', 'in', 'to', 'for', 'is', 'we', 'on', 'that', 'with', 'as', 'by', 'this', 'from', 'are', 'an', 
        'nan', 'unnamed', 'none', 'null', 
        'url', 'html', 'jpg', 'png', 'ndsj', 'stats', 'gov', 'ch', 'sj', '下载', '图片', '点击'
    }
    stop_words = base_stop_words | noise_words
    
    # 图片格式在白名单中
    supported_exts = ['.txt', '.docx', '.pdf', '.xlsx', '.jpg', '.jpeg', '.png']
    
    all_files = [os.path.join(root, f) for root, dirs, files in os.walk(path) for f in files if os.path.splitext(f)[1].lower() in supported_exts]
    total_files = len(all_files)
    print(f"总计发现 {total_files} 个支持解析的文件（含图片）。开始硬核吞吐...\n")

    for index, file_path in enumerate(all_files):
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        
        # 频率调低：因为图片识别较慢，每处理 10 个文件就报一次平安
        if (index + 1) % 10 == 0:
            print(f"[{index+1}/{total_files}] 正在处理中，当前文件: {file_name}")

        raw_text = extract_text_from_file(file_path, ext)
        
        if len(raw_text.strip()) > 10:
            text_no_urls = re.sub(r'http\S+|www.\S+', ' ', raw_text) 
            text_no_digits = re.sub(r'\d+', ' ', text_no_urls)
            text_clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]+', ' ', text_no_digits)
            
            words = jieba.lcut(text_clean.lower())
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
    
    for i in range(num_clusters):
        print(f"【类别 {i}】的核心词汇:")
        top_words = [terms[ind] for ind in order_centroids[i, :12]]
        print(" | ".join(top_words))
        print("-" * 40)
        
    print("\n💾 正在将分类结果落盘保存为 CSV 文件...")
    df = pd.DataFrame({
        '文件名': file_names,
        '文件格式': file_types,
        '聚类标签': labels,
        '清洗后文本特征': documents
    })
    
    # 📍 输出路径修改：确保 CSV 保存到项目根目录的 results 文件夹下
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True) # 如果没有 results 文件夹就自动建一个
    output_path = results_dir / "problem1_clustering_results.csv"
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"🎉 结果已保存至: {output_path}")
    print("第一问完美收官！可以准备开工问题二了！")

if __name__ == "__main__":
    docs, names, types = load_and_clean_all_documents(folder_path)
    if docs:
        cluster_and_save(docs, names, types, num_clusters=6)