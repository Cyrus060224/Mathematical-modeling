import warnings
# 屏蔽 PyTorch MPS 等底层警告，保持终端输出整洁
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
# 项目根目录定位
# ==========================================
# 基于当前脚本所在路径，动态获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================
# 图像光学字符识别 (OCR) 模块初始化
# ==========================================
try:
    import easyocr
    print("[INFO] 已检测到 EasyOCR，图像识别模块初始化完成。")
    # 全局初始化 OCR 模型，避免重复加载开销
    ocr_reader = easyocr.Reader(['ch_sim', 'en'])
except ImportError:
    ocr_reader = None
    print("[WARNING] 未检测到 EasyOCR 库，将跳过图像文件处理。如需启用该功能，请执行 'pip install easyocr'")

# ==========================================
# 跨平台相对路径配置
# ==========================================
def get_dataset_path():
    # 基于项目根目录构建数据集相对路径
    dataset_path = BASE_DIR / "data" / "数据集1：历史真实文件数据"
    
    current_os = platform.system()
    if current_os == "Darwin": 
        print(f"[INFO] 操作系统环境: macOS，数据集路径映射为: {dataset_path}")
    elif current_os == "Windows":
        print(f"[INFO] 操作系统环境: Windows，数据集路径映射为: {dataset_path}")
    else:
        print(f"[INFO] 操作系统环境: 其他系统，数据集路径映射为: {dataset_path}")
        
    return dataset_path

folder_path = get_dataset_path()

# ==========================================
# 数据提取与特征处理核心模块
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
                
        # 图像文件处理分支
        elif ext in ['.jpg', '.jpeg', '.png']:
            if ocr_reader is not None:
                # 规避特定系统下 OpenCV 读取中文路径的兼容性问题
                # 采用二进制流方式读取图像文件数据
                with open(file_path_str, 'rb') as img_file:
                    img_bytes = img_file.read()
                    
                # detail=0 参数用于仅提取纯文本内容，忽略边界框等空间坐标数据
                result = ocr_reader.readtext(img_bytes, detail=0)
                text = " ".join(result)
                
    except Exception as e:
        pass
        
    return text

def load_and_clean_all_documents(path):
    print("\n[INFO] 开始执行数据集加载与清洗模块...\n")
    
    if not path.exists():
        print(f"[ERROR] 路径不存在: {path}\n请检查数据集目录结构是否完整。")
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
    
    # 支持解析的文件扩展名列表
    supported_exts = ['.txt', '.docx', '.pdf', '.xlsx', '.jpg', '.jpeg', '.png']
    
    all_files = [os.path.join(root, f) for root, dirs, files in os.walk(path) for f in files if os.path.splitext(f)[1].lower() in supported_exts]
    total_files = len(all_files)
    print(f"[INFO] 共扫描到 {total_files} 个受支持的文件，开始批量提取与清洗...\n")

    for index, file_path in enumerate(all_files):
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        
        # 设置进度输出频率，控制终端打印开销
        if (index + 1) % 10 == 0:
            print(f"[PROCESS] 进度: {index+1}/{total_files} | 当前解析文件: {file_name}")

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
                        
    print(f"\n[INFO] 数据清洗完成，共提取 {len(valid_documents)} 个有效文本特征数据。")
    return valid_documents, file_names, file_types

def cluster_and_save(documents, file_names, file_types, num_clusters=6):
    print(f"\n[INFO] 启动文本特征提取 (TF-IDF) 与 K-Means 聚类分析 (K={num_clusters})...")
    
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=15)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(tfidf_matrix)
    
    print("\n[RESULT] 聚类模型收敛，各簇核心特征词如下：")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    for i in range(num_clusters):
        print(f"Cluster {i} 核心特征词:")
        top_words = [terms[ind] for ind in order_centroids[i, :12]]
        print(" | ".join(top_words))
        print("-" * 40)
        
    print("\n[INFO] 正在将聚类结果序列化并导出至 CSV 文件...")
    df = pd.DataFrame({
        '文件名': file_names,
        '文件格式': file_types,
        '聚类标签': labels,
        '清洗后文本特征': documents
    })
    
    # 配置输出路径：结果将保存于项目根目录的 results 目录下
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True) # 确保输出目录存在
    output_path = results_dir / "problem1_clustering_results.csv"
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"[SUCCESS] 聚类结果已成功保存至: {output_path}")

if __name__ == "__main__":
    docs, names, types = load_and_clean_all_documents(folder_path)
    if docs:
        cluster_and_save(docs, names, types, num_clusters=6)