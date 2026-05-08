import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import cosine_distances
from docx import Document
import fitz

# ====================== 精确路径 ======================
DATA3_PATH = r"D:\B题数据集\数据集\数据集3：后续流入的匿名原始文件数据.xlsx"
DATA2_DIR = r"D:\B题数据集\数据集\数据集2：后续流入的半结构化记录数据"

print("="*70)
print("【路径严格检查】")
print("数据集2 路径:", DATA2_DIR)
print("是否存在:", os.path.exists(DATA2_DIR))
print("是否是文件夹:", os.path.isdir(DATA2_DIR))

# 列出文件夹内实际内容
if os.path.exists(DATA2_DIR):
    all_items = os.listdir(DATA2_DIR)
    print(f"文件夹内共有 {len(all_items)} 个文件/文件夹")
    print("前15个内容:", all_items[:15])
else:
    print("❌ 路径不存在！")

# ====================== 文本清洗和模型（保持不变） ======================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：、]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# 加载模型和训练分类器（省略中间部分，与你之前一致）
print("\n加载分类模型...")
df_train = pd.read_csv("问题一_分类结果_优化版.csv")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
df_train['clean_text'] = df_train['文本内容'].apply(clean_text)
X_train = model.encode(df_train['clean_text'].tolist(), batch_size=16, show_progress_bar=True)
y_train = df_train['类别'].values

clf = NearestCentroid()
clf.fit(X_train, y_train)
print("✅ 分类器训练完成\n")

def predict_category(text):
    if not isinstance(text, str) or len(text.strip()) < 30:
        return -1, 0.0
    cleaned = clean_text(text)
    if len(cleaned) < 30:
        return -1, 0.0
    emb = model.encode([cleaned])
    pred = clf.predict(emb)[0]
    centroid = clf.centroids_[pred].reshape(1, -1)
    dist = cosine_distances(emb, centroid)[0][0]
    confidence = round(float(1 / (1 + dist)), 4)
    return int(pred), confidence

# ====================== 处理数据集2（强化版） ======================
print("正在处理数据集2...")
data2 = []

for root, dirs, files in os.walk(DATA2_DIR):
    for file in files:
        if file.lower().endswith(('.txt', '.pdf', '.docx')):
            full_path = os.path.join(root, file)
            try:
                if file.lower().endswith('.txt'):
                    text = open(full_path, 'r', encoding='utf-8', errors='ignore').read()
                elif file.lower().endswith('.pdf'):
                    doc = fitz.open(full_path)
                    text = "\n".join([page.get_text() for page in doc])
                    doc.close()
                elif file.lower().endswith('.docx'):
                    doc = Document(full_path)
                    text = "\n".join([p.text for p in doc.paragraphs])
                
                if len(text.strip()) > 30:
                    data2.append({
                        "文件名": file,
                        "文件类型": os.path.splitext(file)[1].lower(),
                        "文本内容": text[:8000]
                    })
            except Exception as e:
                print(f"读取失败 {file}: {e}")

df2 = pd.DataFrame(data2)
print(f"\n✅ 数据集2 共提取 {len(df2)} 个有效文件")

if len(df2) > 0:
    df2['clean_text'] = df2['文本内容'].apply(clean_text)
    results2 = [predict_category(text) for text in df2['clean_text']]
    df2['预测类别'] = [r[0] for r in results2]
    df2['置信度'] = [r[1] for r in results2]
    df2['预测主题'] = df2['预测类别'].map({
        -1: "未明确类别（需人工复核）", 0: "统计数据与年报类", 1: "财务决算类", 
        2: "混合文档类", 3: "居民消费统计类", 4: "城市建设类", 
        5: "工作报告总结类", 6: "社会事业统计类", 7: "其他"
    })
    df2.to_excel("问题二_数据集2_最终分类结果.xlsx", index=False)
    print("✅ 数据集2 分类结果已保存")
else:
    print("⚠️ 仍然提取为0，请把上面【路径严格检查】部分的内容复制给我")

print("\n🎉 处理结束")