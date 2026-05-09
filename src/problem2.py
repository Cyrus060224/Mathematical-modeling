import os
import re
import warnings
import numpy as np
import pandas as pd
import platform
from pathlib import Path
import jieba

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.stats import entropy

# 🤫 屏蔽可能出现的第三方库警告，保持终端清爽
warnings.filterwarnings("ignore")

# =====================================================
# 📍 动态项目根目录与路径配置
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# 问题一的训练数据
TRAIN_FILE = BASE_DIR / "results" / "problem1_clustering_results.csv"

# 输出结果
OUTPUT_FILE = BASE_DIR / "results" / "problem2_predictions.csv"
METRICS_FILE = BASE_DIR / "results" / "problem2_metrics.txt"

# 数据源路径：数据集2是文件夹，数据集3是单独的 Excel 文件
DATASET2_DIR = BASE_DIR / "data" / "数据集2：后续流入的半结构化记录数据"
DATASET3_FILE = BASE_DIR / "data" / "数据集3：后续流入的匿名原始文件数据.xlsx"

# =====================================================
# 🛠️ 轻量级文本读取与清洗
# =====================================================
def clean_text(text):
    if pd.isna(text) or not str(text).strip():
        return ""
    text = str(text)
    # 纯净去噪：只保留汉字和字母
    text_clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]+', ' ', text)
    words = jieba.lcut(text_clean.lower())
    return " ".join([w for w in words if len(w) > 1])

def load_new_datasets():
    """读取数据集2(文件夹)和数据集3(文件)中的新流入数据"""
    print("\n📂 正在扫描数据集2和数据集3的新流入数据...")
    new_data = []
    
    # -----------------------------------------
    # 1. 处理数据集 2 (遍历文件夹中的文件)
    # -----------------------------------------
    if DATASET2_DIR.exists():
        for root, _, files in os.walk(DATASET2_DIR):
            for file in files:
                if file.startswith('~') or file.startswith('.'): continue
                file_path = Path(root) / file
                ext = file_path.suffix.lower()
                
                text_content = ""
                try:
                    if ext == '.txt':
                        text_content = file_path.read_text(encoding='utf-8', errors='ignore')
                    elif ext == '.csv':
                        df = pd.read_csv(file_path, nrows=50)
                        # 🚨 修复点：强制转换为 numpy 数组再压平，避开 Arrow 引擎报错
                        text_content = " ".join(df.astype(str).to_numpy().flatten())
                    elif ext == '.xlsx':
                        df = pd.read_excel(file_path, nrows=50)
                        # 🚨 修复点：同上
                        text_content = " ".join(df.astype(str).to_numpy().flatten())
                except Exception:
                    pass
                
                if len(text_content.strip()) > 5:
                    new_data.append({
                        '文件名': file,
                        '数据来源': '数据集2',
                        '原始文本': text_content
                    })
    else:
        print(f"⚠️ 警告: 未找到文件夹 {DATASET2_DIR}")

    # -----------------------------------------
    # 2. 处理数据集 3 (多重引擎容错机制)
    # -----------------------------------------
    if DATASET3_FILE.exists():
        print(f"📄 发现数据集3文件，正在尝试破解读取...")
        df3 = None
        try:
            # 尝试方案 A：强制使用 openpyxl 引擎
            df3 = pd.read_excel(DATASET3_FILE, engine='openpyxl')
            print("✅ 成功以 Excel 格式读取数据集3！")
        except Exception as e1:
            print(f"⚠️ Excel引擎解析失败，启动备用 CSV 方案...")
            try:
                # 尝试方案 B：UTF-8 编码 CSV
                df3 = pd.read_csv(DATASET3_FILE, encoding='utf-8-sig')
                print("✅ 成功以 UTF-8 CSV 格式读取数据集3！")
            except Exception as e2:
                try:
                    # 尝试方案 C：GBK 编码 CSV
                    df3 = pd.read_csv(DATASET3_FILE, encoding='gbk')
                    print("✅ 成功以 GBK CSV 格式读取数据集3！")
                except Exception as e3:
                    print(f"❌ 数据集3读取彻底失败。")
        
        # 如果读取成功，开始拼装行文本
        if df3 is not None:
            for index, row in df3.iterrows():
                # 🚨 致命报错修复点：对于 Series 行对象，直接转成 Python List
                text_content = " ".join(row.astype(str).tolist())
                text_content = text_content.replace('nan', ' ').strip()
                
                if len(text_content) > 5:
                    new_data.append({
                        '文件名': f"匿名文件_行号_{index+1}", 
                        '数据来源': '数据集3',
                        '原始文本': text_content
                    })
    else:
        print(f"⚠️ 警告: 未找到文件 {DATASET3_FILE}")

    # -----------------------------------------
    # 3. 统一清洗返回
    # -----------------------------------------
    df_new = pd.DataFrame(new_data)
    if not df_new.empty:
        print(f"✅ 成功加载 {len(df_new)} 条新数据！正在清洗特征词...")
        df_new['清洗后文本特征'] = df_new['原始文本'].apply(clean_text)
        # 过滤掉清洗后变为空的数据
        df_new = df_new[df_new['清洗后文本特征'].str.len() > 0].copy()
    else:
        print("❌ 未提取到任何有效数据，请检查 data 文件夹。")
        
    return df_new

# =====================================================
# 🧠 核心：训练基座模型与量化评价指标
# =====================================================
def run_classification_pipeline():
    print("=" * 60)
    print("🚀 问题二：多源异构数据迁移分类与评价系统启动")
    print("=" * 60)

    # 1. 加载第一问的“黄金标准”作为训练集
    if not TRAIN_FILE.exists():
        print(f"❌ 严重错误：找不到第一问结果 {TRAIN_FILE}")
        return
    
    print("\n📚 第一阶段：加载问题一主题分类体系...")
    df_train = pd.read_csv(TRAIN_FILE)
    X_train_text = df_train['清洗后文本特征'].fillna('')
    y_train = df_train['聚类标签']
    
    # 使用统一的 TF-IDF 向量空间 (与第一问对齐)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    
    # 训练逻辑回归分类器
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    print("✅ 分类器训练完毕 (基于第一问历史体系)")

    # 2. 加载并转换新流入数据
    df_new = load_new_datasets()
    if df_new.empty:
        print("❌ 分类中止：新数据集为空。")
        return
        
    print("\n🔮 第二阶段：处理新数据并计算概率分布...")
    X_new = vectorizer.transform(df_new['清洗后文本特征'])
    
    # 获取预测概率矩阵
    probs = model.predict_proba(X_new)
    
    # 3. 核心机制：计算评价指标与异常捕捉
    print("\n📐 第三阶段：计算综合量化评价指标与异常拦截...")
    
    predictions = []
    confidences = []
    ambiguity_scores = []
    entropies = []
    status_flags = []
    
    for i, prob_dist in enumerate(probs):
        sorted_indices = np.argsort(prob_dist)[::-1]
        top1_class = sorted_indices[0]
        top2_class = sorted_indices[1]
        
        top1_prob = prob_dist[top1_class]
        top2_prob = prob_dist[top2_class]
        
        confidences.append(top1_prob)
        margin = top1_prob - top2_prob
        ambiguity_scores.append(margin)
        ent = entropy(prob_dist, base=2)
        entropies.append(ent)
        
        # 异常拦截规则
        if top1_prob < 0.40:
            status_flags.append("无法明确归类 (低置信度)")
            predictions.append(-1)
        elif margin < 0.15:
            status_flags.append(f"多类别特征 (倾向 {model.classes_[top1_class]} 和 {model.classes_[top2_class]})")
            predictions.append(-1) 
        else:
            status_flags.append("自动归类成功")
            predictions.append(model.classes_[top1_class])

    # 4. 组装并保存结果
    df_new['预测类别'] = predictions
    df_new['最大置信度'] = confidences
    df_new['类别混淆度'] = ambiguity_scores
    df_new['信息熵'] = entropies
    df_new['处理状态'] = status_flags
    
    df_new.drop(columns=['原始文本', '清洗后文本特征']).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n💾 预测结果已保存至: {OUTPUT_FILE}")
    
    # 5. 输出宏观评价报告
    success_rate = (df_new['预测类别'] != -1).mean() * 100
    avg_entropy = np.mean(entropies)
    
    report = f"""
==================================================
📊 问题二：分类系统综合量化评价报告
==================================================
【1. 迁移适用性评价】
- 新数据总测试量: {len(df_new)} 份
- 模型有效识别率: {success_rate:.2f}%
- 需人工复核率: {100 - success_rate:.2f}% 

【2. 分类合理性与可解释性评价】
- 全局平均信息熵 (H): {avg_entropy:.4f} (越低代表模型判定越果断)
- 平均预测置信度: {np.mean(confidences):.4f}

【3. 异常数据处理说明】
模型共拦截了 {sum(df_new['预测类别'] == -1)} 份异常数据，已统一标记为 -1 (待复核)。
这些数据将被移交至问题三的人工/规则复核流程。
==================================================
"""
    print(report)
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"📝 量化评价报告已保存至: {METRICS_FILE}")

if __name__ == "__main__":
    run_classification_pipeline()