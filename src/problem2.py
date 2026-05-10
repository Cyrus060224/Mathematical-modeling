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

# 屏蔽第三方库的底层警告，保持日志输出整洁
warnings.filterwarnings("ignore")

# =====================================================
# 项目根目录与绝对路径配置
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# 问题一聚类结果（先验训练集）
TRAIN_FILE = BASE_DIR / "results" / "problem1_clustering_results.csv"

# 预测结果与量化评价指标输出路径
OUTPUT_FILE = BASE_DIR / "results" / "problem2_predictions.csv"
METRICS_FILE = BASE_DIR / "results" / "problem2_metrics.txt"

# 新流入数据源路径设定
DATASET2_DIR = BASE_DIR / "data" / "数据集2：后续流入的半结构化记录数据"
DATASET3_FILE = BASE_DIR / "data" / "数据集3：后续流入的匿名原始文件数据.xlsx"

# =====================================================
# 文本数据的加载与预处理模块
# =====================================================
def clean_text(text):
    if pd.isna(text) or not str(text).strip():
        return ""
    text = str(text)
    # 数据降噪：仅保留中文字符与英文字母
    text_clean = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]+', ' ', text)
    words = jieba.lcut(text_clean.lower())
    return " ".join([w for w in words if len(w) > 1])

def load_new_datasets():
    """加载并解析数据集2与数据集3的新流入样本"""
    print("\n[INFO] 开始扫描数据集2与数据集3中的新流入文件...")
    new_data = []
    
    # -----------------------------------------
    # 1. 提取数据集 2 (遍历半结构化目录)
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
                        # 规避底层引擎兼容性问题，将 DataFrame 转换为 NumPy 数组并展平
                        text_content = " ".join(df.astype(str).to_numpy().flatten())
                    elif ext == '.xlsx':
                        df = pd.read_excel(file_path, nrows=50)
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
        print(f"[WARNING] 路径不存在: {DATASET2_DIR}")

    # -----------------------------------------
    # 2. 提取数据集 3 (多引擎容错读取机制)
    # -----------------------------------------
    if DATASET3_FILE.exists():
        print(f"[INFO] 检测到数据集3，启动多引擎解析机制...")
        df3 = None
        try:
            # 优先级 1：调用 openpyxl 引擎解析标准 Excel 格式
            df3 = pd.read_excel(DATASET3_FILE, engine='openpyxl')
            print("[INFO] 成功以 Excel 格式加载数据集3。")
        except Exception as e1:
            print(f"[WARNING] Excel引擎解析异常，启动备用 CSV 解析方案...")
            try:
                # 优先级 2：采用 UTF-8 编码读取伪装的 CSV 文件
                df3 = pd.read_csv(DATASET3_FILE, encoding='utf-8-sig')
                print("[INFO] 成功以 UTF-8 CSV 格式加载数据集3。")
            except Exception as e2:
                try:
                    # 优先级 3：回退至 GBK 编码读取
                    df3 = pd.read_csv(DATASET3_FILE, encoding='gbk')
                    print("[INFO] 成功以 GBK CSV 格式加载数据集3。")
                except Exception as e3:
                    print(f"[ERROR] 数据集3解析失败，请检查文件完整性。")
        
        # 将表格行数据序列化为文本特征
        if df3 is not None:
            for index, row in df3.iterrows():
                # 对于 Series 结构，转换为原生 List 进行特征拼接
                text_content = " ".join(row.astype(str).tolist())
                text_content = text_content.replace('nan', ' ').strip()
                
                if len(text_content) > 5:
                    new_data.append({
                        '文件名': f"匿名文件_行号_{index+1}", 
                        '数据来源': '数据集3',
                        '原始文本': text_content
                    })
    else:
        print(f"[WARNING] 文件不存在: {DATASET3_FILE}")

    # -----------------------------------------
    # 3. 统一执行文本清洗与特征过滤
    # -----------------------------------------
    df_new = pd.DataFrame(new_data)
    if not df_new.empty:
        print(f"[SUCCESS] 成功提取 {len(df_new)} 条有效样本，启动特征空间映射...")
        df_new['清洗后文本特征'] = df_new['原始文本'].apply(clean_text)
        # 剔除清洗后产生的空特征数据
        df_new = df_new[df_new['清洗后文本特征'].str.len() > 0].copy()
    else:
        print("[ERROR] 未提取到有效文本数据，请检查输入源。")
        
    return df_new

# =====================================================
# 分类模型训练与综合量化评价模块
# =====================================================
def run_classification_pipeline():
    print("=" * 60)
    print("[INFO] 启动问题二：多源异构数据迁移分类与综合评价系统")
    print("=" * 60)

    # 1. 载入问题一的聚类结果作为监督学习训练集
    if not TRAIN_FILE.exists():
        print(f"[ERROR] 缺失前置依赖：未找到训练集文件 {TRAIN_FILE}")
        return
    
    print("\n[PROCESS] 阶段一：加载历史主题分类体系及训练样本...")
    df_train = pd.read_csv(TRAIN_FILE)
    X_train_text = df_train['清洗后文本特征'].fillna('')
    y_train = df_train['聚类标签']
    
    # 统一特征向量空间参数，确保跨数据集维度一致
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    
    # 训练引入类权重平衡的逻辑回归模型
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    print("[SUCCESS] 基线分类模型训练完成 (基于先验知识库)。")

    # 2. 载入并处理新流入的泛化测试集
    df_new = load_new_datasets()
    if df_new.empty:
        print("[ERROR] 分类流程中止：测试集为空。")
        return
        
    print("\n[PROCESS] 阶段二：新流入数据特征映射与后验概率分布计算...")
    X_new = vectorizer.transform(df_new['清洗后文本特征'])
    
    # 获取各类别的预测概率矩阵
    probs = model.predict_proba(X_new)
    
    # 3. 计算量化评价指标与异常识别策略
    print("\n[PROCESS] 阶段三：计算综合量化评价指标并执行异常样本拦截...")
    
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
        
        # 异常样本识别与拦截阈值设定
        if top1_prob < 0.40:
            status_flags.append("无法明确归类 (低置信度)")
            predictions.append(-1)
        elif margin < 0.15:
            status_flags.append(f"多类别特征 (倾向类别 {model.classes_[top1_class]} 与 {model.classes_[top2_class]})")
            predictions.append(-1) 
        else:
            status_flags.append("自动归类成功")
            predictions.append(model.classes_[top1_class])

    # 4. 汇总分类结果与指标参数
    df_new['预测类别'] = predictions
    df_new['最大置信度'] = confidences
    df_new['类别混淆度'] = ambiguity_scores
    df_new['信息熵'] = entropies
    df_new['处理状态'] = status_flags
    
    df_new.drop(columns=['原始文本', '清洗后文本特征']).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n[SUCCESS] 分类预测结果已成功导出至: {OUTPUT_FILE}")
    
    # 5. 生成系统宏观量化评价报告
    success_rate = (df_new['预测类别'] != -1).mean() * 100
    avg_entropy = np.mean(entropies)
    
    report = f"""
==================================================
[REPORT] 问题二：分类系统综合量化评价报告
==================================================
【1. 模型迁移与适用性评价】
- 测试样本总数: {len(df_new)} 份
- 模型有效识别率: {success_rate:.2f}%
- 待人工复核率: {100 - success_rate:.2f}% 

【2. 分类合理性与可解释性评价】
- 全局平均信息熵 (H): {avg_entropy:.4f} (值越低表明模型分类确定性越高)
- 平均预测置信度: {np.mean(confidences):.4f}

【3. 异常数据识别与处理机制说明】
基于设定的边界阈值，模型共拦截了 {sum(df_new['预测类别'] == -1)} 份异常特征样本。
上述样本已统一标记为 -1 类别，将被移交至问题三的规则复核流程作进一步判断。
==================================================
"""
    print(report)
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[SUCCESS] 综合量化评价报告已保存至: {METRICS_FILE}")

if __name__ == "__main__":
    run_classification_pipeline()