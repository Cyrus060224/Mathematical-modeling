import os
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report





# =====================================================
# 1. 配置区域
# =====================================================

# 输入文件
INPUT_FILE = "results/problem1_clustering_results.csv"

# 输出文件
OUTPUT_FILE = "results/problem2_result.csv"

# 真实文本列
TEXT_COLUMN = "清洗后文本特征"

# 真实类别列
LABEL_COLUMN = "聚类标签"


# =====================================================
# 2. 规则关键词映射（基于你的KMeans结果）
# =====================================================

RULE_KEYWORDS = {

    # 类别0：统计/数字/指标
    "数字": 0,
    "指数": 0,
    "数据": 0,
    "人数": 0,
    "生产总值": 0,
    "机构": 0,
    "计算": 0,

    # 类别1：工业/制造业
    "制造业": 1,
    "服务业": 1,
    "工业": 1,
    "加工": 1,
    "制品业": 1,
    "批发": 1,
    "热力": 1,
    "信息技术": 1,

    # 类别2：居民消费/收入
    "收入": 2,
    "支出": 2,
    "消费": 2,
    "居民": 2,
    "基金": 2,
    "人均": 2,

    # 类别3：行业/城市/指标
    "城市": 3,
    "行业": 3,
    "指标": 3,
    "规模": 3,
    "登记注册": 3,

    # 类别4：投资/区域经济
    "投资": 4,
    "面积": 4,
    "全国": 4,
    "亿元": 4,
    "黑龙江": 4,
    "辽宁": 4,
    "上海": 4,

    # 类别5：情感/消费倾向
    "情感": 5,
    "积极": 5,
    "消极": 5,
    "倾向": 5,
    "消费品": 5,
    "耐用": 5
}


# =====================================================
# 3. 工具函数
# =====================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # 去除特殊字符
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)

    # 去除多余空格
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def rule_based_classification(text):

    for keyword, category in RULE_KEYWORDS.items():

        if keyword in text:
            return category

    return None


def is_unknown(label):

    if pd.isna(label):
        return True

    if str(label).strip() in ["", "未知", "未分类", "None", "nan"]:
        return True

    return False


# =====================================================
# 4. 读取数据
# =====================================================

print("=" * 60)
print("🚀 问题二分类系统启动")
print("=" * 60)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ 找不到输入文件: {INPUT_FILE}")

print(f"\n📂 正在读取文件: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"✅ 数据读取成功")
print(f"📊 总数据量: {len(df)}")

print("\n📋 CSV列名:")
print(df.columns)


# =====================================================
# 5. 文本预处理
# =====================================================

print("\n🧹 正在清洗文本...")

df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)


# =====================================================
# 6. 分离已分类和未分类数据
# =====================================================

classified_df = df[~df[LABEL_COLUMN].apply(is_unknown)].copy()

unclassified_df = df[df[LABEL_COLUMN].apply(is_unknown)].copy()

print(f"\n✅ 已分类数据: {len(classified_df)}")

print(f"⚠️ 未分类数据: {len(unclassified_df)}")


# =====================================================
# 7. 第一阶段：规则分类
# =====================================================

print("\n🧠 第一阶段：规则分类处理中...")

rule_predictions = []

for text in unclassified_df["clean_text"]:

    pred = rule_based_classification(text)

    rule_predictions.append(pred)

unclassified_df["rule_prediction"] = rule_predictions

rule_success_df = unclassified_df[
    unclassified_df["rule_prediction"].notna()
].copy()

rule_failed_df = unclassified_df[
    unclassified_df["rule_prediction"].isna()
].copy()

print(f"✅ 规则分类成功: {len(rule_success_df)}")

print(f"⚠️ 规则分类失败: {len(rule_failed_df)}")


# =====================================================
# 8. 第二阶段：机器学习分类
# =====================================================

if len(rule_failed_df) > 0:

    print("\n🤖 第二阶段：TF-IDF + LogisticRegression")

    X_train = classified_df["clean_text"]

    y_train = classified_df[LABEL_COLUMN]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=3000,
        random_state=42
    )

    model.fit(X_train_vec, y_train)

    print("✅ 模型训练完成")

    X_test_vec = vectorizer.transform(
        rule_failed_df["clean_text"]
    )

    ml_predictions = model.predict(X_test_vec)

    probabilities = model.predict_proba(X_test_vec)

    confidence_scores = np.max(probabilities, axis=1)

    rule_failed_df["ml_prediction"] = ml_predictions

    rule_failed_df["confidence"] = confidence_scores

    print("✅ 机器学习分类完成")


# =====================================================
# 9. 第三阶段：文本相似度修正
# =====================================================

if len(rule_failed_df) > 0:

    print("\n🔍 第三阶段：文本相似度修正")

    classified_texts = classified_df["clean_text"].tolist()

    classified_labels = classified_df[LABEL_COLUMN].tolist()

    all_texts = classified_texts + rule_failed_df["clean_text"].tolist()

    sim_vectorizer = TfidfVectorizer(max_features=3000)

    sim_matrix = sim_vectorizer.fit_transform(all_texts)

    train_matrix = sim_matrix[:len(classified_texts)]

    test_matrix = sim_matrix[len(classified_texts):]

    similarity = cosine_similarity(test_matrix, train_matrix)

    similarity_predictions = []

    for row in similarity:

        idx = np.argmax(row)

        similarity_predictions.append(classified_labels[idx])

    rule_failed_df["similarity_prediction"] = similarity_predictions

    final_predictions = []

    for _, row in rule_failed_df.iterrows():

        if row["confidence"] >= 0.7:
            final_predictions.append(row["ml_prediction"])
        else:
            final_predictions.append(row["similarity_prediction"])

    rule_failed_df["final_prediction"] = final_predictions

    print("✅ 相似度修正完成")


# =====================================================
# 10. 合并结果
# =====================================================

print("\n📦 正在合并结果...")

if len(rule_success_df) > 0:
    rule_success_df[LABEL_COLUMN] = rule_success_df["rule_prediction"]

if len(rule_failed_df) > 0:
    rule_failed_df[LABEL_COLUMN] = rule_failed_df["final_prediction"]

final_df = pd.concat([
    classified_df,
    rule_success_df,
    rule_failed_df
], ignore_index=True)

print(f"✅ 最终数据量: {len(final_df)}")


# =====================================================
# 11. 分类统计
# =====================================================

print("\n📈 分类统计")

print("=" * 60)

counts = final_df[LABEL_COLUMN].value_counts()

for category, count in counts.items():

    print(f"类别 {category} : {count}")

print("=" * 60)


# =====================================================
# 12. 保存结果
# =====================================================

drop_columns = [
    "clean_text",
    "rule_prediction",
    "ml_prediction",
    "confidence",
    "similarity_prediction",
    "final_prediction"
]

existing_columns = [
    col for col in drop_columns
    if col in final_df.columns
]

final_df = final_df.drop(columns=existing_columns)

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n✅ 结果保存成功: {OUTPUT_FILE}")


# =====================================================
# 13. 模型评估
# =====================================================

try:

    print("\n🧪 模型评估中...")

    X = classified_df["clean_text"]

    y = classified_df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    eval_vectorizer = TfidfVectorizer(max_features=5000)

    X_train_vec = eval_vectorizer.fit_transform(X_train)

    X_test_vec = eval_vectorizer.transform(X_test)

    eval_model = LogisticRegression(max_iter=3000)

    eval_model.fit(X_train_vec, y_train)

    y_pred = eval_model.predict(X_test_vec)

    print("\n📋 分类报告")

    print(classification_report(y_test, y_pred))

except Exception as e:

    print(f"\n⚠️ 模型评估失败: {e}")


# =====================================================
# 14. 完成
# =====================================================

print("\n🎉 问题二处理完成！")
print("=" * 60)