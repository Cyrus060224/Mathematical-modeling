import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# 📍 路径与参数配置
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "results" / "problem2_predictions.csv"
OUTPUT_FILE = BASE_DIR / "results" / "problem3_final_decision.csv"

# 假设的人工复核成本参数 (根据实际比赛经验设定)
TIME_PER_FILE = 10 / 60  # 每份文件复核需 10 分钟 (0.167 小时)
COST_PER_FILE = 20       # 每份文件人工成本 20 元

# 数据集 4 场景定义: (可用小时, 可用金额)
SCENARIOS = {
    "场景1 (低资源)": (50, 10000),
    "场景2 (中资源)": (150, 30000),
    "场景3 (高资源)": (500, 100000)
}

# =====================================================
# 🧠 核心：构建权重评价模型 (RUN Model)
# =====================================================
def calculate_review_score(df):
    """
    计算复核分 $S$ = w1*Urgency + w2*Risk + w3*Necessity
    """
    # 1. 错分风险 (Risk): 1 - 置信度
    df['risk_score'] = (1 - df['最大置信度']) * 100
    
    # 2. 紧急程度与关键主题 (Urgency): 
    # 重点考虑：涉及资金分配、投资、收入的类别 (假设类别2和4)
    # 以及文件名包含敏感词的文件
    df['urgency_score'] = 20  # 基础分
    money_categories = [2, 4] # 根据第一问结果确定
    df.loc[df['预测类别'].isin(money_categories), 'urgency_score'] += 50
    
    # 3. 复核必要性 (Necessity): 
    # 处理状态为“无法明确归类”或“多类别特征”的优先级最高
    df['necessity_score'] = 10
    df.loc[df['处理状态'] != "自动归类成功", 'necessity_score'] += 70
    
    # 4. 综合加权评分 (权重可根据论文需求调整)
    w_risk, w_urgency, w_necessity = 0.4, 0.3, 0.3
    df['total_priority_score'] = (
        df['risk_score'] * w_risk + 
        df['urgency_score'] * w_urgency + 
        df['necessity_score'] * w_necessity
    )
    
    # 划分高、中、低等级 (按百分位数)
    df['优先级等级'] = pd.qcut(df['total_priority_score'], 3, labels=['低', '中', '高'])
    
    return df.sort_values(by='total_priority_score', ascending=False)

# =====================================================
# 📉 场景决策模拟 (背包优化逻辑)
# =====================================================
def run_scenario_optimization(df):
    print("=" * 60)
    print("🚀 问题三：多场景资源约束优化系统启动")
    print("=" * 60)
    
    # 计算基础分
    df = calculate_review_score(df)
    
    final_results = df.copy()
    
    for name, (limit_h, limit_m) in SCENARIOS.items():
        print(f"\n📊 正在模拟 {name} (预算: {limit_h}h / {limit_m}元)")
        
        # 按照分数从高到低尝试选择
        temp_df = df.copy()
        temp_df['cumulative_time'] = (np.arange(len(temp_df)) + 1) * TIME_PER_FILE
        temp_df['cumulative_cost'] = (np.arange(len(temp_df)) + 1) * COST_PER_FILE
        
        # 判断是否在约束范围内
        temp_df[f'is_reviewed_{name}'] = (
            (temp_df['cumulative_time'] <= limit_h) & 
            (temp_df['cumulative_cost'] <= limit_m)
        )
        
        reviewed_count = temp_df[f'is_reviewed_{name}'].sum()
        coverage = (reviewed_count / len(df)) * 100
        
        print(f"✅ 决策完成：建议复核 {reviewed_count} 份文件 (覆盖率: {coverage:.2f}%)")
        
        # 将决策结果合并回主表
        final_results[f'复核决策_{name}'] = temp_df[f'is_reviewed_{name}'].map({True: '人工复核', False: '直接归档'})

    # 保存最终大表
    final_results.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n💾 所有场景决策结果已保存至: {OUTPUT_FILE}")
    
    return final_results

if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"❌ 找不到输入文件，请先运行问题二脚本。")
    else:
        df_results = pd.read_csv(INPUT_FILE)
        final_df = run_scenario_optimization(df_results)