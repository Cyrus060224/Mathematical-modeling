import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# 项目根目录与文件路径配置
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "results" / "problem2_predictions.csv"
OUTPUT_FILE = BASE_DIR / "results" / "problem3_final_decision.csv"

# 真实数据集 4 场景定义: (可用工时上限, 人工复核能力上限, 自动归档能力上限)
SCENARIOS = {
    "S1": (60, 200, 1200),
    "S2": (80, 300, 1500),
    "S3": (100, 400, 1800)
}

# =====================================================
# 权重评价与优先级划分模型 (RUN Model)
# =====================================================
def calculate_review_score(df):
    """
    计算人工复核优先级综合得分
    """
    df['risk_score'] = (1 - df['最大置信度']) * 100
    
    df['urgency_score'] = 20
    money_categories = [2, 4] 
    df.loc[df['预测类别'].isin(money_categories), 'urgency_score'] += 50
    
    df['necessity_score'] = 10
    df.loc[df['处理状态'] != "自动归类成功", 'necessity_score'] += 70
    
    w_risk, w_urgency, w_necessity = 0.4, 0.3, 0.3
    df['total_priority_score'] = (
        df['risk_score'] * w_risk + 
        df['urgency_score'] * w_urgency + 
        df['necessity_score'] * w_necessity
    )
    
    df['优先级等级'] = pd.qcut(df['total_priority_score'], 3, labels=['低', '中', '高'], duplicates='drop')
    
    # 必须降序排列：高分在前（高风险），低分在后（最安全）
    return df.sort_values(by='total_priority_score', ascending=False)

# =====================================================
# 多场景资源约束模拟与决策优化 (两端截断与积压控制)
# =====================================================
def run_scenario_optimization(df):
    print("=" * 60)
    print("[INFO] 启动问题三：多场景资源约束优化与决策系统")
    print("=" * 60)
    
    df = calculate_review_score(df)
    final_results = df.copy()
    total_files = len(df)
    
    for name, (limit_h, limit_manual, limit_auto) in SCENARIOS.items():
        dynamic_time_per_file = limit_h / limit_manual 
        print(f"\n[PROCESS] 启动场景模拟: {name} (人工限额: {limit_manual} 份 | 自动限额: {limit_auto} 份)")
        
        # 核心逻辑：“两头掐尖法”分配状态
        def assign_decision(rank):
            # 1. 掐头：最高风险的交给人工 (限制数量)
            if rank <= limit_manual:
                return '人工复核'
            # 2. 掐尾：最低风险的交给系统自动归档 (排名在总数最后 limit_auto 范围内的)
            elif rank > (total_files - limit_auto):
                return '自动归档'
            # 3. 中间：既没额度人工看，又不敢自动归档的，积压到明天
            else:
                return '暂缓处理(系统积压)'
        
        # 赋予每一行排名 (1 到 N)
        ranks = np.arange(total_files) + 1
        final_results[f'复核决策_{name}'] = [assign_decision(r) for r in ranks]
        
        # 统计指标
        manual_count = (final_results[f'复核决策_{name}'] == '人工复核').sum()
        auto_count = (final_results[f'复核决策_{name}'] == '自动归档').sum()
        backlog_count = (final_results[f'复核决策_{name}'] == '暂缓处理(系统积压)').sum()
        
        print(f"[RESULT] {name} 决策完毕：人工复核 {manual_count} 份 | 自动归档 {auto_count} 份 | 积压未处理 {backlog_count} 份")

    final_results.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n[SUCCESS] 多场景决策结果已成功保存至: {OUTPUT_FILE}")
    
    return final_results

if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"[ERROR] 缺失前置依赖：未找到输入文件 {INPUT_FILE}")
    else:
        df_results = pd.read_csv(INPUT_FILE)
        final_df = run_scenario_optimization(df_results)