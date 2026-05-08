import os
from collections import Counter

# ⚠️ 注意：把这里引号里的路径，换成你存放“数据集1”的实际文件夹路径
# 前面的 r 不要删，它可以防止 Windows 路径里的反斜杠转义报错
folder_path = r"C:\Users\wxcyr\Desktop\数学建模赛题和论文模板\B题\B题数据集\数据集1：历史真实文件数据"

def count_files(path):
    print(f"老葛正在帮你扫描文件夹: {path}\n文件有点多，让子弹飞一会儿...")
    file_extensions = []
    total_files = 0

    for root, dirs, files in os.walk(path):
        for file in files:
            total_files += 1
            # 提取后缀名并统一转为小写，防止出现 .PDF 和 .pdf 算两类的情况
            ext = os.path.splitext(file)[1].lower()
            file_extensions.append(ext)

    print(f"\n--- 扫描完成，准备汇报 ---")
    print(f"总计发现文件: {total_files} 个")
    print(f"各类格式分布情况如下：")
    
    # 统计各个后缀名的数量并按数量降序排列
    ext_counts = Counter(file_extensions).most_common()
    for ext, count in ext_counts:
        print(f"格式 {ext if ext else '无后缀'}: {count} 个")

if __name__ == "__main__":
    count_files(folder_path)