import pandas as pd
import numpy as np

# 读取原始 CSV 文件
file_path = 'data/cpu.csv'
df = pd.read_csv(file_path)

# 生成随机的 CPU 使用率数据
num_rows = len(df)
random_cpu_usage = np.random.uniform(0, 100, num_rows).round(1)

# 替换原有数据
df['cpu_usage'] = random_cpu_usage

# 保存新的 CSV 文件
new_file_path = 'data/cpu_random.csv'
df.to_csv(new_file_path, index=False)

print(f"随机化的 CPU 使用率数据已保存到 {new_file_path}")

# 显示新数据的前几行
print("\n新数据的前几行：")
print(df.head())