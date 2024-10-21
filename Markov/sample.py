import numpy as np
import pandas as pd

# 模拟数据，假设有 5 个客户，每个客户有 10 个月的信用记录
np.random.seed(42)

data = {
    'cust': np.repeat(np.arange(1, 6), 10),  # 5 个客户，每个客户 10 个时间步长
    't': np.tile(np.arange(1, 11), 5),      # 每个客户从第 1 到第 10 个月
    'y': np.random.choice([0, 1, 2, 3], size=50, p=[0.6, 0.2, 0.1, 0.1])  # 逾期状态的分布
}

df = pd.DataFrame(data)

# 展示模拟的前几行数据
#print(df)

# 按照客户和时间步长排序，以确保数据的顺序正确
df = df.sort_values(by=['cust', 't'])

# 创建 'y_next' 列，表示每个客户的下一时间步长的状态
df['y_next'] = df.groupby('cust')['y'].shift(-1)

# 删除 'y_next' 为 NaN 的行（这些是每个客户的最后一个时间步）
df_clean = df.dropna(subset=['y_next'])

print(df_clean)

# 创建转移矩阵，统计从状态 'y' 转移到 'y_next' 的次数
transition_matrix = pd.crosstab(df_clean['y'], df_clean['y_next'], normalize='index')

# 展示转移矩阵（表示从一种状态转移到另一种状态的概率）
print("转移矩阵 (Markov 模型):")
print(transition_matrix)

# 示例：查看状态 1（逾期 1 个月）的转移概率
print("\n状态 1 的转移概率:")
print(transition_matrix.loc[1])

############################################### NN Part ###############################################

# 特征和目标变量
X = df[['cust', 'grade', 't']]
y = df['y']

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建MLP回归器
mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, alpha=1e-4,
                   solver='adam', verbose=10, random_state=1,
                   learning_rate_init=.01)

# 训练模型
mlp.fit(X_train, y_train)

# 预测
y_pred = mlp.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差 (MSE): {mse}")