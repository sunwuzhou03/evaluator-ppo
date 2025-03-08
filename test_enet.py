import json
import numpy as np
import torch
import torch.nn as nn

# 你的代码部分，定义网络和加载权重
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Nne(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluate_network = nn.Sequential(
            layer_init(nn.Linear(6, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x):
        score = self.evaluate_network(x)
        return score

# 加载模型
evaluate_network = Nne()
evaluate_network.load_state_dict(torch.load('custom_envs/evaluate_network_weights_h.pth'))

# 定义特征变化的观测数据
obs = [
    [5, 5, 1, 1] + [3, 3],  # 基础特征
    [5, 5, 1, 1] + [3, 2],
    [5, 5, 1, 1] + [3, 1],
    [5, 5, 1, 1] + [3, 0],

    [5, 5, 1, 1] + [3, 3],
    [5, 5, 1, 1] + [2, 3],
    [5, 5, 1, 1] + [1, 3],
    [5, 5, 1, 1] + [0, 3],

    [-5, 5, 1, 1] + [3, 3],  # x先增大后减小
    [-2, 5, 1, 1] + [3, 3],
    [2, 5, 1, 1] + [3, 3],
    [5, 5, 1, 1] + [3, 3],

    [5, 5, 1, 1] + [3, 3],  # y减小
    [5, 3, 1, 1] + [3, 3],
    [5, 1, 1, 1] + [3, 3],
    [5, 0.1, 1, 1] + [3, 3],

    [5, 5, 1, 1] + [3, 3],  # |v|减小
    [5, 5, 0.5, 1] + [3, 3],
    [5, 5, 0.2, 1] + [3, 3],
    [5, 5, 0.1, 1] + [3, 3],

    [5, 5, 1, 1] + [3, 3],  # hp减小
    [5, 5, 1, 0.5] + [3, 3],
    [5, 5, 1, 0.2] + [3, 3],
    [5, 5, 1, 0.1] + [3, 3],
]

# 定义特征名称
feature_names = ["rx", "ry", "m_v", "m_hp", "t_hp", "t_qa"]

# 初始化结果字典
results = {feature: [] for feature in feature_names}

# 分析每个特征的变化对威胁评分的影响
for i, feature_name in enumerate(feature_names):
    # 提取固定特征值
    base_obs = obs[0].copy()
    base_value = base_obs[i]

    # 变化范围
    if feature_name in ["rx"]:
        values = np.linspace(-5, 5, 10)  # 线性变化范围
    elif feature_name in ["ry"]:
        values = np.linspace(0, 10, 10)  # 线性变化范围
    
    elif feature_name in ["m_v", "m_hp"]:
        values = np.linspace(0, 1, 10)  # 线性变化范围
    else:
        values = np.linspace(0, 3, 4)  # 其他特征的变化范围

    for value in values:
        obs_copy = base_obs.copy()
        obs_copy[i] = value
        threat_score = -evaluate_network(torch.tensor([obs_copy], dtype=torch.float32)).item()
        results[feature_name].append({"value": value, "threat": threat_score})

# 将结果保存为JSON文件
with open("threat_analysis.json", "w") as f:
    json.dump(results, f, indent=4)

print("威胁分析结果已保存到 threat_analysis.json 文件中。")