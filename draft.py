import torch

# 假设 q_values 是一个二维张量，形状为 (batch_size, action_dim)
q_values = torch.tensor([[0.1, 0.5, 0.3, 0.7, 0.2],
                         [0.4, 0.6, 0.8, 0.9, 0.1]])

# 取前三大值及其索引
topk_values, topk_indices = torch.topk(q_values, k=3, dim=1)

print(q_values.shape)
# 打印结果
print("Top-3 values:", topk_values.shape)
print("Top-3 indices:", topk_indices.shape)