import torch
import torch.nn as nn

# 假设 x 是输入数据，尺寸为 (batch_size, num_classes)，比如 (4, 3)
x = torch.tensor([[1.0, 2.0, 3.0],
                  [2.0, 3.0, 4.0],
                  [3.0, 4.0, 5.0],
                  [4.0, 5.0, 6.0]])

# 创建 LogSoftmax 操作，并指定 dim 参数
log_softmax = nn.LogSoftmax(dim=1)  # 在第1个维度上进行 LogSoftmax，即对 num_classes 这个维度进行操作

# 对输入数据进行 LogSoftmax 操作
output = log_softmax(x)

print(output)


