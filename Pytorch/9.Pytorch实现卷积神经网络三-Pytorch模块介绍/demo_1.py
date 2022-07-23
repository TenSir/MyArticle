import torch

# 检查当前设备的GPU可用情况
print(torch.cuda.is_available())
# 获取GPU的数量
print(torch.cuda.device_count())
# 获取GPU的名称
print(torch.cuda.get_device_name())
# 显存的使用量
print(torch.cuda.memory_usage())
# 清空缓存
print(torch.cuda.empty_cache())