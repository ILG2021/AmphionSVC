import torch

model = torch.load("pretrained/nsf_hifigan/110000.pt", map_location=torch.device('cpu') )
# 获取模型的状态字典
model_state_dict = model['generator']

# 打印模型的结构（状态字典的键）
for key in model_state_dict.keys():
    print(key)