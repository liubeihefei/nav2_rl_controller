import numpy as np
import onnxruntime as ort

# 你的数据
sector_obs = [0.85, 1, 1.05, 1.2, 10, 10, 10, 10, 3.65, 3.2, 3.1, 3.05, 3.2, 3.8, 3, 2.7, 2.3, 2.05, 2.05, 1.95]
target_info = [2.50807, 0.282852, 0.959164]
action_info = [0, 0.915783]

# 组合并重复51次
pattern = sector_obs + target_info + action_info
# full_input = pattern * 51  # 重复51次
full_input = pattern

# 转换为模型输入
input_array = np.array(full_input, dtype=np.float32).reshape(1, -1)

# 加载模型并推理
model_path = "../logs/SAC_actor.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 推理
outputs = session.run([output_name], {input_name: input_array})
result = outputs[0]

print(f"输入形状: {input_array.shape}")
print(f"输出形状: {result.shape}")
print(f"输出: {result}")