import numpy as np
import onnxruntime as ort

# 你的数据
sector_obs = [10, 3.6, 3.2, 3.2, 3.25, 3.65, 4.4, 3.4, 3.1, 3.1, 2.85, 1.65, 1.2, 1.05, 0.95, 0.95, 0.9, 0.95, 0.95, 1.05]
target_info = [-0.288599, -0.95745, 2.59999]
action_info = [0, 0]

# 组合并重复51次
pattern = sector_obs + target_info + action_info
full_input = pattern * 51  # 重复51次

# 转换为模型输入
input_array = np.array(full_input, dtype=np.float32).reshape(1, -1)

# 加载模型并推理
model_path = "../model/SAC_actor.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 推理
outputs = session.run([output_name], {input_name: input_array})
result = outputs[0]

print(f"输入形状: {input_array.shape}")
print(f"输出形状: {result.shape}")
print(f"输出: {result}")