"""
一次解决！SAC Actor转换脚本
根据权重文件自动确定模型结构
"""
import torch
import torch.nn as nn
import argparse
import os
import sys

def analyze_state_dict(state_dict):
    """分析state_dict来确定模型结构"""
    print("分析权重文件结构...")

    # 收集所有层的权重和偏置
    layers = {}
    for key, weight in state_dict.items():
        if 'weight' in key:
            # 解析层编号，如 'trunk.0.weight' -> 0
            parts = key.split('.')
            if len(parts) >= 2 and parts[0] == 'trunk':
                layer_idx = int(parts[1])
                in_features = weight.shape[1]
                out_features = weight.shape[0]
                layers[layer_idx] = {
                    'in_features': in_features,
                    'out_features': out_features,
                    'has_bias': f'trunk.{layer_idx}.bias' in state_dict
                }
                print(f"  层 {layer_idx}: Linear({in_features}, {out_features})")

    # 排序层
    sorted_indices = sorted(layers.keys())

    # 确定输入维度和隐藏层
    obs_dim = layers[sorted_indices[0]]['in_features']
    action_dim = layers[sorted_indices[-1]]['out_features'] // 2  # 最后输出是 2*action_dim

    # 提取隐藏层大小
    hidden_layers = []
    for idx in sorted_indices[:-1]:  # 排除输出层
        hidden_layers.append(layers[idx]['out_features'])

    print(f"分析结果:")
    print(f"  输入维度 (obs_dim): {obs_dim}")
    print(f"  输出维度 (action_dim): {action_dim}")
    print(f"  隐藏层: {hidden_layers}")

    return obs_dim, action_dim, hidden_layers

def create_model_from_weights(weights_path):
    """根据权重文件创建模型"""
    print(f"加载权重文件: {weights_path}")

    # 加载state_dict
    state_dict = torch.load(weights_path, map_location='cpu')

    # 如果是完整的状态字典（包含'actor'键），提取actor部分
    if isinstance(state_dict, dict) and 'actor' in state_dict:
        print("检测到完整状态字典，提取actor权重")
        state_dict = state_dict['actor']

    # 分析模型结构
    obs_dim, action_dim, hidden_layers = analyze_state_dict(state_dict)

    # 创建匹配的模型
    class ExactSACActor(nn.Module):
        def __init__(self, obs_dim, action_dim, hidden_layers):
            super().__init__()

            # 构建MLP
            layers = []
            current_dim = obs_dim

            # 输入层和隐藏层
            for i, hidden_size in enumerate(hidden_layers):
                layers.append(nn.Linear(current_dim, hidden_size))
                layers.append(nn.ReLU(inplace=True))
                current_dim = hidden_size

            # 输出层（输出2*action_dim）
            layers.append(nn.Linear(current_dim, 2 * action_dim))

            self.trunk = nn.Sequential(*layers)
            self.log_std_bounds = [-5, 2]

        def forward(self, x):
            # 前向传播
            mu, log_std = self.trunk(x).chunk(2, dim=-1)

            # 约束log_std
            log_std = torch.tanh(log_std)
            log_std_min, log_std_max = self.log_std_bounds
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

            # 返回确定性输出
            return torch.tanh(mu)

    # 创建模型实例
    model = ExactSACActor(obs_dim, action_dim, hidden_layers)

    # 加载权重
    model.load_state_dict(state_dict)
    model.eval()

    print(f"✓ 模型创建成功")
    print(f"  输入: {obs_dim}维")
    print(f"  输出: {action_dim}维")

    return model, obs_dim, action_dim

def main():
    parser = argparse.ArgumentParser(description='一键转换SAC Actor模型')
    parser.add_argument('--weights', type=str, required=True,
                       help='权重文件路径 (.pth)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出ONNX文件路径 (默认: 同权重文件名，扩展名为.onnx)')
    parser.add_argument('--opset', type=int, default=13,
                       help='ONNX opset版本 (默认: 13)')

    args = parser.parse_args()

    print("=" * 60)
    print("SAC Actor模型一键转换工具")
    print("=" * 60)

    # 检查文件是否存在
    if not os.path.exists(args.weights):
        print(f"错误: 文件不存在: {args.weights}")
        return

    try:
        # 1. 根据权重创建模型
        model, obs_dim, action_dim = create_model_from_weights(args.weights)

        # 2. 设置输出路径
        if args.output is None:
            base_name = os.path.splitext(args.weights)[0]
            output_path = base_name + ".onnx"
        else:
            output_path = args.output

        print(f"\n导出ONNX模型到: {output_path}")

        # 3. 创建示例输入
        dummy_input = torch.randn(1, obs_dim)

        # 4. 测试模型
        with torch.no_grad():
            output = model(dummy_input)
            print(f"测试输出形状: {output.shape}")
            print(f"输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")

        # 5. 导出ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )

        print(f"✓ ONNX导出成功！")

        # 6. 验证
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX模型验证通过")

            # 文件信息
            file_size = os.path.getsize(output_path) / 1024  # KB
            print(f"文件大小: {file_size:.1f} KB")

        except ImportError:
            print("⚠ 未安装onnx，跳过验证")

    except Exception as e:
        print(f"✗ 转换失败: {e}")
        import traceback
        traceback.print_exc()

def simple_convert():
    """最简单的一键转换，无需参数"""
    print("执行最简单的一键转换...")

    # 假设权重文件在当前目录的上一级logs文件夹中
    weights_path = "../logs/SAC_actor.pth"

    if not os.path.exists(weights_path):
        print(f"请将权重文件放在: {weights_path}")
        weights_path = input("或输入权重文件路径: ").strip()

    # 自动处理
    state_dict = torch.load(weights_path, map_location='cpu')

    # 提取权重
    if isinstance(state_dict, dict) and 'actor' in state_dict:
        state_dict = state_dict['actor']

    # 创建模型
    class SimpleActor(nn.Module):
        def __init__(self):
            super().__init__()
            # 根据你的权重文件：3个Linear层 (0, 2, 4)
            # trunk.0: 25 -> 1024
            # trunk.2: 1024 -> 1024
            # trunk.4: 1024 -> 4 (2*action_dim)
            self.trunk = nn.Sequential(
                nn.Linear(25, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 4)  # 2 * 2
            )
            self.log_std_bounds = [-5, 2]

        def forward(self, x):
            mu, _ = self.trunk(x).chunk(2, dim=-1)
            return torch.tanh(mu)

    model = SimpleActor()
    model.load_state_dict(state_dict)
    model.eval()

    # 导出
    dummy_input = torch.randn(1, 25)
    torch.onnx.export(
        model,
        dummy_input,
        "SAC_actor.onnx",
        input_names=['input'],
        output_names=['output'],
        opset_version=13
    )

    print("✅ 转换完成！文件: SAC_actor.onnx")
    print("   输入: 25维")
    print("   输出: 2维 (范围: [-1, 1])")

if __name__ == "__main__":
    # 使用方法1: python script.py --weights ../logs/SAC_actor.pth
    # 使用方法2: 直接运行simple_convert()

    # 检查命令行参数
    if len(sys.argv) > 1 and '--weights' in ' '.join(sys.argv):
        main()
    else:
        simple_convert()