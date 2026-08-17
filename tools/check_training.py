import yaml

# 读取配置
with open('configs/rtdetr/_base_/rtdetr_r50vd.yml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("="*60)
print("配置文件检查")
print("="*60)

# 检查RTDETRTransformer配置
rt_config = config.get('RTDETRTransformer', {})

# 检查是否有不支持的参数
problematic_keys = ['use_progressive_decoder', 'progressive_mode', 'use_scale_selector']
has_problem = False

for key in problematic_keys:
    if key in rt_config:
        print(f"❌ 发现问题参数: {key} = {rt_config[key]}")
        has_problem = True

if not has_problem:
    print("✅ RTDETRTransformer配置正确，无冲突参数")
else:
    print("\n⚠️  请移除上述参数！")
    print("这些参数导致您的训练loss异常高！")

# 检查 AMFEM 配置
amfem_config = config.get('AMFEM', {})
print(f"\n📋 AMFEM配置:")
print(f"  - out_channels: {amfem_config.get('out_channels')}")
print(f"  - use_ffn: {amfem_config.get('use_ffn')}")
print(f"  - use_eca: {amfem_config.get('use_eca')}")

mkse_config = config.get('MKSE', {})
print(f"\n📋 MKSE配置:")
print(f"  - act: {mkse_config.get('act')}")

# 检查HybridEncoder配置
encoder_config = config.get('HybridEncoder', {})
print(f"\n📋 HybridEncoder配置:")
print(f"  - use_dsf: {encoder_config.get('use_dsf')}")
print(f"  - dim_feedforward: {encoder_config.get('encoder_layer', {}).get('dim_feedforward')}")
print(f"  - dropout: {encoder_config.get('encoder_layer', {}).get('dropout')}")

print("\n" + "="*60)