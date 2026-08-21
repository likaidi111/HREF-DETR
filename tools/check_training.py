import yaml

# Load config
with open('configs/rtdetr/_base_/rtdetr_r50vd.yml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print("="*60)
print("Config file chec")
print("="*60)

# Check RTDETRTransformer config
rt_config = config.get('RTDETRTransformer', {})

# Check for unsupported parameters
problematic_keys = ['use_progressive_decoder', 'progressive_mode', 'use_scale_selector']
has_problem = False

for key in problematic_keys:
    if key in rt_config:
        print(f"❌ Found problematic parameter: {key} = {rt_config[key]}")
        has_problem = True

if not has_problem:
    print("✅ config is valid; no conflicting parameters")
else:
    print("\n⚠️  Please remove the parameters above!")
    print("These parameters can cause abnormally high training loss!")

# 检查 AMFEM 配置
amfem_config = config.get('AMFEM', {})
print(f"\n📋 AMFEM config:")
print(f"  - out_channels: {amfem_config.get('out_channels')}")
print(f"  - use_ffn: {amfem_config.get('use_ffn')}")
print(f"  - use_eca: {amfem_config.get('use_eca')}")

mkse_config = config.get('MKSE', {})
print(f"\n📋 MKSE config :")
print(f"  - act: {mkse_config.get('act')}")

# check HybridEncoder config
encoder_config = config.get('HybridEncoder', {})
print(f"\n📋 HHybridEncoder config:")
print(f"  - use_dsf: {encoder_config.get('use_dsf')}")
print(f"  - dim_feedforward: {encoder_config.get('encoder_layer', {}).get('dim_feedforward')}")
print(f"  - dropout: {encoder_config.get('encoder_layer', {}).get('dropout')}")

print("\n" + "="*60)
