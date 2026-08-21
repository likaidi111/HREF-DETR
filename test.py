# test_import.py
try:
    import numpy as np
    print("numpy 导入成功")
    
    from ppdet.modeling.transformers import AMFEM, MKSE
    print("AMFEM / MKSE 导入成功")
    
except ImportError as e:
    print(f"导入错误: {e}")
except Exception as e:
    print(f"其他错误: {e}")
