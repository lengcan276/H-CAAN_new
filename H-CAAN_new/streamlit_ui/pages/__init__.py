# 显式导入所有页面
try:
    from . import data_page
    from . import model_page
    from . import training_page
    from . import results_page
    from . import paper_page
except ImportError as e:
    print(f"Warning: Failed to import page module: {e}")