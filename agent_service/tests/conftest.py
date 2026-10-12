import sys
from pathlib import Path

# 让测试以 agent_service 为根导入模块（与 uvicorn app:app 的运行方式一致）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
