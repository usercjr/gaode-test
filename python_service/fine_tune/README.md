# Fine-tuned Model Utilities

This folder contains a standalone script to use your fine-tuned Wenxin model together with the local PaddleNLP classifier.

It does NOT modify or run inside the FastAPI service. Keep your production server running while you experiment here.

## 1) Create/activate venv

PowerShell:

```
Set-Location D:\gaode_test\python_service
py -3.10 -m venv .venv310
.\.venv310\Scripts\Activate.ps1
```

## 2) Install deps (isolated)

```
pip install -U pip
pip install -r .\fine_tune\requirements-train.txt
```

If PaddlePaddle wheels are slow, consider a mirror such as:
`-i https://mirror.baidu.com/pypi/simple` or `https://pypi.tuna.tsinghua.edu.cn/simple`.

## 3) Set environment variables

```
$env:BAIDU_LLM_API_KEY="YOUR_KEY"
$env:BAIDU_BASE_URL="https://aistudio.baidu.com/llm/lmapi"
$env:BAIDU_API_VERSION="v3"          # or v1, depending on your model
$env:BAIDU_MODEL="ernie-3.5-8k"      # or your fine-tuned model id
$env:BEST_MODEL_DIR="D:\gaode_test\best_model"
```

## 4) Run

```
python .\fine_tune\AI_Travel_Assistant.py
```

Expected:
- A quick preference classification printout
- Generated itinerary text printed in the console

## Notes

- Keep secrets in env vars; do not hardcode keys.
- If your Wenxin fine-tuned model requires a specific API version/base URL, adjust `BAIDU_API_VERSION` and `BAIDU_BASE_URL` accordingly.
- This script merges system instructions and user intent as the first `user` message to satisfy providers that require the first message to be user/assistant.
*** End Patch```} />

