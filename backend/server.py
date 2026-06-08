"""
server.py — Medical VQA FastAPI 后端推理服务

使用 Qwen2-VL-2B-Instruct + LoRA adapter (SFT v2) 进行医学视觉问答。
支持流式输出（SSE），供前端实时展示生成内容。

当前最优模型：SFT v2 — 准确率 82.3%
"""

import io
import os
import threading
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from peft import PeftModel

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = os.path.join(PROJECT_ROOT, "models", "qwen", "Qwen2-VL-2B-Instruct")
ADAPTER_DIR = os.path.join(PROJECT_ROOT, "outputs", "sft_lora_v2")
TEMPERATURE = 0.0

# CPU 模式下减少 token 数，否则等待时间过长
_IS_CPU = not torch.cuda.is_available()
MAX_NEW_TOKENS = 128 if _IS_CPU else 512

SYSTEM_PROMPT = (
    "你是一位资深的皮肤科医学专家，拥有二十年的临床诊断经验。"
    "请根据用户提供的皮肤病变图像，给出专业、准确、详细的分析。"
    "如果图像质量较差或无法判断，请如实说明。"
    "你只能用中文回答。回答末尾请以'诊断结论：疾病名称'格式给出明确的诊断。"
)

# ──────────────────────────────────────────────
# 全局模型实例
# ──────────────────────────────────────────────
model = None
processor = None
device = None
model_lock = threading.Lock()


def load_model():
    """加载模型和 processor（线程安全，只加载一次）。"""
    global model, processor, device

    with model_lock:
        if model is not None:
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        bf16_ok = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if bf16_ok else (torch.float16 if device == "cuda" else torch.float32)

        print(f"[server] 设备: {device}, dtype: {dtype}")
        if device == "cpu":
            print("[server] ⚠ 未检测到 GPU！CPU 推理较慢，每次回答约需 1-3 分钟（max_new_tokens=128）")
            print("[server] ⚠ 如有 NVIDIA 显卡，请安装 CUDA 版 PyTorch：")
            print("[server]     pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121")

        # 加载 processor — 优先从 adapter 目录加载（含 tokenizer + chat_template）
        adapter_path = Path(ADAPTER_DIR)
        if adapter_path.exists():
            print(f"[server] 从 adapter_dir 加载 processor: {ADAPTER_DIR}")
            try:
                processor = AutoProcessor.from_pretrained(ADAPTER_DIR)
            except Exception:
                print(f"[server] adapter processor 加载失败，回退到 base_model: {BASE_MODEL}")
                processor = AutoProcessor.from_pretrained(BASE_MODEL)
        else:
            print(f"[server] adapter_dir 不存在，从 base_model 加载 processor: {BASE_MODEL}")
            processor = AutoProcessor.from_pretrained(BASE_MODEL)

        # 配置图像处理器
        if hasattr(processor, "image_processor"):
            processor.image_processor.min_pixels = 65536   # 256×256
            processor.image_processor.max_pixels = 262144  # 512×512

        # 加载基础模型
        print(f"[server] 加载 base model: {BASE_MODEL}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
        )
        if device == "cuda":
            model = model.to("cuda")
            print(f"[server] 模型已移至 GPU，显存使用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # 挂载 LoRA adapter
        if adapter_path.exists():
            print(f"[server] 挂载 LoRA adapter: {ADAPTER_DIR}")
            model = PeftModel.from_pretrained(model, ADAPTER_DIR)
            model = model.merge_and_unload()
        else:
            print("[server] ⚠ 未找到 LoRA adapter，使用基础模型（效果可能较差）")

        model.eval()
        print(f"[server] 模型加载完成 ✓  (max_new_tokens={MAX_NEW_TOKENS})")


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(title="Medical VQA API (Qwen2-VL)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # 必须为 False，否则 CORS 规范与 allow_origins="*" 冲突，浏览器拒绝读取流式响应
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.on_event("startup")
def startup():
    load_model()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "Qwen2-VL-2B-Instruct",
        "model_loaded": model is not None,
        "device": device,
        "adapter": ADAPTER_DIR,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


@app.post("/api/vqa")
async def vqa(
    image: UploadFile = File(...),
    question: str = Form(default="请仔细观察这张皮肤病变图像，给出最可能的诊断及诊断依据。"),
    stream: bool = Form(default=True),
):
    """医学 VQA 推理接口。

    - **image**: 上传的医学图像（JPEG / PNG）
    - **question**: 用户问题
    - **stream**: 是否使用流式输出（默认 true）
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    # 读取并验证图像
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"[server] 收到请求 — 图片尺寸: {pil_image.size}, 问题: {question[:50]}...")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法解析图像: {e}")

    # 构建消息
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ]},
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        images=[pil_image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"[server] 输入 token 数: {inputs['input_ids'].shape[1]}, 开始生成...")

    if stream:
        return StreamingResponse(
            _stream_generate(inputs),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            },
        )
    else:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=(TEMPERATURE > 0),
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
            )

        input_len = inputs["input_ids"].shape[1]
        output = processor.batch_decode(
            generated_ids[:, input_len:], skip_special_tokens=True
        )[0]
        print(f"[server] 生成完成，回答长度: {len(output)} 字符")

        return {"answer": output.strip()}


def _stream_generate(inputs: dict):
    """生成器函数：在独立线程中运行 generate，通过 streamer 逐 token yield SSE 事件。

    线程中如果出错，会将异常信息以 SSE 事件发出，避免前端无限等待。
    """
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=(TEMPERATURE > 0),
        temperature=TEMPERATURE if TEMPERATURE > 0 else None,
        streamer=streamer,
    )

    # 在独立线程中运行 generate，streamer 在主线程迭代
    generation_error = []

    def _run_generate():
        try:
            with torch.no_grad():
                model.generate(**generation_kwargs)
        except Exception as e:
            generation_error.append(f"{type(e).__name__}: {e}")
            traceback.print_exc()

    thread = threading.Thread(target=_run_generate, daemon=True)
    thread.start()

    try:
        for token_text in streamer:
            # 如果生成线程报错，把错误发给前端
            if generation_error:
                yield f"data: [ERROR] {generation_error[0]}\n\n"
                yield "data: [DONE]\n\n"
                return
            yield f"data: {token_text}\n\n"
        yield "data: [DONE]\n\n"
        print("[server] 流式生成完成")
    except GeneratorExit:
        print("[server] 客户端断开连接")
    finally:
        thread.join(timeout=10)


# ──────────────────────────────────────────────
# 直接运行入口
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Medical VQA 后端服务 (Qwen2-VL)")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Adapter:    {ADAPTER_DIR}")
    print(f"设备: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU (慢)'}")
    print(f"Max New Tokens: {MAX_NEW_TOKENS}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=7860)
