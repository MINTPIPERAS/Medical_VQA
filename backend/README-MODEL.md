# 皮肤病变多模态 VQA 系统 — Qwen2-VL 模型端

基于 **Qwen2-VL-2B-Instruct** + **LoRA** 的皮肤病变图像识别与医学视觉问答系统。

**当前最优模型：SFT v2 — 准确率 82.3%（600 条平衡评测）**

## 环境配置

```bash
conda create -n qwen_vqa python=3.10 -y
conda activate qwen_vqa
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.3 peft==0.14.0 accelerate==1.13.0 datasets pillow modelscope
```

**关键版本**：transformers 必须是 4.46.x（5.x 不兼容），torch 需 CUDA 版本。无 GPU 会自动回退 CPU。

## 模型下载

> 如果已经拿到完整的 `backend/` 目录，跳过本节。模型在 `models/` 下，adapter 在 `outputs/` 下。

基座模型约 4.4GB，两种下载方式任选其一：

### 方式一：ModelScope（推荐，国内快）

```bash
conda activate qwen_vqa
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen2-VL-2B-Instruct', cache_dir='./models')"
```

下载后模型位于 `models/qwen/Qwen2-VL-2B-Instruct/`。

### 方式二：HuggingFace 镜像

```bash
conda activate qwen_vqa
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir ./models/Qwen2-VL-2B-Instruct --resume-download
```

如果镜像不可用，设置环境变量后再试：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir ./models/Qwen2-VL-2B-Instruct --resume-download
```

## 目录结构

```
backend/
├── models/qwen/Qwen2-VL-2B-Instruct/   ← 基座模型（需下载，~4.4GB）
├── outputs/
│   ├── sft_lora_v2/                    ← **当前最优 LoRA adapter（147MB）**
│   ├── sft_lora/                       ← 旧版 adapter（已废弃）
│   └── cpt_lora/                       ← CPT 预训练 adapter
├── data/                               ← 训练/评测数据
├── Skin Lesion Dataset/                ← 皮肤病变图片数据集
├── scripts/eval_auto_score.py          ← 评测脚本
├── cpt_qwen.py / sft_qwen.py           ← 训练脚本
└── infer_qwen.py                       ← 推理脚本
```

## Python API（给交互系统同学）

以下代码中所有路径均为相对路径，基于项目根目录（`backend/`）解析。放到任何电脑上，只要保持目录结构不变即可运行。

```python
import os
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# 项目根目录 — 基于本脚本位置自动推断
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class SkinLesionModel:
    """皮肤病变诊断模型。加载基座模型 + SFT v2 LoRA adapter。

    使用方式:
        model = SkinLesionModel()                     # 使用默认路径
        model = SkinLesionModel(base_model="...")     # 自定义基座模型路径
        model = SkinLesionModel(adapter_dir="...")    # 自定义 adapter 路径
    """

    def __init__(self, base_model=None, adapter_dir=None):
        if base_model is None:
            base_model = os.path.join(PROJECT_ROOT, "models", "qwen", "Qwen2-VL-2B-Instruct")
        if adapter_dir is None:
            adapter_dir = os.path.join(PROJECT_ROOT, "outputs", "sft_lora_v2")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        bf16_ok = self.device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if bf16_ok else (torch.float16 if self.device == "cuda" else torch.float32)

        # Processor — 优先从 adapter 目录加载（含 tokenizer + chat_template）
        try:
            self.processor = AutoProcessor.from_pretrained(adapter_dir)
        except Exception:
            self.processor = AutoProcessor.from_pretrained(base_model)

        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.min_pixels = 65536   # 256×256
            self.processor.image_processor.max_pixels = 262144  # 512×512

        # 基座模型
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=dtype
        )
        if self.device == "cuda":
            self.model = self.model.to("cuda")

        # 加载 LoRA adapter 并合并
        self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model = self.model.merge_and_unload()
        self.model.eval()

    def predict(self, image_path: str, question: str = None) -> str:
        """
        输入图片路径和问题，返回中文诊断文本。

        参数:
            image_path: JPEG/PNG 图片路径（绝对路径或相对于当前工作目录的相对路径）
            question:   问题文本。默认使用标准诊断提示。

        返回:
            str: 中文诊断结论（含"诊断结论：XXX"格式）
        """
        image = Image.open(image_path).convert("RGB")

        if question is None:
            question = "请仔细观察这张皮肤病变图像，给出最可能的诊断及诊断依据。"

        messages = [
            {"role": "system", "content": [{"type": "text",
                "text": "你只能用中文回答。回答末尾请以'诊断结论：疾病名称'格式给出明确的诊断。"}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]},
        ]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )
        input_len = inputs["input_ids"].shape[1]
        output = self.processor.batch_decode(
            generated_ids[:, input_len:], skip_special_tokens=True
        )[0]
        return output.strip()


# ===== 使用示例 =====
if __name__ == "__main__":
    model = SkinLesionModel()
    # 默认诊断
    answer = model.predict("Skin Lesion Dataset/test/Chickenpox/CHP_03_01_9.jpg")
    print(answer)
    # 自定义问题
    answer = model.predict(
        "Skin Lesion Dataset/test/Chickenpox/CHP_03_01_9.jpg",
        question="这张图片中的皮损属于哪个疾病分期？"
    )
    print(answer)
```

### 交互系统同学注意事项

1. **把 `SkinLesionModel` 类的 `.py` 文件放在 `backend/` 根目录下**，`PROJECT_ROOT` 会自动推断正确的路径
2. 如果只需要 adapter（不需要重新训练），只需拷贝 `outputs/` 目录
3. 基座模型需要单独下载（见上方"模型下载"章节），或直接拷贝 `models/` 目录（~4.4GB）
4. **首次加载模型需要 30-60 秒**（加载 shard + 合并 LoRA），之后每张图推理约 3-5 秒
5. 输出的中文文本末尾固定包含 `诊断结论：XXX`，可直接解析疾病名称

## 接口约定

| 项目 | 说明 |
|------|------|
| 输入 | JPEG/PNG 图片路径 + 文本问题（可选） |
| 输出 | 中文医学诊断文本，末尾含"诊断结论：XXX" |
| 图片格式 | RGB，自动 resize 到 256~512px |
| 支持的疾病类别 | Chickenpox / Monkeypox / HFMD / Measles / Cowpox / Healthy |
| 错误处理 | 图片不存在抛 FileNotFoundError；无 GPU 自动回退 CPU |

## 推理

以下命令中的 `--base_model` 使用相对路径 `models/qwen/Qwen2-VL-2B-Instruct`，在项目根目录下运行即可。

```bash
# SFT 医学问答（当前最优模型）
python infer_qwen.py --base_model models/qwen/Qwen2-VL-2B-Instruct --adapter_dir outputs/sft_lora_v2 --mode sft --image "path/to/image.jpg" --question "请仔细观察这张皮肤病变图像，给出最可能的诊断及诊断依据。"

# CPT 图像描述
python infer_qwen.py --base_model models/qwen/Qwen2-VL-2B --adapter_dir outputs/cpt_lora --mode cpt --image "path/to/image.jpg"
```

## 训练

所有路径均使用相对路径，在 `backend/` 目录下运行即可。

```bash
# CPT 持续预训练
python cpt_qwen.py --train_file data/train_pt_balanced.jsonl --valid_file data/valid_pt.jsonl --output_dir outputs/cpt_lora --num_train_epochs 1.0 --learning_rate 2e-4 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --max_length 512 --logging_steps 10 --save_steps 200 --fp16

# SFT 指令微调（当前最优配置：rank=32, 7 LoRA 模块）
python sft_qwen.py --model_name models/qwen/Qwen2-VL-2B-Instruct --train_file data/high_quality_sft_balanced.jsonl --output_dir outputs/sft_lora_v2 --num_train_epochs 5 --learning_rate 2e-4 --per_device_train_batch_size 1 --max_length 768 --logging_steps 10 --save_steps 200 --fp16
```

## 评测

```bash
# 快速评测（600 条平衡子集，~30 分钟）
python scripts/eval_auto_score.py --base_model models/qwen/Qwen2-VL-2B-Instruct --adapter_dir outputs/sft_lora_v2 --eval_file data/eval_balanced_600.jsonl --output data/eval_results_v2.json

# 全量评测（3402 条，~2 小时）
python scripts/eval_auto_score.py --base_model models/qwen/Qwen2-VL-2B-Instruct --adapter_dir outputs/sft_lora_v2 --eval_file data/eval_data.jsonl --output data/eval_results_full.json
```

## 性能基准

| 模型版本 | 总体准确率 | 说明 |
|---------|-----------|------|
| SmolVLM-500M SFT | ~10% | 已废弃，保留在 `../workspace/` |
| Qwen2-VL v1 | 60.0% | LoRA r=16, 4 模块, 无中文强制 |
| **Qwen2-VL v2** | **82.3%** | LoRA r=32, 7 模块, 中文强制 |

### v2 各类别准确率（600 条平衡评测）

| 疾病 | 准确率 |
|------|--------|
| Healthy | 94% |
| Measles | 93% |
| HFMD | 86% |
| Cowpox | 86% |
| Chickenpox | 68% |
| Monkeypox | 67% |

## 已知局限

1. 模型 2B 参数，准确率有限，**不可代替医生诊断**
2. 水痘与猴痘容易混淆（皮损形态高度相似），两者准确率仅 ~67%
3. 图片质量敏感，模糊/过曝/极端角度效果下降
4. 训练数据主要来自公开数据集，不同肤色泛化能力未充分验证
