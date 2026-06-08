# 皮肤病变多模态 VQA 系统 — Qwen2-VL 模型端

基于 **Qwen2-VL-2B-Instruct** + **LoRA** 的皮肤病变图像识别与医学视觉问答系统。

**当前最优模型：SFT v2 — 准确率 82.3%（600 条平衡评测）**

> 📘 详细的 API 文档、性能基准和集成说明请参阅 [README-MODEL.md](README-MODEL.md)。

## 环境配置

```bash
conda create -n qwen_vqa python=3.10 -y
conda activate qwen_vqa
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.3 peft==0.14.0 accelerate==1.13.0 datasets pillow modelscope
```

**关键版本**：transformers 必须是 4.46.x（5.x 不兼容），torch 需要 CUDA 版本。无 GPU 会自动回退 CPU。

## 快速启动

### 1. 安装依赖

```bash
conda activate qwen_vqa
pip install fastapi uvicorn python-multipart
# 或一键安装
pip install -r requirements.txt
```

### 2. 下载基座模型（约 4.4GB）

```bash
# ModelScope（推荐，国内快）
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen2-VL-2B-Instruct', cache_dir='./models')"
```

### 3. 启动推理服务

```bash
python server.py
```

服务启动后将监听 `http://0.0.0.0:7860`，API 接口：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查，返回模型状态 |
| `/api/vqa` | POST | 医学 VQA 推理（支持 SSE 流式输出） |

### 4. 命令行推理

```bash
# SFT 医学问答（当前最优模型）
python infer_qwen.py --base_model models/qwen/Qwen2-VL-2B-Instruct --adapter_dir outputs/sft_lora_v2 --mode sft --image "path/to/image.jpg"

# CPT 图像描述
python infer_qwen.py --base_model models/qwen/Qwen2-VL-2B --adapter_dir outputs/cpt_lora --mode cpt --image "path/to/image.jpg"
```

## 模型

| 阶段 | 基座模型 | 训练数据 | Adapter |
|------|---------|---------|---------|
| CPT | `Qwen/Qwen2-VL-2B` | 3960 张皮肤病变图像 + 临床描述 | `outputs/cpt_lora/` |
| SFT v1 | `Qwen/Qwen2-VL-2B-Instruct` | 3300 条医学 VQA 问答对 | `outputs/sft_lora/`（已废弃） |
| **SFT v2** | `Qwen/Qwen2-VL-2B-Instruct` | 3300 条高质量均衡数据 | `outputs/sft_lora_v2/` ⭐ |

## 性能基准

| 模型版本 | 总体准确率 | 说明 |
|---------|-----------|------|
| SmolVLM-500M SFT | ~10% | 已废弃，保留在 `../workspace/` |
| Qwen2-VL v1 | 60.0% | LoRA r=16, 4 模块 |
| **Qwen2-VL v2** | **82.3%** | LoRA r=32, 7 模块, 中文强制 |

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
├── server.py                           ← FastAPI 推理服务
├── cpt_qwen.py / sft_qwen.py           ← 训练脚本
└── infer_qwen.py                       ← 推理脚本
```

## 接口约定

| 项目 | 说明 |
|------|------|
| 输入 | JPEG/PNG 图片路径 + 文本问题（可选） |
| 输出 | 中文医学诊断文本，末尾含"诊断结论：XXX" |
| 图片格式 | RGB，自动 resize 到 256~512px |
| 支持的疾病类别 | Chickenpox / Monkeypox / HFMD / Measles / Cowpox / Healthy |
| 错误处理 | 图片不存在抛 FileNotFoundError；无 GPU 自动回退 CPU |

## 已知局限

1. 模型 2B 参数，准确率有限，**不可代替医生诊断**
2. 水痘与猴痘容易混淆（皮损形态高度相似），两者准确率仅 ~67%
3. 图片质量敏感，模糊/过曝/极端角度效果下降
4. 训练数据主要来自公开数据集，不同肤色泛化能力未充分验证
