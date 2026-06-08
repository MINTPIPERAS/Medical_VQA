# Medical VQA 交互系统 — 技术说明

## 项目概述

基于 **Qwen2-VL-2B-Instruct + LoRA** 微调的皮肤病变医学视觉问答（VQA）系统，支持用户上传皮肤病变图像并输入问题，AI 模型实时生成专业诊断分析。前端采用 Vue3 + TailwindCSS，后端采用 FastAPI + PyTorch。

**当前最优模型：SFT v2 — 准确率 82.3%**

---

## 系统架构

```
┌─────────────────────┐       HTTP/SSE        ┌──────────────────────┐
│   Frontend (Vue3)   │ ◄──────────────────► │  Backend (FastAPI)   │
│   Port: 5173        │                       │  Port: 7860          │
│                      │   /api/vqa (stream)   │                      │
│  • ChatSidebar      │   /api/health         │  • Qwen2-VL-2B        │
│  • ChatWindow       │                       │  • SFT LoRA Adapter  │
│  • MessageBubble    │                       │  • TextIteratorStream│
│  • ChatInput        │                       │                      │
│  • IndexedDB        │                       │                      │
└─────────────────────┘                       └──────────────────────┘
```

| 层 | 技术栈 | 说明 |
|---|--------|------|
| **前端** | Vue 3.5 (Composition API) + Vite 8 + TailwindCSS v4 | 纯 JS，无 TypeScript；OKLCH 色彩空间，支持跟随系统暗色模式 |
| **后端** | FastAPI + PyTorch + Transformers + PEFT | SSE 流式输出，启动时预加载模型，全局单例 |
| **模型** | Qwen2-VL-2B-Instruct + SFT v2 LoRA | 6 类皮肤病变：水痘/猴痘/手足口病/麻疹/牛痘/健康 |
| **存储** | IndexedDB（前端） | 对话持久化，无需登录系统 |

---

## 文件结构

```
Medical_VQA/
├── frontend/                         # Vue3 前端项目
│   ├── src/
│   │   ├── App.vue                   # 根组件：管理对话状态 & 布局
│   │   ├── main.js                   # 入口
│   │   ├── style.css                 # TailwindCSS v4 + OKLCH 主题
│   │   ├── api.js                    # 后端 API 客户端（SSE 流式请求）
│   │   ├── db.js                     # IndexedDB 封装（对话 CRUD）
│   │   └── components/
│   │       ├── ChatSidebar.vue       # 侧边栏：对话列表 & 新建/删除
│   │       ├── ChatWindow.vue        # 主窗口：消息列表 & 空状态 & 自动滚动
│   │       ├── MessageBubble.vue     # 消息气泡：Markdown 渲染 & 复制按钮
│   │       └── ChatInput.vue         # 输入区：图片上传 & 文本输入 & 发送
│   ├── index.html
│   ├── vite.config.js                # Vite 配置 + TailwindCSS 插件 + API 代理
│   └── package.json
│
├── backend/                          # 模型端（Qwen2-VL）
│   ├── server.py                     # ★ FastAPI 后端推理服务
│   ├── infer_qwen.py                 # 命令行推理脚本
│   ├── sft_qwen.py                   # SFT 微调训练脚本
│   ├── cpt_qwen.py                   # CPT 预训练脚本
│   ├── models/                       # 基座模型（需下载）
│   ├── outputs/
│   │   ├── sft_lora_v2/              # ★ SFT v2 LoRA 权重（当前最优，82.3%）
│   │   ├── sft_lora/                 # SFT v1 LoRA 权重（已废弃）
│   │   └── cpt_lora/                 # CPT LoRA 权重
│   ├── data/                         # 训练/评测数据集
│   └── scripts/                      # 数据处理 & 评测脚本
│
├── workspace/                        # 旧模型端（SmolVLM-500M，已废弃⚠️）
│   └── DEPRECATED.md
│
└── docs/
    └── Devs.md                       # 本文档
```

---

## 后端 API 接口

### Base URL: `http://127.0.0.1:7860`

### `GET /api/health`
健康检查，返回模型加载状态和设备信息。

### `POST /api/vqa`
医学 VQA 推理（支持流式/非流式）

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | File | ✅ | JPEG/PNG 皮肤病变图像 |
| `question` | String | ❌ | 用户问题（默认：诊断请求） |
| `stream` | Bool | ❌ | 是否流式输出（默认 true） |

**流式响应格式（SSE）：**
```
data: 根据
data: 图像
data: 显示
...
data: [DONE]
```

---

## 启动方法

### 前提条件

- **GPU**：NVIDIA 显卡 ≥ 8GB VRAM（RTX 3060+）
- **OS**：Windows / Linux
- **Conda**：需安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Anaconda

### 第一步：创建 Conda 环境并安装依赖

```bash
# 创建虚拟环境
conda create -n qwen_vqa python=3.10 -y
conda activate qwen_vqa

# 安装 PyTorch + 核心依赖
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.3 peft==0.14.0 accelerate==1.13.0 datasets pillow modelscope

# 安装 FastAPI 后端依赖
pip install fastapi uvicorn python-multipart
```

> ⚠️ **必须使用 transformers 4.46.x**，5.x 不兼容 Qwen2-VL。

### 第二步：下载基座模型并启动后端服务

```bash
# 进入 backend 目录
cd backend

# 下载基座模型（约 4.4GB，仅首次）
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen2-VL-2B-Instruct', cache_dir='./models')"

# 启动 FastAPI 服务（端口 7860）
python server.py
```

启动后，模型会自动加载（约 30-60 秒，含 LoRA 合并），看到 `模型加载完成 ✓` 即就绪。

### 第三步：启动前端

```bash
# 新开一个终端

# 进入 frontend 目录
cd frontend

# 安装依赖（仅首次）
npm install

# 启动开发服务器（端口 5173）
npm run dev
```

### 第四步：访问系统

浏览器打开 **http://localhost:5173**

---

## 使用说明

1. 点击输入框左侧的 📷 按钮上传皮肤病变图像
2. 输入问题（如："这是什么皮肤病？严重程度如何？"）
3. 按 Enter 发送
4. AI 模型会逐字流式输出诊断分析
5. 可点击消息右下角的「复制」按钮复制回答
6. 对话自动保存到 IndexedDB（左侧边栏可见历史记录）
7. 点击右上角 🌙/☀️ 按钮切换暗色/亮色模式

---

## 开发说明

### 前端设计要点

| 需求 | 实现 |
|------|------|
| 图像上传控件 | `ChatInput.vue` — `<input type="file">` + 预览 + 移除 |
| 文本输入框 | `<textarea>` + Enter 发送 |
| 多轮对话展示 | `ChatWindow.vue` — 消息列表 + 自动滚动 |
| 流式输出 | SSE `ReadableStream` + 逐 token 追加 |
| 加载提示 | 发送按钮显示旋转动画 + 流式光标闪烁 |
| 错误提示 | 网络错误/后端异常在前端气泡中以红色显示 |
| 暗色模式 | TailwindCSS `dark:` variant + `document.documentElement.classList.toggle('dark')` |
| OKLCH 色彩 | `@theme` 中定义 OKLCH 颜色变量 |
| GPT 风格 | 气泡式对话、侧边栏历史、简约布局 |
| 对话存储 | IndexedDB（`db.js`），按更新时间排序，支持增删查 |
| 复制按钮 | `navigator.clipboard.writeText()` + 状态反馈 |

### 后端设计要点

| 需求 | 实现 |
|------|------|
| 接收图像+文本 | FastAPI `UploadFile` + `Form` |
| 喂入微调模型 | 加载 SFT LoRA adapter + `merge_and_unload()` |
| 专业回答 | System prompt 设定皮肤科专家角色 + 6 类疾病鉴别知识 |
| 稳定性 | 模型全局单例加载、GPU 推理、异常捕获返回 400/503 |
| 流式输出 | `TextIteratorStreamer` + 独立线程 generate + SSE yield |

---

## 已知局限

1. **准确率有限**：Qwen2-VL-2B 准确率约 82.3%，不可替代医生诊断
2. **水痘/猴痘混淆**：两者皮损形态高度相似，准确率仅 ~67%
3. **硬件要求**：需 NVIDIA GPU ≥ 8GB VRAM，纯 CPU 推理极慢
4. **图片质量敏感**：模糊、过曝、极端角度效果下降
5. **肤色泛化**：训练数据主要来自公开数据集
6. **仅中文输出**：英文也可处理但质量略低
