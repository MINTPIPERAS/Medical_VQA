"""
SFT 指令微调：Qwen2-VL-2B-Instruct + LoRA，多模态医学 VQA。
支持两种数据格式：
  messages 多轮: {"image": "...", "messages": [{...}, ...]}
  question/answer 单轮: {"image": "...", "question": "...", "answer": "..."}
"""
import argparse, json, logging, math, os, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
import datasets  # 必须在 transformers.Trainer 之前，防止 Windows DLL segfault
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

LOGGER = logging.getLogger("qwen2vl_sft_lora")


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen2-VL-2B-Instruct")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_root", type=str,
                        default=".",
                        help="图片数据集根目录")

    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--max_pixels", type=int, default=262144)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    return parser.parse_args()


def resolve_path(image_path: str, dataset_root: str) -> str:
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    return str(Path(dataset_root) / image_path)


class MultiModalSFTDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file: str, dataset_root: str, max_samples: Optional[int] = None):
        self.data = []
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"Data file not found: {jsonl_file}")

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_id, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Line {line_id} invalid JSON: {e}")

                if "image" not in item:
                    raise ValueError(f"Line {line_id}: missing 'image'")

                img_path = resolve_path(item["image"], dataset_root)
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Line {line_id}: image not found: {img_path}")
                item["image"] = img_path

                if "messages" not in item:
                    if "question" not in item or "answer" not in item:
                        raise ValueError(f"Line {line_id}: need 'messages' or 'question'+'answer'")
                    item["messages"] = [
                        {"role": "user", "content": item["question"]},
                        {"role": "assistant", "content": item["answer"]},
                    ]

                msgs = item["messages"]
                if not isinstance(msgs, list) or len(msgs) < 2:
                    raise ValueError(f"Line {line_id}: messages must have >=2 turns")

                self.data.append(item)
                if max_samples and len(self.data) >= max_samples:
                    break

        if not self.data:
            raise ValueError(f"Empty data file: {jsonl_file}")
        LOGGER.info("Loaded %d samples from %s", len(self.data), jsonl_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"]).convert("RGB")
        return {
            "image": image,
            "messages": item["messages"],
            "image_path": item["image"],
        }


@dataclass
class MultiModalCollator:
    processor: Any
    max_length: int

    def __call__(self, features) -> Dict[str, torch.Tensor]:
        all_texts = []
        all_images = []
        all_msgs = []

        for f in features:
            pil_img = f["image"]
            msgs = self._convert_messages(f["messages"], pil_img)
            all_msgs.append(msgs)
            all_images.append([pil_img])

            prompt = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
            )
            all_texts.append(prompt)

        batch = self.processor(
            text=all_texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # 确保 input_ids 为 long 类型
        batch["input_ids"] = batch["input_ids"].long()
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].long()

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        # Mask non-assistant tokens 逐条增量编码
        for i, msgs in enumerate(all_msgs):
            prev_len = 0
            cum_msgs = []
            for msg in msgs:
                cum_msgs.append(msg)
                add_gen = (msg["role"] != "assistant")
                cum_text = self.processor.apply_chat_template(
                    cum_msgs, tokenize=False, add_generation_prompt=add_gen,
                )
                cum_inputs = self.processor.tokenizer(
                    cum_text, return_tensors="pt", add_special_tokens=False,
                )
                curr_len = cum_inputs["input_ids"].shape[1]
                if msg["role"] != "assistant":
                    labels[i, prev_len:min(curr_len, labels.shape[1])] = -100
                prev_len = curr_len

        batch["labels"] = labels
        return batch

    @staticmethod
    def _convert_messages(messages: List[Dict], pil_img: Image.Image) -> List[Dict]:
        """将 messages 转为 Qwen2-VL content 块格式，用 PIL image 填充。"""
        converted = []
        image_placed = False
        chinese_hint = "你只能用中文进行回答。回答末尾请以'诊断结论：'格式给出明确的疾病名称。"
        for msg in messages:
            role = msg["role"]
            content = str(msg.get("content", ""))
            if role == "system":
                content = content + " " + chinese_hint
                converted.append({"role": role, "content": [{"type": "text", "text": content}]})
            elif role == "user":
                has_image = bool(re.search(r'<image[_\d]*>', content))
                text = re.sub(r'<image[_\d]*>', '', content).strip()
                if not text:
                    text = "请分析这张图像。"
                blocks = []
                if has_image and not image_placed:
                    blocks.append({"type": "image"})
                    image_placed = True
                blocks.append({"type": "text", "text": text})
                converted.append({"role": role, "content": blocks})
            else:
                converted.append({"role": role, "content": [{"type": "text", "text": content}]})
        return converted


def find_lora_target_modules(model: torch.nn.Module) -> List[str]:
    candidate_suffixes = {"q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"}
    found: Set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in candidate_suffixes:
                found.add(suffix)
    targets = sorted(found)
    if not targets:
        raise RuntimeError("No LoRA target modules found.")
    LOGGER.info("LoRA targets: %s", targets)
    return targets


def main():
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
    args = parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Cannot use both --bf16 and --fp16")
    use_cuda = torch.cuda.is_available()
    if not args.bf16 and not args.fp16:
        args.fp16 = use_cuda  # RTX 4060 Laptop: fp16 原生支持

    LOGGER.info("=" * 60)
    LOGGER.info("Qwen2-VL-2B LoRA SFT")
    LOGGER.info("Model: %s | Train: %s | Output: %s | CUDA: %s",
                args.model_name, args.train_file, args.output_dir, use_cuda)
    LOGGER.info("bf16: %s | fp16: %s", args.bf16, args.fp16)
    LOGGER.info("=" * 60)

    LOGGER.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = args.max_pixels // 4
        processor.image_processor.max_pixels = args.max_pixels

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    LOGGER.info("Loading model (dtype=%s)...", dtype)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    target_modules = find_lora_target_modules(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    LOGGER.info("Loading dataset...")
    train_dataset = MultiModalSFTDataset(
        args.train_file, args.dataset_root, args.max_train_samples,
    )
    eval_dataset = None
    if args.valid_file:
        eval_dataset = MultiModalSFTDataset(args.valid_file, args.dataset_root)

    collator = MultiModalCollator(processor=processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    LOGGER.info("Starting SFT training...")
    trainer.train()

    LOGGER.info("Saving adapter...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    LOGGER.info("Done. Adapter saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
