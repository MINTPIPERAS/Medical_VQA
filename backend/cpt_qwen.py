"""
CPT 持续预训练：Qwen2-VL-2B + LoRA，输入图像→输出临床描述。
"""
import argparse, json, logging, math, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
import datasets  # 必须在 transformers.Trainer 之前导入，防止 Windows DLL segfault
from PIL import Image, ImageFile
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
LOGGER = logging.getLogger("qwen2vl_cpt_lora")


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA CPT for Qwen2-VL-2B")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_root", type=str,
                        default=".",
                        help="图片数据集根目录（相对路径会拼接此前缀）")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--max_pixels", type=int, default=262144,
                        help="图像最大像素数，控制显存占用 (256*256=65536, 512*512=262144)")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    return parser.parse_args()


def resolve_path(image_path: str, dataset_root: str) -> str:
    """将 JSONL 中的相对路径转为绝对路径。"""
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    abs_path = Path(dataset_root) / image_path
    return str(abs_path)


def load_jsonl(path: str, dataset_root: str) -> List[Dict[str, str]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} line {lineno}: invalid JSON: {e}")

            if "image" not in obj or "text" not in obj:
                raise ValueError(f"{path} line {lineno}: need 'image' and 'text'")

            image_path = str(obj["image"]).strip()
            text = str(obj["text"]).strip()
            if not image_path or not text:
                raise ValueError(f"{path} line {lineno}: empty image or text")

            records.append({"image": resolve_path(image_path, dataset_root), "text": text})

    if not records:
        raise ValueError(f"No valid samples in {path}")
    return records


def verify_and_filter_records(records, limit=None):
    filtered = []
    bad = 0
    for ex in records:
        path = Path(ex["image"])
        if not path.exists():
            bad += 1
            LOGGER.warning("Skip missing: %s", path)
            continue
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            bad += 1
            LOGGER.warning("Skip unreadable %s: %s", path, e)
            continue
        filtered.append(ex)
        if limit and len(filtered) >= limit:
            break
    if not filtered:
        raise ValueError("All samples filtered out.")
    LOGGER.info("Kept %d, filtered %d bad.", len(filtered), bad)
    return filtered


class JsonlImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


@dataclass
class Qwen2VLCollator:
    processor: Any
    max_length: int

    def __call__(self, features: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        all_texts = []
        all_images = []
        prompt_prefix_lens = []  # prompt（图像+指令）的 token 数，用于 label mask

        for ex in features:
            pil_img = Image.open(ex["image"]).convert("RGB")
            target_text = ex["text"]

            # 构造完整输入：image + 指令 + 目标文本
            full_content = [
                {"type": "image"},
                {"type": "text", "text": f"Describe the image.\n{target_text}"},
            ]
            full_text = self.processor.apply_chat_template(
                full_content, tokenize=False, add_generation_prompt=False,
            )
            all_texts.append(full_text)
            all_images.append([pil_img])

            # 只含 prompt 的版本，用于计算需要 mask 的 token 数
            prompt_content = [
                {"type": "image"},
                {"type": "text", "text": "Describe the image.\n"},
            ]
            prompt_text = self.processor.apply_chat_template(
                prompt_content, tokenize=False, add_generation_prompt=False,
            )
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=[[pil_img]],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            prompt_prefix_lens.append(prompt_inputs["input_ids"].shape[1])

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

        # 构造 labels：mask 掉 prompt 部分和 padding
        labels = batch["input_ids"].clone()
        for i, prefix_len in enumerate(prompt_prefix_lens):
            labels[i, :prefix_len] = -100
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        batch["labels"] = labels
        return batch


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


def compute_metrics(eval_pred):
    return {}


def main():
    setup_logging()
    args = parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16")
    set_seed(args.seed)

    LOGGER.info("Loading processor from %s", args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name)

    # 限制图像分辨率以控制显存
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = args.max_pixels // 4
        processor.image_processor.max_pixels = args.max_pixels
        LOGGER.info("Image processor: min_pixels=%d, max_pixels=%d",
                    processor.image_processor.min_pixels,
                    processor.image_processor.max_pixels)

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    LOGGER.info("Loading model from %s (dtype=%s)", args.model_name, dtype)
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
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_records = load_jsonl(args.train_file, args.dataset_root)
    train_records = verify_and_filter_records(train_records, args.max_train_samples)
    train_dataset = JsonlImageTextDataset(train_records)

    eval_dataset = None
    if args.valid_file:
        valid_records = load_jsonl(args.valid_file, args.dataset_root)
        valid_records = verify_and_filter_records(valid_records, args.max_eval_samples)
        eval_dataset = JsonlImageTextDataset(valid_records)

    collator = Qwen2VLCollator(processor=processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        label_names=["labels"],
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )

    LOGGER.info("Start LoRA CPT training")
    train_result = trainer.train()

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if eval_dataset is not None:
        LOGGER.info("Running evaluation")
        metrics = trainer.evaluate()
        if "eval_loss" in metrics:
            try:
                metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
            except OverflowError:
                metrics["eval_perplexity"] = float("inf")
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    LOGGER.info("CPT finished. Adapter saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
