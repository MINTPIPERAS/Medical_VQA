"""
自动评测脚本：用 Qwen2-VL + LoRA 模型对 Eval 数据集做推理并自动评分。
"""
import json, os, time, argparse
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


def resolve_path(image_path: str, dataset_root: str) -> str:
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    return str(Path(dataset_root) / image_path)


DISEASE_KEYWORDS = {
    "Chickenpox": ["水痘", "chickenpox", "varicella"],
    "Monkeypox": ["猴痘", "monkeypox", "mpox"],
    "HFMD": ["手足口", "hfmd", "hand foot"],
    "Measles": ["麻疹", "measles", "rubeola"],
    "Cowpox": ["牛痘", "cowpox"],
    "Healthy": ["健康", "正常", "healthy", "normal", "未见异常", "无明显异常", "无异常"],
}


def identify_disease(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for disease, keywords in DISEASE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[disease] = score
    if not scores:
        return "Unknown"
    return max(scores, key=scores.get)


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


def print_progress_bar(current, total, accuracy, elapsed, errors, prefix="", suffix="", length=40):
    percent = current / total
    filled = int(length * percent)
    bar = "#" * filled + "-" * (length - filled)
    eta = format_time((elapsed / percent - elapsed)) if percent > 0 else "---"
    elapsed_str = format_time(elapsed)
    print(f"\r  {prefix}|{bar}| {percent:.1%} [{current}/{total}] "
          f"准确率:{accuracy:.1%} 错误:{errors} 耗时:{elapsed_str} ETA:{eta}  {suffix}",
          end="", flush=True)


def run_eval(eval_path: str, base_model: str, adapter_dir: str, output_path: str,
             dataset_root: str = ".",
             max_samples: int = None, max_pixels: int = 262144, debug: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16_ok = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if bf16_ok else (torch.float16 if device == "cuda" else torch.float32)

    print(f"设备: {device}, dtype: {dtype}")

    print("加载 processor...")
    try:
        processor = AutoProcessor.from_pretrained(adapter_dir)
    except Exception:
        processor = AutoProcessor.from_pretrained(base_model)
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = max_pixels // 4
        processor.image_processor.max_pixels = max_pixels

    print("加载模型...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(base_model, torch_dtype=dtype)
    if device == "cuda":
        model = model.to("cuda")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.eval()

    records = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if max_samples:
        records = records[:max_samples]

    print(f"评测 {len(records)} 条数据...\n")

    results = []
    correct_by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    correct_by_label = defaultdict(lambda: {"correct": 0, "total": 0})
    error_log = []
    misclassified = []

    start_time = time.time()
    error_count = 0

    for idx, rec in enumerate(records):
        img_path = resolve_path(rec["image_path"], dataset_root)
        question = rec["question"]
        reference = rec["reference_answer"]
        q_type = rec["question_type"]
        gt_label = rec["ground_truth_label"]

        if not os.path.exists(img_path):
            error_count += 1
            error_log.append({"type": "missing_image", "index": idx, "path": img_path})
            if debug:
                print(f"\n  [跳过] 图片不存在: {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            error_count += 1
            error_log.append({"type": "bad_image", "index": idx, "path": img_path, "error": str(e)})
            if debug:
                print(f"\n  [跳过] 图片损坏: {img_path} ({e})")
            continue

        # 英文问题加中文指令，防止模型输出英文导致关键词匹配失败
        if question and question[0].isascii() and question[0].isalpha():
            question = question + "\n请用中文回答，并明确给出诊断结论。"
        messages = [
            {"role": "system", "content": [{"type": "text",
                "text": "你只能用中文回答。回答末尾请以'诊断结论：疾病名称'格式给出明确的诊断。"}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]},
        ]

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        output = processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]

        pred_disease = identify_disease(output)
        disease_correct = (pred_disease == gt_label)

        correct_by_type[q_type]["total"] += 1
        correct_by_label[gt_label]["total"] += 1
        if disease_correct:
            correct_by_type[q_type]["correct"] += 1
            correct_by_label[gt_label]["correct"] += 1

        result = {
            "image_path": img_path,
            "question": question[:100],
            "ground_truth": gt_label,
            "predicted": pred_disease,
            "correct": disease_correct,
            "question_type": q_type,
            "model_output": output[:300],
            "reference": reference[:200],
        }
        results.append(result)

        if not disease_correct:
            misclassified.append(result)
            if debug:
                print(f"\n  [错误] {img_path}")
                print(f"         真实: {gt_label} | 预测: {pred_disease}")
                print(f"         输出: {output[:120]}...")

        elapsed = time.time() - start_time
        if (idx + 1) % 10 == 0 or idx == len(records) - 1:
            acc = sum(1 for r in results if r["correct"]) / len(results) if results else 0
            print_progress_bar(idx + 1, len(records), acc, elapsed, error_count)

    print()

    total_correct = sum(1 for r in results if r["correct"])
    total_accuracy = total_correct / len(results) if results else 0
    total_elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"评测完成 ({format_time(total_elapsed)})")
    print(f"{'='*60}")
    print(f"总样本: {len(results)} | 异常跳过: {error_count}")
    print(f"疾病识别准确率: {total_accuracy:.2%} ({total_correct}/{len(results)})")

    print(f"\n--- 按题型 ---")
    for q_type in ["classification", "description", "clinical"]:
        stats = correct_by_type[q_type]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        bar = "#" * int(acc * 20) + "-" * (20 - int(acc * 20))
        print(f"  {q_type:15s} {bar} {acc:.1%} ({stats['correct']}/{stats['total']})")

    print(f"\n--- 按疾病类别 ---")
    for label in ["Chickenpox", "Monkeypox", "HFMD", "Measles", "Cowpox", "Healthy"]:
        stats = correct_by_label[label]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        bar = "#" * int(acc * 20) + "-" * (20 - int(acc * 20))
        print(f"  {label:15s} {bar} {acc:.1%} ({stats['correct']}/{stats['total']})")

    print(f"\n--- 混淆矩阵 ---")
    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        if not r["correct"]:
            confusion[r["ground_truth"]][r["predicted"]] += 1
    labels_order = ["Chickenpox", "Monkeypox", "HFMD", "Measles", "Cowpox", "Healthy"]
    header = f"{'':15s}" + "".join(f"{l:>10s}" for l in labels_order)
    print(header)
    for gt in labels_order:
        row = f"{gt:15s}"
        for pred in labels_order:
            count = confusion[gt][pred]
            row += f"{count:>10d}" if count > 0 else f"{'.':>10s}"
        print(row)

    print(f"\n--- 错误样本 TOP 5（按类别） ---")
    shown = set()
    for r in results:
        if not r["correct"] and r["ground_truth"] not in shown:
            shown.add(r["ground_truth"])
            print(f"  真实:{r['ground_truth']} → 预测:{r['predicted']}")
            print(f"    图片: {r['image_path']}")
            print(f"    输出: {r['model_output'][:150]}...")
        if len(shown) >= 5:
            break

    summary = {
        "total_accuracy": total_accuracy,
        "total_samples": len(results),
        "total_correct": total_correct,
        "error_count": error_count,
        "elapsed_seconds": total_elapsed,
        "by_type": {k: {"accuracy": v["correct"]/v["total"] if v["total"]>0 else 0, **v}
                    for k, v in correct_by_type.items()},
        "by_label": {k: {"accuracy": v["correct"]/v["total"] if v["total"]>0 else 0, **v}
                     for k, v in correct_by_label.items()},
    }

    output_data = {
        "summary": summary,
        "errors": error_log,
        "misclassified_count": len(misclassified),
        "details": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果保存至: {output_path}")
    print(f"  正确: {total_correct} | 错误分类: {len(misclassified)} | 异常: {error_count}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval 自动评测 (Qwen2-VL)")
    parser.add_argument("--eval_file", default="data/eval_data.jsonl")
    parser.add_argument("--base_model", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--adapter_dir", default="./outputs/sft_lora")
    parser.add_argument("--output", default="data/eval_results.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=262144)
    parser.add_argument("--dataset_root", type=str,
                        default=".")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run_eval(args.eval_file, args.base_model, args.adapter_dir, args.output,
             args.dataset_root, args.max_samples, args.max_pixels, args.debug)
