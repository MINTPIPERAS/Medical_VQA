"""
Qwen2-VL-2B + LoRA 推理脚本。支持 CPT 和 SFT 两种模式。
"""
import argparse, os, torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2-VL LoRA inference")
    parser.add_argument("--base_model", type=str,
                        default="Qwen/Qwen2-VL-2B-Instruct",
                        help="基座模型。CPT: Qwen2-VL-2B, SFT: Qwen2-VL-2B-Instruct")
    parser.add_argument("--adapter_dir", type=str, default="./outputs/sft_lora",
                        help="LoRA adapter 路径")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--question", type=str,
                        default="请仔细观察这张皮肤病变图像，给出最可能的诊断及诊断依据。",
                        help="问题（SFT 模式）")
    parser.add_argument("--mode", type=str, default="sft", choices=["sft", "cpt"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_pixels", type=int, default=262144)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16_ok = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if bf16_ok else (torch.float16 if device == "cuda" else torch.float32)

    print("=" * 60)
    print(f"Base model:  {args.base_model}")
    print(f"Adapter:     {args.adapter_dir}")
    print(f"Image:       {args.image}")
    print(f"Mode:        {args.mode}")
    print(f"Device:      {device} / {dtype}")
    print("=" * 60)

    # 1. Processor
    print("Loading processor...")
    try:
        processor = AutoProcessor.from_pretrained(args.adapter_dir)
    except Exception:
        print("  fallback to base_model...")
        processor = AutoProcessor.from_pretrained(args.base_model)
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = args.max_pixels // 4
        processor.image_processor.max_pixels = args.max_pixels

    # 2. Base model
    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
    )
    if device == "cuda":
        model = model.to("cuda")

    # 3. LoRA adapter
    adapter_dir_abs = os.path.abspath(args.adapter_dir)
    print(f"Attaching LoRA: {adapter_dir_abs}")
    if not os.path.exists(adapter_dir_abs):
        raise FileNotFoundError(f"Adapter not found: {adapter_dir_abs}")
    model = PeftModel.from_pretrained(model, adapter_dir_abs)
    model = model.merge_and_unload()
    model.eval()

    # 4. Image
    image = Image.open(args.image).convert("RGB")
    print(f"Image size: {image.size}")

    # 5. Messages
    if args.mode == "cpt":
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the image."},
            ]},
        ]
    else:
        system_prompt = (
            "你是一位资深的皮肤科医学专家，拥有二十年的临床诊断经验。"
            "请根据用户提供的皮肤病变图像，给出专业、准确、详细的分析。"
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": args.question},
            ]},
        ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 6. Generate
    print("Generating...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=(args.temperature > 0),
            temperature=args.temperature if args.temperature > 0 else None,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_len:]
    output = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    print("\n" + "=" * 60)
    if args.mode == "sft":
        print("Question:")
        print(args.question)
    print("\nModel answer:")
    print(output.strip())
    print("=" * 60)


if __name__ == "__main__":
    main()
