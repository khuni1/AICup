# eval_qwen3vl_noft_235b.py
import os, json, re, argparse
from typing import List, Dict, Any
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration, set_seed  # ★ Qwen3 전용 클래스

PROMPT_SYSTEM = (
    "你是表格理解与计算助手。给你一张拍摄的中文表格图片和一个与表相关的问题。"
    "请先在心里从表格中定位需要的字段并进行必要的计算，但**不要展示过程**。"
    "严格要求：只输出最终答案一个字符串；不要单位；不要标点；不要解释。"
    "若是判断题，只能输出“是”或“否”。若表格无法回答，输出“无法判断”。"
)

def build_messages_for_infer(question: str) -> List[Dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": PROMPT_SYSTEM}]},
        {"role": "user",   "content": [{"type": "image"}, {"type": "text", "text": question}]},
    ]

def chat_template(processor, messages, *, tokenize=False, add_generation_prompt=False):
    fn = getattr(processor, "apply_chat_template", None)
    if callable(fn):
        return fn(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and callable(getattr(tok, "apply_chat_template", None)):
        return tok.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
    sys = ""
    if messages and messages[0]["role"]=="system":
        sys = "".join([c["text"] for c in messages[0]["content"] if c["type"]=="text"])
    user_text = "".join([c.get("text","") for c in messages[-1]["content"] if c["type"]=="text"])
    return f"{sys}\n\nUser: {user_text}\nAssistant:"

def normalize_answer(s: str) -> str:
    s = (s or "").strip()
    if s in ("是", "否"): return s
    if "无法判断" in s: return "无法判断"
    m = list(re.finditer(r"[-+]?\d+(?:\.\d+)?", s))
    if m: return m[-1].group(0)
    return s.replace(" ", "").replace("\n", "")

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

@torch.no_grad()
def evaluate(model, processor, records: List[Dict], images_dir: str, max_new_tokens: int = 32) -> Dict[str, Any]:
    model.eval()
    preds, correct = [], 0
    for ex in tqdm(records, desc="Evaluate(Qwen3-VL 235B, generation)"):
        qid = ex.get("id"); q = ex["question"]
        gt_raw = ex.get("answer", ""); gt = normalize_answer(gt_raw)
        img_path = os.path.join(images_dir, os.path.basename(ex["image"]))
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (512, 512), (255, 255, 255))

        text = chat_template(processor, build_messages_for_infer(q), tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(model.device)

        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], generated)]
        out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        pred = normalize_answer(out)

        ok = int(str(pred) == str(gt)); correct += ok
        preds.append({"id": qid, "question": q, "image": ex["image"], "pred": pred, "gt": gt_raw, "gt_norm": gt, "raw": out, "correct": ok})
    acc = correct / max(1, len(records))
    return {"acc": acc, "preds": preds}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_json", default="../../train/normalized_train.json")
    ap.add_argument("--images_dir", default="../../train/images")
    ap.add_argument("--output_dir", default="./qwen3vl_235b_eval")
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-235B-A22B-Instruct")  
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--min_pixels", type=int, default=256*28*28)
    ap.add_argument("--max_pixels", type=int, default=512*28*28)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_jsonl(args.data_json)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
    print(f"[INFO] Eval samples: {len(rows)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Loading processor/model...")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    # ★ 공식 카드 권장: Qwen3VLMoeForConditionalGeneration + device_map="auto"
    #   (flash_attention_2를 쓰면 메모리/속도 유리)
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype="auto",
        device_map="auto",
        # attn_implementation="flash_attention_2",  # 설치되어 있으면 주석 해제 권장
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model.to(device)
    model.config.use_cache = True

    res = evaluate(model, processor, rows, args.images_dir, max_new_tokens=args.max_new_tokens)
    acc = res["acc"]; preds = res["preds"]
    print(f"[EVAL] Accuracy = {acc:.4f}  ({sum(p['correct'] for p in preds)}/{len(preds)})")

    df = pd.DataFrame(preds)
    df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"model_id": args.model_id, "samples": len(preds), "accuracy": float(acc)}, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved to {args.output_dir}")

if __name__ == "__main__":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    main()
