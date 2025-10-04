#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse
from typing import List, Dict, Tuple
from collections import OrderedDict

import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

# ---- normalize(학습 스크립트와 동일하게) ----
def normalize_answer(s: str) -> str:
    s = (s or "").strip()
    if s in ("是", "否"):
        return s
    if "无法判断" in s:
        return "无法判断"
    m = list(re.finditer(r"[-+]?\d+(?:\.\d+)?", s))
    if m:
        return m[-1].group(0)
    return s.replace(" ", "").replace("\n", "")

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

def apply_chat_template(processor, messages, *, add_generation_prompt=True):
    fn = getattr(processor, "apply_chat_template", None)
    if callable(fn):
        return fn(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and callable(getattr(tok, "apply_chat_template", None)):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    # Fallback (권장 경로는 아님)
    sys = "".join([c["text"] for c in messages[0]["content"]]) if messages and messages[0]["role"]=="system" else ""
    user_text = "".join([c.get("text","") for c in messages[-1]["content"] if c["type"]=="text"])
    return f"{sys}\n\nUser: {user_text}\nAssistant:"

def load_test_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            rows.append(json.loads(line))
    return rows

def load_fold2_model(fold_dir: str, base_model_id: str, device: str):
    """
    fold-2에 저장된 모델을 로드.
    1) 우선 fold_dir에서 바로 로드 시도
    2) 실패하면 base 모델 + peft 어댑터 로드
    """
    # processor는 fold 디렉토리에서 (chat_template 등 맞춤 파일 포함)
    processor = AutoProcessor.from_pretrained(fold_dir)

    # 1) 직접 로드 시도
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            fold_dir, torch_dtype="auto", low_cpu_mem_usage=True
        )
        model.to(device)
        return model, processor
    except Exception as e1:
        # 2) peft 어댑터 경로로 로드
        try:
            from peft import PeftModel
        except Exception as _:
            raise RuntimeError(f"PEFT가 필요합니다. `pip install peft` 후 다시 실행하세요. (direct load error: {e1})")

        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id, torch_dtype="auto", low_cpu_mem_usage=True
        ).to(device)
        model = PeftModel.from_pretrained(base, fold_dir)  # adapter_model.safetensors 로드
        return model, processor

@torch.no_grad()
def infer_one(model, processor, image_path: str, question: str, max_new_tokens: int = 48) -> Tuple[str, str]:
    model.eval()
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (512, 512), (255, 255, 255))

    text = apply_chat_template(processor, build_messages_for_infer(question), add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False, temperature=0.0,
    )
    gen_ids = out[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return normalize_answer(raw), raw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", default="./qwen25vl_kfold/fold-2", help="fold-2 디렉토리")
    ap.add_argument("--base_model_id", default="Qwen/Qwen2.5-VL-7B-Instruct", help="PEFT fallback용")
    ap.add_argument("--test_json", default="../../test/test.json", help="테스트 jsonl (id,image,question)")
    ap.add_argument("--images_dir", default="../../test/images", help="테스트 이미지 루트")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--out_csv", default="./fold2_test_predictions.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    print(f"[INFO] loading fold-2 from {args.fold_dir}")
    model, processor = load_fold2_model(args.fold_dir, args.base_model_id, device)

    rows = load_test_jsonl(args.test_json)
    print(f"[INFO] test items: {len(rows)}")

    out_rows = []
    for ex in tqdm(rows, desc="Predict(fold-2)"):
        qid = ex["id"]; q = ex["question"]
        img_path = os.path.join(args.images_dir, os.path.basename(ex["image"]))
        pred, raw = infer_one(model, processor, img_path, q, args.max_new_tokens)
        out_rows.append(OrderedDict(id=qid, image=ex["image"], question=q, pred=pred, raw=raw))

    df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] saved predictions -> {args.out_csv}")

if __name__ == "__main__":
    main()
