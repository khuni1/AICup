# finetune_pctr_kfold.py
import os, json, re, argparse
from typing import List, Dict, Tuple, Any
from PIL import Image
from tqdm import tqdm
import numpy as np

# 🔽 메모리 단편화 완화
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

# (옵션) LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False

# 8-bit optimizer 사용 가능 여부
try:
    import bitsandbytes as bnb  # noqa: F401
    _BNB_AVAILABLE = True
except Exception:
    _BNB_AVAILABLE = False

# ----------------------------
# 고정 경로
# ----------------------------
TRAIN_JSON = "../train/train.json"
IMAGES_DIR = "../train/images"

# ----------------------------
# 프롬프트
# ----------------------------
PROMPT_SYSTEM = (
    "你是表格理解与计算助手。给你一张拍摄的中文表格图片和一个与表相关的问题。"
    "请先在心里从表格中定位需要的字段并进行必要的计算，但**不要展示过程**。"
    "严格要求：只输出最终答案一个字符串；不要单位；不要标点；不要解释。"
    "若是判断题，只能输出“是”或“否”。若表格无法回答，输出“无法判断”。"
)

def build_messages_for_train(question: str, answer: str) -> List[Dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": PROMPT_SYSTEM}]},
        {"role": "user",   "content": [{"type": "image"}, {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]

def build_messages_for_infer(question: str) -> List[Dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": PROMPT_SYSTEM}]},
        {"role": "user",   "content": [{"type": "image"}, {"type": "text", "text": question}]},
    ]

# ----------------------------
# 유틸
# ----------------------------
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

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

# ----------------------------
# Collator (텍스트만 수동 패딩)
# ----------------------------
class QwenVLManualCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int | None = 8):
        self.pad_token_id = pad_token_id if pad_token_id is not None else 0
        self.pad_to_multiple_of = pad_to_multiple_of
        self._printed = False

    def _pad_1d(self, x: torch.Tensor, target: int, value: int) -> torch.Tensor:
        if x.shape[0] == target:
            return x
        pad_len = target - x.shape[0]
        return torch.cat([x, torch.full((pad_len,), value, dtype=x.dtype)], dim=0)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        lengths = [f["input_ids"].shape[0] for f in features]
        max_len = max(lengths)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            if max_len % m != 0:
                max_len = ((max_len + m - 1) // m) * m

        input_ids = torch.stack([self._pad_1d(f["input_ids"], max_len, self.pad_token_id) for f in features])
        attention_mask = torch.stack([self._pad_1d(f["attention_mask"], max_len, 0) for f in features])

        labels = None
        if "labels" in features[0]:
            labels = torch.stack([self._pad_1d(f["labels"], max_len, -100) for f in features])

        batch: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if labels is not None:
            batch["labels"] = labels

        # 멀티모달 텐서는 그대로 stack
        for k in features[0].keys():
            if k in ("input_ids", "attention_mask", "labels"):
                continue
            v0 = features[0][k]
            if torch.is_tensor(v0):
                batch[k] = torch.stack([f[k] for f in features])

        # 첫 배치 shape 프린트
        if not self._printed:
            rank0 = (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
            if rank0:
                print({k: (tuple(v.shape) if torch.is_tensor(v) else type(v)) for k, v in batch.items()})
            self._printed = True
        return batch

# ----------------------------
# Dataset (SFT)
# ----------------------------
class PCTRQADataset(Dataset):
    """이미지 전처리는 AutoProcessor에 모두 위임."""
    def __init__(self, records: List[Dict], processor: AutoProcessor, images_dir: str):
        self.records = records
        self.processor = processor
        self.images_dir = images_dir

    def __len__(self):
        return len(self.records)

    def _load_image(self, image_rel: str) -> Image.Image:
        img_path = os.path.join(self.images_dir, os.path.basename(image_rel))
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (512, 512), (255, 255, 255))
        return img

    def __getitem__(self, idx):
        ex = self.records[idx]
        q = ex["question"]
        a = normalize_answer(ex.get("answer", ""))
        img = self._load_image(ex["image"])

        msgs_full = build_messages_for_train(q, a)
        text_full = self.processor.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)

        msgs_prompt = build_messages_for_infer(q)
        text_prompt = self.processor.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)

        proc_full = self.processor(text=[text_full], images=[img], padding=True, return_tensors="pt")
        proc_prompt = self.processor(text=[text_prompt], images=[img], padding=True, return_tensors="pt")

        input_ids = proc_full["input_ids"][0]
        labels = input_ids.clone()
        prompt_len = proc_prompt["input_ids"].shape[1]
        labels[:prompt_len] = -100

        item = {
            "input_ids": input_ids,
            "attention_mask": proc_full["attention_mask"][0],
            "labels": labels,
        }
        # 멀티모달 텐서 포함
        for k, v in proc_full.items():
            if k in ("input_ids", "attention_mask"):
                continue
            if torch.is_tensor(v):
                if v.dim() >= 1 and v.shape[0] == 1:
                    item[k] = v[0]
                else:
                    item[k] = v
            elif isinstance(v, list) and len(v) == 1:
                item[k] = v[0]
        return item

# ----------------------------
# 평가(생성)
# ----------------------------
@torch.no_grad()
def evaluate_fold(model, processor, records: List[Dict], images_dir: str, max_new_tokens: int = 48) -> Tuple[float, List[Dict]]:
    model.eval()
    preds, correct = [], 0
    for ex in tqdm(records, desc="Validate(generation)"):
        qid = ex["id"]; q = ex["question"]
        gt_raw = ex.get("answer", ""); gt = normalize_answer(gt_raw)
        img_path = os.path.join(images_dir, os.path.basename(ex["image"]))
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (512, 512), (255, 255, 255))

        messages = build_messages_for_infer(q)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(model.device)

        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], gen)]
        out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        pred = normalize_answer(out)

        ok = (str(pred) == str(gt)); correct += int(ok)
        preds.append({"id": qid, "question": q, "image": ex["image"], "pred": pred, "gt": gt_raw, "gt_norm": gt, "raw": out, "correct": int(ok)})
    acc = correct / max(1, len(records))
    return acc, preds

# ----------------------------
# 메인
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)       # 멀티모달은 1 권장
    parser.add_argument("--grad_accum", type=int, default=16)      # 유효 배치 = 1 * grad_accum
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--output_dir", default="./qwen25vl_kfold")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--optim", type=str, default="auto",
                        choices=["auto", "adamw", "paged_adamw_8bit", "adafactor"],
                        help="auto: bnb 있으면 8bit, 없으면 adafactor")
    parser.add_argument("--lora", action="store_true", default=True,  help="LoRA로 파라미터 효율 미세튜닝(메모리 대폭 절약)")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 데이터
    all_rows = load_jsonl(TRAIN_JSON)
    print(f"[INFO] loaded train rows: {len(all_rows)}")

    # 모델
    print("[INFO] loading model/processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    # 메모리 절약: 캐시 끄기 + GC
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 이미지 토큰 상한 축소 (메모리↓)
    # 256*28*28 ~= 200k px, 512*28*28 ~= 400k px
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=256*28*28,
        max_pixels=512*28*28,
    )

    # LoRA (옵션)
    if args.lora:
        if not _PEFT_AVAILABLE:
            raise RuntimeError("peft가 필요합니다. pip install peft")
        # 4bit/8bit 양자화 없이도 LoRA만으로 상당히 절약됨.
        # 더 줄이고 싶으면 bitsandbytes의 4bit 로드(QLoRA)도 가능.
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj"
            ],
        )
        # (선택) 비전타워/임베딩을 얼려 추가 절감
        if hasattr(model, "visual"):
            for p in model.visual.parameters():
                p.requires_grad = False
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    print("[INFO] model ready on", device)

    # 폴드 분할
    kf = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    oof_results: List[Dict] = []
    fold_accs = []

    # 옵티마이저 선택
    if args.optim == "auto":
        optim_name = "paged_adamw_8bit" if _BNB_AVAILABLE else "adafactor"
    else:
        optim_name = args.optim

    # 수치 형식
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    for fold, (tr_idx, va_idx) in enumerate(kf.split(all_rows), start=1):
        print(f"\n========== Fold {fold}/{args.k} ==========")
        train_records = [all_rows[i] for i in tr_idx]
        val_records   = [all_rows[i] for i in va_idx]

        train_ds = PCTRQADataset(train_records, processor, IMAGES_DIR)
        val_ds   = PCTRQADataset(val_records,   processor, IMAGES_DIR)

        fold_dir = os.path.join(args.output_dir, f"fold-{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=fold_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            lr_scheduler_type="cosine",
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=1,
            report_to="none",

            # 메모리/성능
            optim=optim_name,
            bf16=use_bf16,
            fp16=(not use_bf16),
            tf32=True,                        # Ampere+ 에서 matmul TF32 허용
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,

            seed=args.seed,
            data_seed=args.seed,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
        )

        collator = QwenVLManualCollator(
            pad_token_id=processor.tokenizer.pad_token_id,
            pad_to_multiple_of=8,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=processor,   # tokenizer 대신
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=1,
                early_stopping_threshold=0.0
            )],
        )

        trainer.train()

        # 저장
        trainer.save_model(fold_dir)
        processor.save_pretrained(fold_dir)

        acc, preds = evaluate_fold(model, processor, val_records, IMAGES_DIR, max_new_tokens=args.max_new_tokens)
        fold_accs.append(acc)
        oof_results.extend([dict(p, fold=fold) for p in preds])
        print(f"[FOLD {fold}] OOF Accuracy = {acc:.4f}")

        torch.cuda.empty_cache()

    oof_acc = np.mean([r["correct"] for r in oof_results])
    print("\n===== OOF Summary =====")
    print(f"Folds: {args.k}")
    print(f"Fold Accs: {', '.join([f'{a:.4f}' for a in fold_accs])}")
    print(f"OOF Accuracy: {oof_acc:.4f}")

    import pandas as pd
    df_oof = pd.DataFrame(oof_results)
    df_oof.to_csv(os.path.join(args.output_dir, "oof_predictions.csv"), index=False)
    with open(os.path.join(args.output_dir, "oof_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"k": args.k, "fold_accs": [float(a) for a in fold_accs], "oof_accuracy": float(oof_acc), "total": int(len(oof_results))},
                  f, ensure_ascii=False, indent=2)
    print(f"[OK] saved OOF to {args.output_dir}")

if __name__ == "__main__":
    # CUDA/TF32 설정(성능+메모리 약간 도움)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    main()
