# finetune_pctr_kfold_noleak.py
import os, json, re, argparse, random
from typing import List, Dict, Tuple, Any
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold, train_test_split
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    set_seed,
)

# (옵션) LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
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
TRAIN_JSON = "../train/normalized_train.json"
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

def build_messages_for_infer(question: str) -> List[Dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": PROMPT_SYSTEM}]},
        {"role": "user",   "content": [{"type": "image"}, {"type": "text", "text": question}]},
    ]

def build_messages_for_train(question: str, answer_or_target: str) -> List[Dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": PROMPT_SYSTEM}]},
        {"role": "user",   "content": [{"type": "image"}, {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer_or_target}]},
    ]

# ----------------------------
# 유틸
# ----------------------------
# 위쪽 유틸 구역에 추가
def chat_template(processor, messages, *, tokenize=False, add_generation_prompt=False):
    # 1) 최신 transformers: processor.apply_chat_template
    fn = getattr(processor, "apply_chat_template", None)
    if callable(fn):
        return fn(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
    # 2) 일부 버전: processor.tokenizer.apply_chat_template
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and callable(getattr(tok, "apply_chat_template", None)):
        return tok.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
    # 3) 최후 수단: 아주 단순한 프롬프트 조립(템플릿 없을 때)
    #    (Qwen 계열은 가급적 1/2 경로가 되게 transformers 최신 버전 사용 권장)
    sys = "".join([c["text"] for c in messages[0]["content"]]) if messages and messages[0]["role"]=="system" else ""
    user_text = "".join([c.get("text","") for c in messages[-1]["content"] if c["type"]=="text"])
    return f"{sys}\n\nUser: {user_text}\nAssistant:"


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

def _save_preds_and_errors(preds: List[Dict], out_dir: str, prefix: str = "val"):
    os.makedirs(out_dir, exist_ok=True)
    df_all = pd.DataFrame(preds)
    df_all.to_csv(os.path.join(out_dir, f"{prefix}_predictions.csv"), index=False)
    df_err = df_all[df_all.get("correct", 0) == 0]
    df_err.to_csv(os.path.join(out_dir, f"{prefix}_errors.csv"), index=False)

# ----------------------------
# Collator (텍스트만 수동 패딩)
# ----------------------------
class QwenVLManualCollator:
    def __init__(self, pad_token_id: int | None, pad_to_multiple_of: int | None = 8):
        self.pad_token_id = pad_token_id if pad_token_id is not None else 0
        self.pad_to_multiple_of = pad_to_multiple_of
        self._printed = False

    def _pad_1d(self, x: torch.Tensor, target: int, value: int) -> torch.Tensor:
        if x.shape[0] == target: return x
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

        # 멀티모달 텐서 그대로 stack
        for k in features[0].keys():
            if k in ("input_ids", "attention_mask", "labels"):
                continue
            v0 = features[0][k]
            if torch.is_tensor(v0):
                batch[k] = torch.stack([f[k] for f in features])

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
    def __init__(self, records: List[Dict], processor: AutoProcessor, images_dir: str, use_solution_supervision: bool):
        self.records = records
        self.processor = processor
        self.images_dir = images_dir
        self.use_solution_supervision = use_solution_supervision

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

        # ====== 학습 타깃: 계산식 감독 옵션 ======
        sol = (ex.get("solution", "") or "").strip().replace("\n", " ").replace("\r", " ")
        if self.use_solution_supervision and sol:
            target_text = f"<calc>{sol}</calc><final>{a}</final>"
        else:
            target_text = a

        # 학습용 full 메시지
        text_full = chat_template(
            self.processor,
            [
                {"role": "system", "content": [{"type": "text", "text": PROMPT_SYSTEM}]},
                {"role": "user",   "content": [{"type": "image"}, {"type": "text", "text": q}]},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        text_prompt = chat_template(
            self.processor,
            build_messages_for_infer(q),
            tokenize=False,
            add_generation_prompt=True,
        )

        proc_full = self.processor(text=[text_full], images=[img], padding=True, return_tensors="pt")
        proc_prompt = self.processor(text=[text_prompt], images=[img], padding=True, return_tensors="pt")

        input_ids = proc_full["input_ids"][0]
        labels = input_ids.clone()
        prompt_len = proc_prompt["input_ids"].shape[1]
        labels[:prompt_len] = -100  # 프롬프트 토큰 마스킹

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
                item[k] = v[0] if (v.dim() >= 1 and v.shape[0] == 1) else v
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
    for ex in tqdm(records, desc="Evaluate(generation)"):
        qid = ex["id"]; q = ex["question"]
        gt_raw = ex.get("answer", ""); gt = normalize_answer(gt_raw)
        img_path = os.path.join(images_dir, os.path.basename(ex["image"]))
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (512, 512), (255, 255, 255))

        # evaluate_fold 안
        text = chat_template(
            processor,
            build_messages_for_infer(q),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(model.device)

        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], gen)]
        out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        pred = normalize_answer(out)

        ok = int(str(pred) == str(gt)); correct += ok
        preds.append({"id": qid, "question": q, "image": ex["image"], "pred": pred, "gt": gt_raw, "gt_norm": gt, "raw": out, "correct": ok})
    acc = correct / max(1, len(records))
    return acc, preds

# ----------------------------
# 체크포인트마다 내부검증 저장 콜백 (cal_inner 전용)
# ----------------------------
class SaveEvalOnCheckpoint(TrainerCallback):
    def __init__(self, processor, cal_records: List[Dict], images_dir: str, max_new_tokens: int, fold_dir: str):
        self.processor = processor
        self.cal_records = cal_records
        self.images_dir = images_dir
        self.max_new_tokens = max_new_tokens
        self.fold_dir = fold_dir

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step is None:
            return
        ckpt_dir = os.path.join(self.fold_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model = kwargs["model"]
        acc, preds = evaluate_fold(model, self.processor, self.cal_records, self.images_dir, self.max_new_tokens)
        _save_preds_and_errors(preds, ckpt_dir, prefix="cal")
        print(f"[CHECKPOINT save] step={state.global_step}  cal_acc={acc:.4f}  -> saved to {ckpt_dir}")

# ----------------------------
# 메인
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--output_dir", default="./qwen25vl_kfold")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--optim", type=str, default="auto",
                        choices=["auto", "adamw", "paged_adamw_8bit", "adafactor"])
    parser.add_argument("--lora", action="store_true", default=True)
    parser.add_argument("--use_solution_supervision", action="store_true", default=True)
    parser.add_argument("--cal_ratio", type=float, default=0.15, help="train 내부에서 cal_inner로 떼는 비율")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 데이터
    all_rows = load_jsonl(TRAIN_JSON)
    print(f"[INFO] loaded train rows: {len(all_rows)}")

    # processor
    print("[INFO] loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=256*28*28,
        max_pixels=512*28*28,
    )

    kf = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    oof_results: List[Dict] = []
    fold_accs = []

    # 옵티마이저 선택
    if args.optim == "auto":
        optim_name = "paged_adamw_8bit" if _BNB_AVAILABLE else "adafactor"
    else:
        optim_name = args.optim

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold, (tr_idx, va_idx) in enumerate(kf.split(all_rows), start=1):
        print(f"\n========== Fold {fold}/{args.k} ==========")

        # 새 모델
        print("[INFO] loading model...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype="auto", low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to(device)

        # LoRA
        if args.lora:
            if not _PEFT_AVAILABLE:
                raise RuntimeError("peft가 필요합니다. pip install peft")
            lora_cfg = LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            )
            if hasattr(model, "visual"):
                for p in model.visual.parameters():
                    p.requires_grad = False
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        # 외부 폴드 분할
        train_records = [all_rows[i] for i in tr_idx]
        val_records   = [all_rows[i] for i in va_idx]  # ← 절대 학습/조기종료에 쓰지 않음

        # 내부 분할: train_inner / cal_inner
        # (생성 과제라 라벨 불균형 이슈가 상대적으로 작아 랜덤 split 사용)
        tr_inner, cal_inner = train_test_split(
            train_records, test_size=args.cal_ratio, random_state=args.seed, shuffle=True
        )

        # Datasets
        train_ds = PCTRQADataset(tr_inner, processor, IMAGES_DIR, use_solution_supervision=args.use_solution_supervision)
        cal_ds   = PCTRQADataset(cal_inner, processor, IMAGES_DIR, use_solution_supervision=False)   # 내부 검증은 정답만
        val_ds   = PCTRQADataset(val_records, processor, IMAGES_DIR, use_solution_supervision=False) # 외부 OOF용(학습 비사용)

        fold_dir = os.path.join(args.output_dir, f"fold-{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=fold_dir,
            per_device_train_batch_size=args.batch_size,
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
            save_total_limit=2,
            report_to="none",
            optim=optim_name,
            bf16=use_bf16,
            fp16=(not use_bf16),
            tf32=True,
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
            pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
            pad_to_multiple_of=8,
        )

        # 체크포인트 저장 순간마다 내부검증(cal_inner) 평가/저장
        ckpt_cb = SaveEvalOnCheckpoint(
            processor=processor,
            cal_records=cal_inner,          # ★ 외부 val 아님
            images_dir=IMAGES_DIR,
            max_new_tokens=args.max_new_tokens,
            fold_dir=fold_dir,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=cal_ds,            # ★ 내부 검증만 사용
            processing_class=processor,
            data_collator=collator,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0),
                ckpt_cb,
            ],
        )

        trainer.train()

        # 저장
        trainer.save_model(fold_dir)
        processor.save_pretrained(fold_dir)

        # 외부 val 폴드로 OOF 생성 평가 (학습과 완전 분리)
        acc, preds = evaluate_fold(model, processor, val_records, IMAGES_DIR, max_new_tokens=args.max_new_tokens)
        fold_accs.append(acc)
        _save_preds_and_errors(preds, fold_dir, prefix="val")   # 폴드별 OOF
        print(f"[FOLD {fold}] OOF Accuracy = {acc:.4f}")

        oof_results.extend([dict(p, fold=fold) for p in preds])

        del trainer, model
        torch.cuda.empty_cache()

    # OOF 집계
    total_correct = int(np.sum([r["correct"] for r in oof_results]))
    total_count = len(oof_results)
    oof_acc = total_correct / max(1, total_count)

    print("\n===== OOF Summary =====")
    print(f"Folds: {args.k}")
    print(f"Fold Accs: {', '.join([f'{a:.4f}' for a in fold_accs])}")
    print(f"OOF Accuracy: {oof_acc:.4f}  ({total_correct}/{total_count})")

    # 결과 저장
    df_oof = pd.DataFrame(oof_results)
    df_oof.to_csv(os.path.join(args.output_dir, "oof_predictions.csv"), index=False)
    with open(os.path.join(args.output_dir, "oof_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "k": args.k,
            "fold_accs": [float(a) for a in fold_accs],
            "oof_accuracy": float(oof_acc),
            "total": int(total_count),
            "correct": int(total_correct),
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved OOF to {args.output_dir}")

if __name__ == "__main__":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    main()
