#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, random
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, set_seed
)
from transformers import EarlyStoppingCallback

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as SkLogReg

from joblib import dump as joblib_dump

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# utility
# =========================
def _safe_logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

def hf_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "recall_weighted": recall_score(labels, preds, average="weighted"),
    }

MIN_LEN_FOR_SPARSE = 768
PAD_MULTIPLE = 64

# =========================
# 0) 재현성 & 경로
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

CAL_METHOD = "platt"     # "platt" or "isotonic"
CAL_INNER_SIZE = 0.2     # 폴드 내부 보정용 분할 비율
THETA_FIXED = 0.5        # 보고/배포 모두 고정 0.5

BASE_DIR   = "../../dataset/train"
MODEL_NAME = "google/bigbird-roberta-base"
OUT_DIR    = "./kfold_results_bigbird_doc"
OUTPUT_MODEL_DIR = "./final_model_bigbird_doc"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
FOLD_MODELS_DIR = os.path.join(OUTPUT_MODEL_DIR, "fold_models")
CALIB_DIR_ROOT  = os.path.join(OUTPUT_MODEL_DIR, "calibrators")
os.makedirs(FOLD_MODELS_DIR, exist_ok=True)
os.makedirs(CALIB_DIR_ROOT, exist_ok=True)

# =========================
# 1) .tex 파서 (문서 단위)
# =========================
def parse_tex_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    blocks = {}
    for block in ["Input","Output","Formula","Explanation"]:
        m = re.search(rf"%{block}(.*?)(?=%|$)", text, re.S)
        blocks[block.lower()] = m.group(1).strip() if m else ""
    return blocks

# =========================
# 2) 전체 데이터 로드 (문서=1행)
# =========================
rows = []
for fname in os.listdir(BASE_DIR):
    if fname.endswith(".tex"):
        p = os.path.join(BASE_DIR, fname)
        b = parse_tex_file(p)
        full_text = " ".join(b.values())
        label = 0 if "_0" in fname else 1     # Human:0 / Machine:1
        rows.append({"fname": fname, "text": full_text, "label": label})

df = pd.DataFrame(rows).reset_index(drop=True)
df["doc_uid"] = np.arange(len(df), dtype=int)

print("=== 전체 라벨 분포 ===")
print(df["label"].value_counts(normalize=True).sort_index().rename({0:"Human",1:"Machine"}))

# =========================
# 3) 토크나이저 (BigBird, 4096)
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

@dataclass
class MinLenPadCollator:
    tokenizer: Any
    min_len: int = MIN_LEN_FOR_SPARSE
    multiple: int = PAD_MULTIPLE
    def __call__(self, features: List[Dict[str, Any]]):
        batch = self.tokenizer.pad(features, padding="longest", return_tensors="pt")
        seq_len = batch["input_ids"].shape[1]
        target_len = max(seq_len, self.min_len)
        if self.multiple:
            target_len = ((target_len + self.multiple - 1) // self.multiple) * self.multiple
        if target_len > seq_len:
            pad_len = target_len - seq_len
            pad_id = self.tokenizer.pad_token_id
            pad_ids  = torch.full((batch["input_ids"].size(0), pad_len), pad_id, dtype=batch["input_ids"].dtype)
            pad_mask = torch.zeros((batch["attention_mask"].size(0), pad_len), dtype=batch["attention_mask"].dtype)
            batch["input_ids"]      = torch.cat([batch["input_ids"], pad_ids], dim=1)
            batch["attention_mask"] = torch.cat([batch["attention_mask"], pad_mask], dim=1)
        return batch

MAX_LEN = 4096

def tokenize_fn(examples):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
        return_offsets_mapping=False
    )
    enc["labels"] = examples["label"]
    return enc

data_collator = MinLenPadCollator(tokenizer=tokenizer, min_len=768, multiple=64)

# =========================
# 4) K-Fold + 내부 분할 보정 + raw OOF 수집
# =========================
K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

oof_probs_raw = []      # raw OOF p(machine)
oof_labels     = []
oof_doc_ids    = []
oof_fold_ids   = []     # 각 OOF 항목이 나온 fold id (CV-calibration에 필요)

supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

for fold, (train_idx, val_idx) in enumerate(skf.split(df["text"], df["label"]), start=1):
    print(f"\n========== Fold {fold}/{K} ==========")
    fold_dir = os.path.join(OUT_DIR, f"fold-{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    # ----- 폴드 내부 보정용 분할 -----
    sss = StratifiedShuffleSplit(n_splits=1, test_size=CAL_INNER_SIZE, random_state=SEED)
    (idx_tr_in, idx_cal_in), = sss.split(train_df["text"], train_df["label"])
    train_in_df = train_df.iloc[idx_tr_in].reset_index(drop=True)
    cal_in_df   = train_df.iloc[idx_cal_in].reset_index(drop=True)

    # Datasets
    def _mk_ds(pdf):
        ds = Dataset.from_pandas(pdf).map(tokenize_fn, batched=True, remove_columns=list(pdf.columns))
        ds.set_format("torch")
        return ds

    train_in_ds = _mk_ds(train_in_df)
    cal_in_ds   = _mk_ds(cal_in_df)
    val_ds      = _mk_ds(val_df)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, use_safetensors=True
    ).to(device)
    model.config.num_random_blocks = 2
    model.config.attention_type = "block_sparse"

    training_args = TrainingArguments(
        output_dir=fold_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir=os.path.join(fold_dir, "logs"),
        logging_steps=50,
        eval_strategy="epoch",   
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        group_by_length=False,
        bf16=supports_bf16,
        fp16=not supports_bf16,
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="adamw_torch",
        seed=SEED, data_seed=SEED,
        label_smoothing_factor=0.1
    )

    # ★ 누수 방지: 훈련 중 평가는 cal_in_ds만 사용
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_in_ds,
        eval_dataset=cal_in_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=1e-4)],
    )
    trainer.train()

    # ===== 폴드 모델 저장 =====
    fold_export_dir = os.path.join(FOLD_MODELS_DIR, f"fold-{fold}")
    os.makedirs(fold_export_dir, exist_ok=True)
    trainer.save_model(fold_export_dir)
    tokenizer.save_pretrained(fold_export_dir)
    with open(os.path.join(fold_export_dir, "training_args.json"), "w", encoding="utf-8") as fta:
        json.dump(training_args.to_dict(), fta, ensure_ascii=False, indent=2)

    # ----- cal_inner 예측으로 폴드 보정기 학습 -----
    preds_cal_in = trainer.predict(cal_in_ds)
    p1_cal_in = torch.softmax(torch.tensor(preds_cal_in.predictions), dim=1).numpy()[:, 1]
    y_cal_in  = preds_cal_in.label_ids

    calib_dir = os.path.join(CALIB_DIR_ROOT, f"fold-{fold}")
    os.makedirs(calib_dir, exist_ok=True)

    if CAL_METHOD == "platt":
        X_platt = _safe_logit(p1_cal_in).reshape(-1, 1)
        platt = SkLogReg(solver="lbfgs", C=1e6, max_iter=1000)
        platt.fit(X_platt, y_cal_in)
        joblib_dump(platt, os.path.join(calib_dir, "platt.joblib"))
    else:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p1_cal_in, y_cal_in)
        joblib_dump(iso, os.path.join(calib_dir, "isotonic.joblib"))

    # ----- 외부 val 예측(raw) → OOF(raw) 수집 -----
    preds_val = trainer.predict(val_ds)
    p1_val_raw = torch.softmax(torch.tensor(preds_val.predictions), dim=1).numpy()[:, 1]
    y_val      = preds_val.label_ids
    doc_ids    = val_df["doc_uid"].values

    oof_probs_raw.append(p1_val_raw)
    oof_labels.append(y_val)
    oof_doc_ids.append(doc_ids)
    oof_fold_ids.append(np.full_like(y_val, fill_value=fold))

    del trainer, model, preds_val, preds_cal_in
    torch.cuda.empty_cache()

# =========================
# 5) OOF(raw) 집계 (θ 학습 없음)
# =========================
p_oof_raw   = np.concatenate(oof_probs_raw)
y_oof       = np.concatenate(oof_labels)
doc_ids_oof = np.concatenate(oof_doc_ids)
fold_ids_oof= np.concatenate(oof_fold_ids)

# =========================
# 6) CV-Calibration (정석; 각 fold는 나머지 OOF로 학습한 보정기 적용)
# =========================
p_oof_cvcal = np.zeros_like(p_oof_raw, dtype=float)

for fold in range(1, K+1):
    mask_val_fold = (fold_ids_oof == fold)
    mask_train_cal = ~mask_val_fold

    p_tr, y_tr = p_oof_raw[mask_train_cal], y_oof[mask_train_cal]
    p_te       = p_oof_raw[mask_val_fold]

    if CAL_METHOD == "platt":
        X_tr = _safe_logit(p_tr).reshape(-1, 1)
        platt = SkLogReg(solver="lbfgs", C=1e6, max_iter=1000)
        platt.fit(X_tr, y_tr)
        p_oof_cvcal[mask_val_fold] = platt.predict_proba(_safe_logit(p_te).reshape(-1,1))[:, 1]
    else:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_tr, y_tr)
        p_oof_cvcal[mask_val_fold] = iso.predict(p_te)

# =========================
# 7) CV-calibrated OOF @ θ=0.5 (보수적 리포트)
# =========================
theta_fixed = THETA_FIXED
yhat_oof_cvcal = (p_oof_cvcal >= theta_fixed).astype(int)

print("\n===== OOF Results (CV-calibrated, theta=0.5) =====")
print(classification_report(y_oof, yhat_oof_cvcal, target_names=["Human","Machine"]))

cm_cv = confusion_matrix(y_oof, yhat_oof_cvcal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_cv, display_labels=["Human","Machine"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"OOF Confusion Matrix (CV-calibrated: {CAL_METHOD}, θ=0.5)")
plt.savefig(os.path.join(OUT_DIR, f"OOF_confusion_matrix_CVCal_{CAL_METHOD}_theta0.5.png"),
            dpi=150, bbox_inches="tight")
plt.close()

with open(os.path.join(OUT_DIR, f"OOF_metrics_CVCal_{CAL_METHOD}_theta0.5.json"), "w", encoding="utf-8") as f:
    json.dump({
        "calibration": CAL_METHOD,
        "theta_fixed": float(theta_fixed),
        "oof_report": classification_report(
            y_oof, yhat_oof_cvcal, target_names=["Human","Machine"], output_dict=True
        ),
        "n_docs": int(len(y_oof)),
    }, f, ensure_ascii=False, indent=2)

# =========================
# 8) 오류 사례 목록(CV-calibrated @ θ=0.5)
# =========================
id_to_fname = dict(zip(df["doc_uid"].values, df["fname"].values))
is_fp = (y_oof == 0) & (yhat_oof_cvcal == 1)
is_fn = (y_oof == 1) & (yhat_oof_cvcal == 0)

def _collect(mask, etype):
    sel_ids = doc_ids_oof[mask]
    return pd.DataFrame({
        "doc_id": sel_ids,
        "fname": [id_to_fname[i] for i in sel_ids],
        "y_true": y_oof[mask],
        "y_pred": yhat_oof_cvcal[mask],
        "p_machine_cal": p_oof_cvcal[mask],
        "error_type": etype,
    })

df_fp = _collect(is_fp, "FP")
df_fn = _collect(is_fn, "FN")
df_mis = pd.concat([df_fp, df_fn], ignore_index=True).sort_values(
    ["error_type", "p_machine_cal"], ascending=[True, False]
)

df_fp.to_csv(os.path.join(OUT_DIR, "OOF_FP_CVCal.csv"), index=False)
df_fn.to_csv(os.path.join(OUT_DIR, "OOF_FN_CVCal.csv"), index=False)
df_mis.to_csv(os.path.join(OUT_DIR, "OOF_misclassified_CVCal.csv"), index=False)

with open(os.path.join(OUT_DIR, "OOF_error_summary_CVCal.json"), "w", encoding="utf-8") as f:
    json.dump({
        "num_fp": int(is_fp.sum()),
        "num_fn": int(is_fn.sum()),
        "fp_files": df_fp["fname"].tolist(),
        "fn_files": df_fn["fname"].tolist(),
    }, f, ensure_ascii=False, indent=2)

print(f"Saved FP/FN lists to: {OUT_DIR}")

# =========================
# 9) 배포용 최종 보정기 (ALL OOF → 1개)
# =========================
final_calib_dir = os.path.join(OUTPUT_MODEL_DIR, "final_calibrator")
os.makedirs(final_calib_dir, exist_ok=True)

if CAL_METHOD == "platt":
    X_all = _safe_logit(p_oof_raw).reshape(-1, 1)
    final_platt = SkLogReg(solver="lbfgs", C=1e6, max_iter=1000)
    final_platt.fit(X_all, y_oof)
    joblib_dump(final_platt, os.path.join(final_calib_dir, "platt.joblib"))
else:
    final_iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    final_iso.fit(p_oof_raw, y_oof)
    joblib_dump(final_iso, os.path.join(final_calib_dir, "isotonic.joblib"))

with open(os.path.join(final_calib_dir, "calibration_meta.json"), "w", encoding="utf-8") as f:
    json.dump({
        "calibration": CAL_METHOD,
        "note": "This calibrator is trained on ALL raw OOF and should be applied to raw test probs before thresholding.",
        "theta_for_inference": float(THETA_FIXED)
    }, f, ensure_ascii=False, indent=2)

# =========================
# 10) 최종 θ 저장 (고정 0.5)
# =========================
theta_path = os.path.join(OUTPUT_MODEL_DIR, "oof_threshold.json")
with open(theta_path, "w", encoding="utf-8") as ft:
    json.dump({
        "calibration": CAL_METHOD,
        "theta": float(THETA_FIXED),
        "selected_metric": "fixed_after_cv_calibration",
        "note": "Use CV-calibrated probs at theta=0.5 for reporting; for deployment, apply FINAL calibrator on raw probs then threshold 0.5."
    }, ft, ensure_ascii=False, indent=2)

print("\nAll done.")
