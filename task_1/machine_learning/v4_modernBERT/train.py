import os, re, random, json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, set_seed, DataCollatorWithPadding
)
from transformers import EarlyStoppingCallback
from sklearn.isotonic import IsotonicRegression
from joblib import dump as joblib_dump
from typing import Dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# utility functions
# =========================
def _safe_logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

def best_theta_for_accuracy(p, y):
    ts = np.linspace(0.0, 1.0, 1001)
    best_t, best_s = 0.5, -1.0
    for t in ts:
        s = accuracy_score(y, (p >= t).astype(int))
        if s > best_s:
            best_t, best_s = t, s
    return float(best_t), float(best_s)

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

# =========================
# 0) 재현성 & 경로
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

CAL_METHOD = "platt"   # "platt" or "isotonic"

BASE_DIR   = "../../dataset/train"
MODEL_NAME = "answerdotai/ModernBERT-large"   # ★ ModernBERT
OUT_DIR    = "./kfold_results_modernbert_doc"
OUTPUT_MODEL_DIR = "./final_model_modernbert_doc"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# 폴드 별 베스트 모델/보정기 저장
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
# 3) 토크나이저 (ModernBERT, 4096)
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
MAX_LEN = 4096

def tokenize_fn(examples):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,              # 동적 패딩
        return_offsets_mapping=False
    )
    enc["labels"] = examples["label"]
    return enc

# pad_to_multiple_of=64 → 텐서코어 활용
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=64)

# =========================
# 4) K-Fold + OOF 수집 (문서 단위)
# =========================
K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

oof_probs, oof_probs_cal, oof_labels, oof_doc_ids = [], [], [], []

supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold, (train_idx, val_idx) in enumerate(skf.split(df["text"], df["label"]), start=1):
    print(f"\n========== Fold {fold}/{K} ==========")
    fold_dir = os.path.join(OUT_DIR, f"fold-{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    train_ds = Dataset.from_pandas(train_df).map(
        tokenize_fn, batched=True, remove_columns=list(train_df.columns)
    )
    val_ds = Dataset.from_pandas(val_df).map(
        tokenize_fn, batched=True, remove_columns=list(val_df.columns)
    )
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, use_safetensors=True, trust_remote_code=True
    ).to(device)

    training_args = TrainingArguments(
        output_dir=fold_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,       # 4→8로 상향 (4096 기준, VRAM에 맞게 조절)
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,       # 유효 배치 8
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
        group_by_length=False,               # 문서 길이 편차가 크면 False 권장
        bf16=supports_bf16,
        fp16=not supports_bf16,
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="adamw_torch",
        seed=SEED, data_seed=SEED,
        label_smoothing_factor=0.1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=1e-4)],
    )

    trainer.train()

    # ===== 폴드별 베스트 모델 저장 =====
    fold_export_dir = os.path.join(FOLD_MODELS_DIR, f"fold-{fold}")
    os.makedirs(fold_export_dir, exist_ok=True)
    trainer.save_model(fold_export_dir)
    tokenizer.save_pretrained(fold_export_dir)
    with open(os.path.join(fold_export_dir, "training_args.json"), "w", encoding="utf-8") as fta:
        json.dump(training_args.to_dict(), fta, ensure_ascii=False, indent=2)

    # ---------- 문서 단위 예측 (val) ----------
    preds_val = trainer.predict(val_ds)
    p1_val = torch.softmax(torch.tensor(preds_val.predictions), dim=1).numpy()[:, 1]
    y_val  = preds_val.label_ids

    # ---------- 폴드별 보정기 학습 & 저장 ----------
    calib_dir = os.path.join(CALIB_DIR_ROOT, f"fold-{fold}")
    os.makedirs(calib_dir, exist_ok=True)

    # Platt (logit(p) -> calibrated p)
    X_platt = _safe_logit(p1_val).reshape(-1, 1)
    from sklearn.linear_model import LogisticRegression as SkLogReg
    platt = SkLogReg(solver="lbfgs", C=1e6, max_iter=1000)
    platt.fit(X_platt, y_val)
    joblib_dump(platt, os.path.join(calib_dir, "platt.joblib"))

    # Isotonic (p -> calibrated p)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p1_val, y_val)
    joblib_dump(iso, os.path.join(calib_dir, "isotonic.joblib"))

    # 선택 보정기로 val 확률 보정 → OOF에 쌓기
    if CAL_METHOD == "platt":
        p_val_cal = platt.predict_proba(X_platt)[:, 1]
    elif CAL_METHOD == "isotonic":
        p_val_cal = iso.predict(p1_val)
    else:
        raise ValueError("CAL_METHOD must be 'platt' or 'isotonic'.")

    # ----- OOF 누적 -----
    oof_probs.append(p1_val)
    oof_probs_cal.append(p_val_cal)
    oof_labels.append(y_val)
    oof_doc_ids.append(val_df["doc_uid"].values)

    del trainer, model, preds_val
    torch.cuda.empty_cache()

# =========================
# 5) OOF 리포트 (보정 전 ACC-최대 θ*)
# =========================
p_oof  = np.concatenate(oof_probs)
y_oof  = np.concatenate(oof_labels)
doc_ids_oof = np.concatenate(oof_doc_ids)

theta_star, acc_star = best_theta_for_accuracy(p_oof, y_oof)
yhat_oof = (p_oof >= theta_star).astype(int)

print("\n===== OOF Results (Raw probs) =====")
print(f"Best theta*(ACC): {theta_star:.4f} | ACC={acc_star:.4f}")
print(classification_report(y_oof, yhat_oof, target_names=["Human","Machine"]))

cm = confusion_matrix(y_oof, yhat_oof)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human","Machine"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"OOF Confusion Matrix (ModernBERT raw, theta*={theta_star:.3f})")
plt.savefig(os.path.join(OUT_DIR, "OOF_confusion_matrix_ModernBERT_raw.png"), dpi=150, bbox_inches="tight")
plt.close()

with open(os.path.join(OUT_DIR, "OOF_metrics_ModernBERT_raw.json"), "w", encoding="utf-8") as f:
    json.dump({
        "theta_star_acc": theta_star,
        "oof_acc": float(acc_star),
        "oof_report": classification_report(y_oof, yhat_oof, target_names=["Human","Machine"], output_dict=True),
        "n_docs": int(len(y_oof)),
    }, f, ensure_ascii=False, indent=2)

# =========================
# 6) OOF (보정 후, θ=0.5 고정)
# =========================
p_oof_cal = np.concatenate(oof_probs_cal)
theta_fixed = 0.5
yhat_oof_cal = (p_oof_cal >= theta_fixed).astype(int)

print("\n===== OOF Results (Calibrated, theta=0.5) =====")
print(classification_report(y_oof, yhat_oof_cal, target_names=["Human","Machine"]))

cm_cal = confusion_matrix(y_oof, yhat_oof_cal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_cal, display_labels=["Human","Machine"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"OOF Confusion Matrix (ModernBERT calibrated: {CAL_METHOD}, theta=0.5)")
plt.savefig(os.path.join(OUT_DIR, f"OOF_confusion_matrix_ModernBERT_{CAL_METHOD}_theta0.5.png"),
            dpi=150, bbox_inches="tight")
plt.close()

with open(os.path.join(OUT_DIR, f"OOF_metrics_ModernBERT_{CAL_METHOD}_theta0.5.json"), "w", encoding="utf-8") as f:
    json.dump({
        "calibration": CAL_METHOD,
        "theta_fixed": theta_fixed,
        "oof_report": classification_report(
            y_oof, yhat_oof_cal, target_names=["Human","Machine"], output_dict=True
        ),
        "n_docs": int(len(y_oof)),
    }, f, ensure_ascii=False, indent=2)

# =========================
# 7) FP/FN 리스트 저장 (보정 후 기준)
# =========================
id_to_fname = dict(zip(df["doc_uid"].values, df["fname"].values))
is_fp = (y_oof == 0) & (yhat_oof_cal == 1)
is_fn = (y_oof == 1) & (yhat_oof_cal == 0)

def _collect(mask, etype):
    sel_ids = doc_ids_oof[mask]
    return pd.DataFrame({
        "doc_id": sel_ids,
        "fname": [id_to_fname[i] for i in sel_ids],
        "y_true": y_oof[mask],
        "y_pred": yhat_oof_cal[mask],
        "p_machine_cal": p_oof_cal[mask],
        "error_type": etype,
    })

df_fp = _collect(is_fp, "FP")
df_fn = _collect(is_fn, "FN")
df_mis = pd.concat([df_fp, df_fn], ignore_index=True).sort_values(
    ["error_type", "p_machine_cal"], ascending=[True, False]
)

df_fp.to_csv(os.path.join(OUT_DIR, "OOF_FP.csv"), index=False)
df_fn.to_csv(os.path.join(OUT_DIR, "OOF_FN.csv"), index=False)
df_mis.to_csv(os.path.join(OUT_DIR, "OOF_misclassified.csv"), index=False)

with open(os.path.join(OUT_DIR, "OOF_error_summary.json"), "w", encoding="utf-8") as f:
    json.dump({
        "theta_star_acc_raw": float(theta_star),
        "num_fp": int(is_fp.sum()),
        "num_fn": int(is_fn.sum()),
        "fp_files": df_fp["fname"].tolist(),
        "fn_files": df_fn["fname"].tolist(),
    }, f, ensure_ascii=False, indent=2)

print(f"Saved FP/FN lists to: {OUT_DIR}")

# =========================
# 8) 최종 OOF θ 저장 (보정 후 0.5 사용)
# =========================
theta_path = os.path.join(OUTPUT_MODEL_DIR, "oof_threshold.json")
with open(theta_path, "w", encoding="utf-8") as ft:
    json.dump({
        "calibration": CAL_METHOD,          # "platt" or "isotonic"
        "theta": 0.5,
        "selected_metric": "fixed_after_calibration",
        "note": "Used calibrated OOF probabilities with fixed theta=0.5 (doc-level, ModernBERT)"
    }, ft, ensure_ascii=False, indent=2)
