import os, re, random, json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)
from joblib import dump as joblib_dump 
from sklearn.isotonic import IsotonicRegression

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# utility functions
# =========================
def _safe_logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

def logit_mean_arr(ps):
    z = _safe_logit(np.asarray(ps))
    z_bar = z.mean()
    return 1/(1+np.exp(-z_bar))

def lse_pool(ps, tau=3.0):
    z = _safe_logit(np.asarray(ps))
    val = (1.0/tau)*np.log(np.exp(tau*z).sum())
    return 1/(1+np.exp(-val))

def topk_mean_arr(ps, k=3):
    ps = np.asarray(ps)
    k = min(k, len(ps))
    return np.sort(ps)[-k:].mean()

def build_doc_features(df_pred):
    """
    df_pred: columns = [doc_id, fname, y_true, p1] (청크 단위)
    return: (features_df, X, y_doc)  # 문서 단위
    """
    g = df_pred.groupby("doc_id")["p1"]
    feat = pd.DataFrame({
        "doc_id": g.size().index,
        "max":    g.max().values,
        "mean":   g.mean().values,
        "std":    g.std().fillna(0.0).values,
        "n":      g.size().values,
        "topk3":  g.apply(lambda s: topk_mean_arr(s.values, k=3)).values,
        "lse3":   g.apply(lambda s: lse_pool(s.values, tau=3.0)).values,
        "lmean":  g.apply(lambda s: logit_mean_arr(s.values)).values,
    })
    y_doc = df_pred.groupby("doc_id")["y_true"].first().reindex(feat["doc_id"]).values
    X = feat[["max","mean","std","n","topk3","lse3","lmean"]].values
    return feat, X, y_doc

def best_theta_for_accuracy(p, y):
    """Accuracy 기준 전역 θ 선택"""
    ts = np.linspace(0.0, 1.0, 1001)
    best_t, best_s = 0.5, -1.0
    for t in ts:
        yhat = (p >= t).astype(int)
        s = accuracy_score(y, yhat)
        if s > best_s:
            best_t, best_s = t, s
    return float(best_t), float(best_s)

# 지표함수
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
CAL_METHOD = "platt"

BASE_DIR   = "../../dataset/train"
MODEL_NAME = "microsoft/deberta-v3-large"
OUT_DIR    = "./kfold_results"
OUTPUT_MODEL_DIR = "./final_model"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# 폴드 별 Best 모델
FOLD_MODELS_DIR = os.path.join(OUTPUT_MODEL_DIR, "fold_models")
# 폴드별 로지스틱 집계기 모델
LOGREG_DIR      = os.path.join(OUTPUT_MODEL_DIR, "logreg_models")

os.makedirs(FOLD_MODELS_DIR, exist_ok=True)
os.makedirs(LOGREG_DIR, exist_ok=True)

# =========================
# 1) 유틸: .tex 파서
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
# 2) 전체 데이터 로드
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
# 3) 토크나이저 
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =========================
# 3-β) 슬라이딩 윈도우 청크 토크나이즈
# =========================
CHUNK_MAX_LEN = 512
CHUNK_STRIDE  = 256

def tokenize_fn_chunked(examples):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        max_length=CHUNK_MAX_LEN,
        stride=CHUNK_STRIDE,
        return_overflowing_tokens=True,
        padding=False,
        return_offsets_mapping=False
    )
    mapping = enc.get("overflow_to_sample_mapping", None)
    if mapping is None:
        mapping = list(range(len(enc["input_ids"])))
    labels = examples["label"]
    fnames = examples["fname"]
    uids   = examples["doc_uid"]
    enc["labels"] = [labels[i] for i in mapping]
    enc["fname"]  = [fnames[i] for i in mapping]
    enc["doc_id"] = [uids[i]   for i in mapping]
    return enc

# =========================
# 5) K-Fold + OOF 수집
# =========================
K = 5
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

fold_metrics = []

# OOF 확률(보정 전/후) 컨테이너
oof_probs_lr = []         # 기존: 보정 전 (LR-메타) 확률
oof_probs_cal = []        # 추가: 보정 후 확률(선택한 CAL_METHOD로)
oof_labels   = []
oof_doc_ids  = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df["text"], df["label"]), start=1):
    print(f"\n========== Fold {fold}/{K} ==========")
    fold_dir = os.path.join(OUT_DIR, f"fold-{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    # Dataset 생성 & 토크나이즈 (chunked)
    train_ds = Dataset.from_pandas(train_df).map(
        tokenize_fn_chunked, batched=True, remove_columns=list(train_df.columns)
    )
    val_ds = Dataset.from_pandas(val_df).map(
        tokenize_fn_chunked,   batched=True, remove_columns=list(val_df.columns)
    )
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # 모델
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, use_safetensors=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=fold_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
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
        lr_scheduler_type="linear",
        warmup_ratio=0.06,
        optim="adamw_torch",
        seed=SEED, data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=hf_metrics,
    )

    trainer.train()
    # ===== 폴드별 베스트 HF 모델 저장 =====
    fold_export_dir = os.path.join(FOLD_MODELS_DIR, f"fold-{fold}")
    os.makedirs(fold_export_dir, exist_ok=True)
    trainer.save_model(fold_export_dir) 
    tokenizer.save_pretrained(fold_export_dir)
    # 학습 세팅 저장
    with open(os.path.join(fold_export_dir, "training_args.json"), "w", encoding="utf-8") as fta:
        json.dump(training_args.to_dict(), fta, ensure_ascii=False, indent=2)


    # ---------- 예측: train/val 청크 확률 ----------
    # (로지스틱 집계기 학습은 train 문서 피처로, 예측은 val 문서 피처로)
    # train
    preds_tr  = trainer.predict(train_ds)
    logits_tr = preds_tr.predictions
    p1_tr     = torch.softmax(torch.tensor(logits_tr), dim=1).numpy()[:, 1]
    df_pred_tr = pd.DataFrame({
        "doc_id": np.array(train_ds["doc_id"]),
        "fname":  np.array(train_ds["fname"]),
        "y_true": preds_tr.label_ids,
        "p1":     p1_tr,
    })
    # val
    preds_val  = trainer.predict(val_ds)
    logits_val = preds_val.predictions
    p1_val     = torch.softmax(torch.tensor(logits_val), dim=1).numpy()[:, 1]
    df_pred_val = pd.DataFrame({
        "doc_id": np.array(val_ds["doc_id"]),
        "fname":  np.array(val_ds["fname"]),
        "y_true": preds_val.label_ids,
        "p1":     p1_val,
    })

    # ---------- 문서 단위 피처 ----------
    _, X_tr,  y_tr  = build_doc_features(df_pred_tr)
    feat_val, X_val, y_val = build_doc_features(df_pred_val)

    # ---------- 로지스틱 집계기 학습(Train 문서) → Val 문서 예측(OOF 수집) ----------
    if len(np.unique(y_tr)) >= 2:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(X_tr, y_tr)
        p_doc_val_lr = clf.predict_proba(X_val)[:, 1]

        # ===== 폴드별 로지스틱 메타모델 저장 =====
        logreg_export_dir = os.path.join(LOGREG_DIR, f"fold-{fold}")
        os.makedirs(logreg_export_dir, exist_ok=True)
        joblib_dump(clf, os.path.join(logreg_export_dir, "logreg_meta.joblib"))
        
        # 메타 피처 스키마 기록(열 순서 고정 참고용)
        with open(os.path.join(logreg_export_dir, "meta_feature_order.json"), "w", encoding="utf-8") as fm:
            json.dump(["max","mean","std","n","topk3","lse3","lmean"], fm, ensure_ascii=False, indent=2)

        # === 폴드별 확률 보정기 학습 & 저장 ===
        calib_dir = os.path.join(logreg_export_dir, "calibrator")
        os.makedirs(calib_dir, exist_ok=True)

        # 1) Platt scaling (1D 로지스틱). 입력은 logit(p)
        X_platt = _safe_logit(p_doc_val_lr).reshape(-1, 1)
        platt = LogisticRegression(
            solver="lbfgs", C=1e6, max_iter=1000, class_weight=None
        )
        platt.fit(X_platt, y_val)
        joblib_dump(platt, os.path.join(calib_dir, "platt.joblib"))

        # 2) Isotonic regression (단조 회귀). 입력은 p 자체
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_doc_val_lr, y_val)
        joblib_dump(iso, os.path.join(calib_dir, "isotonic.joblib"))

        # 이 폴드의 '검증 문서 확률'을 선택한 보정기로 변환해서 OOF에 쌓기
        if CAL_METHOD == "platt":
            X_platt = _safe_logit(p_doc_val_lr).reshape(-1, 1)
            p_doc_val_cal = platt.predict_proba(X_platt)[:, 1]
        elif CAL_METHOD == "isotonic":
            p_doc_val_cal = iso.predict(p_doc_val_lr)
        else:
            raise ValueError("CAL_METHOD must be 'platt' or 'isotonic'.")
                

    else:
        # 클래스 한쪽만 있는 이례적 상황: 단순 대안으로 mean 점수 사용(안정성 확보)
        p_doc_val_lr = X_val[:, 1]  # 'mean' 컬럼 위치
        # 보정값도 동일하게 맞춰서 OOF에 쌓이도록
        p_doc_val_cal = p_doc_val_lr

    # 기존(보정 전) OOF도 그대로 유지
    oof_probs_lr.append(p_doc_val_lr)
    # 추가: 보정 후 OOF
    oof_probs_cal.append(p_doc_val_cal)
    oof_labels.append(y_val)
    oof_doc_ids.append(feat_val["doc_id"].values)

    # (선택) 폴드별 리포트 저장
    y_hat_fold_tmp = (p_doc_val_lr >= 0.5).astype(int)
    with open(os.path.join(fold_dir, "val_doc_report_logreg_thresh0.5.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_val, y_hat_fold_tmp, target_names=["Human","Machine"]))

    del trainer, model, preds_tr, preds_val
    torch.cuda.empty_cache()

# =========================
# 6) OOF 전역 θ★(Accuracy 최대) 선택 & 리포트
# =========================
p_oof = np.concatenate(oof_probs_lr)
y_oof = np.concatenate(oof_labels)
doc_ids_oof = np.concatenate(oof_doc_ids)

theta_star, acc_star = best_theta_for_accuracy(p_oof, y_oof)
yhat_oof = (p_oof >= theta_star).astype(int)

print("\n===== OOF Results (LogReg aggregator) =====")
print(f"Best theta*(ACC): {theta_star:.4f} | ACC={acc_star:.4f}")
print(classification_report(y_oof, yhat_oof, target_names=["Human","Machine"]))

cm = confusion_matrix(y_oof, yhat_oof)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human","Machine"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"OOF Confusion Matrix (LogReg agg, theta*={theta_star:.3f})")
plt.savefig(os.path.join(OUT_DIR, "OOF_confusion_matrix_LogReg_ACC.png"), dpi=150, bbox_inches="tight")
plt.close()

with open(os.path.join(OUT_DIR, "OOF_metrics_LogReg_ACC.json"), "w", encoding="utf-8") as f:
    json.dump({
        "theta_star_acc": theta_star,
        "oof_acc": float(acc_star),
        "oof_report": classification_report(y_oof, yhat_oof, target_names=["Human","Machine"], output_dict=True),
        "n_docs": int(len(y_oof)),
    }, f, ensure_ascii=False, indent=2)

print(f"Saved OOF confusion matrix & metrics to: {OUT_DIR}")

# =========================
# 6-b) (추가) 보정 후 θ=0.5 고정 OOF 리포트
# =========================
p_oof_cal = np.concatenate(oof_probs_cal)

theta_fixed = 0.5
yhat_oof_cal = (p_oof_cal >= theta_fixed).astype(int)

print("\n===== OOF Results (Calibrated, theta=0.5) =====")
print(classification_report(y_oof, yhat_oof_cal, target_names=["Human","Machine"]))

cm_cal = confusion_matrix(y_oof, yhat_oof_cal)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_cal, display_labels=["Human","Machine"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"OOF Confusion Matrix (Calibrated: {CAL_METHOD}, theta=0.5)")
plt.savefig(os.path.join(OUT_DIR, f"OOF_confusion_matrix_{CAL_METHOD}_theta0.5.png"),
            dpi=150, bbox_inches="tight")
plt.close()

with open(os.path.join(OUT_DIR, f"OOF_metrics_{CAL_METHOD}_theta0.5.json"), "w", encoding="utf-8") as f:
    json.dump({
        "calibration": CAL_METHOD,
        "theta_fixed": theta_fixed,
        "oof_report": classification_report(
            y_oof, yhat_oof_cal, target_names=["Human","Machine"], output_dict=True
        ),
        "n_docs": int(len(y_oof)),
    }, f, ensure_ascii=False, indent=2)


# =========================
# 7) FP/FN 저장
# =========================
# doc_id → 파일명 매핑
id_to_fname = dict(zip(df["doc_uid"].values, df["fname"].values))

# 예측/오류 마스크 (보정 후 θ=0.5 기준)
y_pred_oof = yhat_oof_cal
p_oof_machine = p_oof_cal  # 1(=Machine) 확률(보정 후)
is_fp = (y_oof == 0) & (y_pred_oof == 1)  # Human을 Machine으로
is_fn = (y_oof == 1) & (y_pred_oof == 0)  # Machine을 Human으로

def _collect(mask, etype):
    sel_ids = doc_ids_oof[mask]
    return pd.DataFrame({
        "doc_id": sel_ids,
        "fname": [id_to_fname[i] for i in sel_ids],
        "y_true": y_oof[mask],
        "y_pred": y_pred_oof[mask],
        "p_machine": p_oof_machine[mask],
        "error_type": etype,
    })

df_fp = _collect(is_fp, "FP")
df_fn = _collect(is_fn, "FN")
df_mis = pd.concat([df_fp, df_fn], ignore_index=True).sort_values(
    ["error_type", "p_machine"], ascending=[True, False]
)

# CSV 저장
df_fp.to_csv(os.path.join(OUT_DIR, "OOF_FP.csv"), index=False)
df_fn.to_csv(os.path.join(OUT_DIR, "OOF_FN.csv"), index=False)
df_mis.to_csv(os.path.join(OUT_DIR, "OOF_misclassified.csv"), index=False)

# 요약 JSON 저장
with open(os.path.join(OUT_DIR, "OOF_error_summary.json"), "w", encoding="utf-8") as f:
    json.dump({
        "theta_star_acc": float(theta_star),
        "num_fp": int(is_fp.sum()),
        "num_fn": int(is_fn.sum()),
        "fp_files": df_fp["fname"].tolist(),
        "fn_files": df_fn["fname"].tolist(),
    }, f, ensure_ascii=False, indent=2)

print(f"Saved FP/FN lists to: {OUT_DIR}")

# =========================
# 8) 최종 OOF θ 저장
# =========================
theta_path = os.path.join(OUTPUT_MODEL_DIR, "oof_threshold.json")
with open(theta_path, "w", encoding="utf-8") as ft:
    json.dump({
        "calibration": CAL_METHOD,          # "platt" or "isotonic"
        "theta": 0.5,
        "selected_metric": "fixed_after_calibration",
        "note": "Used calibrated OOF probabilities with fixed theta=0.5"
    }, ft, ensure_ascii=False, indent=2)
