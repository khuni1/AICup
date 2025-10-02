#!/usr/bin/env python3
import os, re, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Any

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer
)
from joblib import load as joblib_load

# =========================
# 경로/세팅
# =========================
BASE_TEST_DIR      = "../../dataset/test"                 # 테스트 .tex 디렉토리
OUTPUT_MODEL_DIR   = "./final_model_bigbird_doc"         # ★ BigBird 학습 스크립트의 OUTPUT_MODEL_DIR
FOLD_MODELS_DIR    = os.path.join(OUTPUT_MODEL_DIR, "fold_models")
CALIB_DIR_ROOT     = os.path.join(OUTPUT_MODEL_DIR, "calibrators")
RESULT_DIR         = "./final_model_result_bigbird"      # 결과 저장 폴더
os.makedirs(RESULT_DIR, exist_ok=True)a

OUT_PRED_CSV       = os.path.join(RESULT_DIR, "test_predictions.csv")
MODEL_NAME         = "google/bigbird-roberta-base"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# =========================
# 유틸
# =========================
def _safe_logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

def apply_platt(platt, p):
    z = _safe_logit(p).reshape(-1, 1)
    return platt.predict_proba(z)[:, 1]

def apply_isotonic(iso, p):
    return iso.predict(p)

def parse_tex_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    blocks = {}
    for block in ["Input","Output","Formula","Explanation"]:
        m = re.search(rf"%{block}(.*?)(?=%|$)", text, re.S)
        blocks[block.lower()] = m.group(1).strip() if m else ""
    return blocks

def load_test_dataframe(test_dir):
    rows = []
    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith(".tex"):
            continue
        p = os.path.join(test_dir, fname)
        b = parse_tex_file(p)
        full_text = " ".join(b.values())
        rows.append({"fname": fname, "text": full_text})
    df = pd.DataFrame(rows).reset_index(drop=True)
    df["doc_id"] = np.arange(len(df), dtype=int)
    return df

# =========================
# 토크나이저 / Collator
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

MIN_LEN_FOR_SPARSE = 768   # 12 * 64 (안정 버퍼)
PAD_MULTIPLE = 64

@dataclass
class MinLenPadCollator:
    tokenizer: Any
    min_len: int = MIN_LEN_FOR_SPARSE
    multiple: int = PAD_MULTIPLE

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # fname, doc_id는 토크나이저가 처리하지 않으니 분리/복원 필요 없음 (Trainer는 텐서 키만 사용)
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

data_collator = MinLenPadCollator(tokenizer=tokenizer, min_len=MIN_LEN_FOR_SPARSE, multiple=PAD_MULTIPLE)

# =========================
# 토크나이즈(문서 단위)
# =========================
MAX_LEN = 4096

def tokenize_fn_doc(examples):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
        return_offsets_mapping=False
    )
    # 추후 결과 매핑을 위해 보존
    enc["fname"]  = examples["fname"]
    enc["doc_id"] = examples["doc_id"]
    return enc

# =========================
# fold별 예측 (문서 확률)
# =========================
def predict_with_fold(fold_idx, test_df, calibrator="platt"):
    # 모델 로드
    fold_dir = os.path.join(FOLD_MODELS_DIR, f"fold-{fold_idx}")
    model = AutoModelForSequenceClassification.from_pretrained(fold_dir)
    model.config.attention_type = "block_sparse"
    model.config.num_random_blocks = 2
    model.to(DEVICE).eval()

    # 데이터셋 (문서 단위)
    ds = Dataset.from_pandas(test_df).map(
        tokenize_fn_doc, batched=True, remove_columns=list(test_df.columns)
    )
    # 위에서 fname/doc_id를 enc에 넣었으므로 남아있음
    ds.set_format("torch")

    # 예측
    trainer = Trainer(model=model, tokenizer=tokenizer, data_collator=data_collator)
    preds = trainer.predict(ds)
    logits = preds.predictions
    p_doc = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    # === 폴드 보정기 적용 ===
    calib_dir = os.path.join(CALIB_DIR_ROOT, f"fold-{fold_idx}")
    if calibrator == "platt":
        path = os.path.join(calib_dir, "platt.joblib")
        if os.path.exists(path):
            platt = joblib_load(path)
            p_doc = apply_platt(platt, p_doc)
    elif calibrator == "isotonic":
        path = os.path.join(calib_dir, "isotonic.joblib")
        if os.path.exists(path):
            iso = joblib_load(path)
            p_doc = apply_isotonic(iso, p_doc)

    # 반환: doc_id 순서대로 확률
    id2fname = dict(zip(test_df["doc_id"].values, test_df["fname"].values))
    # ds 내부에도 fname/doc_id가 있으나, 여기서는 원본 순서 사용
    order_ids = test_df["doc_id"].values
    fnames    = [id2fname[i] for i in order_ids]
    return pd.DataFrame({
        "doc_id": order_ids,
        "fname": fnames,
        f"p_fold{fold_idx}": p_doc
    })

# =========================
# 앙상블 & 임계치 적용
# =========================
def ensemble_probs(prob_cols, how="mean"):
    arr = np.vstack(prob_cols)  # shape: (n_folds, n_docs)
    if how == "mean":
        return arr.mean(axis=0)
    elif how == "topk3":
        # fold 수가 3 미만이면 자동으로 가능한 k로 줄어듦
        k = min(3, arr.shape[0])
        return np.sort(arr, axis=0)[-k:, :].mean(axis=0)
    else:  # "logit_mean"
        z = _safe_logit(arr)
        return 1.0 / (1.0 + np.exp(-z.mean(axis=0)))

def predict_test(
    test_dir=BASE_TEST_DIR,
    folds=(1,2,3,4,5),
    ensemble="mean",   # "logit_mean" | "mean" | "topk3"
    theta=0.5,
    save_csv=OUT_PRED_CSV, 
    calibrator="platt"  # "platt" | "isotonic"
):
    test_df = load_test_dataframe(test_dir)

    fold_frames = []
    for k in folds:
        print(f"[Fold {k}] predicting ...")
        df_k = predict_with_fold(k, test_df, calibrator=calibrator)
        fold_frames.append(df_k)

    # doc_id/fname 기준으로 머지
    df_ens = fold_frames[0][["doc_id","fname"]].copy()
    for df_k in fold_frames:
        df_ens = df_ens.merge(df_k, on=["doc_id","fname"], how="left")

    # 앙상블 확률
    prob_cols = [df_ens[c].values for c in df_ens.columns if c.startswith("p_fold")]
    p_ens = ensemble_probs(prob_cols, how=ensemble)
    yhat = (p_ens >= theta).astype(int)

    df_ens["p_machine"] = p_ens
    df_ens["y_pred"]    = yhat
    df_ens["ensemble"]  = ensemble
    df_ens["theta"]     = theta

    # 보기 좋게 정렬
    prob_colnames = [c for c in df_ens.columns if c.startswith("p_fold")]
    df_out = df_ens[["doc_id","fname"] + prob_colnames + ["p_machine","y_pred","ensemble","theta"]].sort_values("doc_id")

    # ===== 시각화 저장 =====
    counts = df_out["y_pred"].value_counts().reindex([0, 1], fill_value=0)
    plt.figure()
    counts.plot(kind="bar")
    plt.xticks([0, 1], ["Human (0)", "Machine (1)"], rotation=0)
    plt.ylabel("Count")
    plt.title("Predicted class distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "pred_class_distribution.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.hist(df_out["p_machine"].values, bins=50)
    plt.xlabel("Predicted P(machine)")
    plt.ylabel("Docs")
    plt.title("P(machine) distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "p_machine_hist.png"), dpi=150)
    plt.close()

    # 텍스트/JSON 요약
    total = int(len(df_out))
    human_cnt   = int(counts.get(0, 0))
    machine_cnt = int(counts.get(1, 0))
    human_pct   = 100.0 * human_cnt / max(total, 1)
    machine_pct = 100.0 * machine_cnt / max(total, 1)

    print(f"[Summary] Predicted distribution -> "
          f"Human(0): {human_cnt} ({human_pct:.2f}%) | "
          f"Machine(1): {machine_cnt} ({machine_pct:.2f}%) | "
          f"Total: {total} | theta={theta}")

    with open(os.path.join(RESULT_DIR, "pred_summary.txt"), "w", encoding="utf-8") as fsum:
        fsum.write(
            f"Predicted class distribution\n"
            f"- Human(0):   {human_cnt} ({human_pct:.2f}%)\n"
            f"- Machine(1): {machine_cnt} ({machine_pct:.2f}%)\n"
            f"- Total:      {total}\n"
            f"- Ensemble:   {ensemble}\n"
            f"- Theta:      {theta}\n"
        )

    summary = {
        "total_docs": total,
        "human_count": human_cnt,
        "machine_count": machine_cnt,
        "human_pct": human_pct,
        "machine_pct": machine_pct,
        "theta": float(theta),
        "ensemble": ensemble,
        "p_machine_stats": {
            "mean": float(np.mean(df_out["p_machine"])),
            "std":  float(np.std(df_out["p_machine"])),
            "min":  float(np.min(df_out["p_machine"])),
            "max":  float(np.max(df_out["p_machine"])),
            "q10":  float(np.quantile(df_out["p_machine"], 0.10)),
            "q50":  float(np.quantile(df_out["p_machine"], 0.50)),
            "q90":  float(np.quantile(df_out["p_machine"], 0.90)),
        }
    }
    with open(os.path.join(RESULT_DIR, "pred_summary.json"), "w", encoding="utf-8") as fj:
        json.dump(summary, fj, ensure_ascii=False, indent=2)

    if save_csv:
        df_out.to_csv(save_csv, index=False, encoding="utf-8")
        print(f"[OK] saved predictions: {save_csv}")
    return df_out

# ============== 예시 실행 ==============
if __name__ == "__main__":
    _ = predict_test(
        test_dir=BASE_TEST_DIR,
        folds=(1,2,3,4,5),
        ensemble="mean",      # "mean" | "logit_mean" | "topk3"
        theta=0.5,            # 보정 후 0.5 컷
        save_csv=OUT_PRED_CSV,
        calibrator="platt"    # "platt" | "isotonic"
    )
