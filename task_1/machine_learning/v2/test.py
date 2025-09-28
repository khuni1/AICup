#!/usr/bin/env python3
import os, re, json, math
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from joblib import load as joblib_load
import matplotlib.pyplot as plt
from joblib import load as joblib_load

# =========================
# 경로/세팅 (학습 시 경로와 동일해야 함)
# =========================
BASE_TEST_DIR      = "../../dataset/test"   # 테스트 .tex 디렉토리
OUTPUT_MODEL_DIR   = "./final_model"        # 학습 스크립트에서 저장한 폴더
FOLD_MODELS_DIR    = os.path.join(OUTPUT_MODEL_DIR, "fold_models")
LOGREG_DIR         = os.path.join(OUTPUT_MODEL_DIR, "logreg_models")
OUT_PRED_CSV       = "./test_predictions.csv"

# 최종 결과 저장 폴더
RESULT_DIR         = "./final_model_result"
os.makedirs(RESULT_DIR, exist_ok=True)

# [CHANGE] 결과 CSV 경로를 결과 폴더로
OUT_PRED_CSV       = os.path.join(RESULT_DIR, "test_predictions.csv")

MODEL_NAME         = "microsoft/deberta-v3-large"
CHUNK_MAX_LEN      = 512
CHUNK_STRIDE       = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# =========================
# 유틸들 (학습 시 함수와 동일)
# =========================
def _safe_logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

def logit_mean_arr(ps):
    z = _safe_logit(np.asarray(ps))
    return 1.0 / (1.0 + np.exp(-z.mean()))

def topk_mean_arr(ps, k=3):
    ps = np.asarray(ps)
    if len(ps) == 0:
        return 0.5
    k = min(k, len(ps))
    return np.sort(ps)[-k:].mean()

def lse_pool(ps, tau=3.0):
    z = _safe_logit(np.asarray(ps))
    val = (1.0/tau)*np.log(np.exp(tau*z).sum())
    return 1.0/(1.0+np.exp(-val))

def parse_tex_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    blocks = {}
    for block in ["Input","Output","Formula","Explanation"]:
        m = re.search(rf"%{block}(.*?)(?=%|$)", text, re.S)
        blocks[block.lower()] = m.group(1).strip() if m else ""
    return blocks

def build_doc_features(df_pred):
    """
    df_pred: columns = [doc_id, fname, p1]  (test에는 y_true 없음)
    return: feat_df(문서 단위 피처), X(피처 행렬), order_doc_ids
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
    X = feat[["max","mean","std","n","topk3","lse3","lmean"]].values
    return feat, X, feat["doc_id"].values
def apply_platt(platt, p):
    z = _safe_logit(p).reshape(-1, 1)
    return platt.predict_proba(z)[:, 1]

def apply_isotonic(iso, p):
    return iso.predict(p)

# =========================
# 토크나이저 (공유)
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
    fnames = examples["fname"]
    uids   = examples["doc_id"]
    enc["fname"]  = [fnames[i] for i in mapping]
    enc["doc_id"] = [uids[i]   for i in mapping]
    return enc

# =========================
# 테스트 데이터 로드
# =========================
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
# fold별 예측 → 문서 확률
# =========================
def predict_with_fold(fold_idx, test_df, calibrator="platt"):
    fold_dir = os.path.join(FOLD_MODELS_DIR, f"fold-{fold_idx}")
    logreg_dir = os.path.join(LOGREG_DIR, f"fold-{fold_idx}")

    # 모델 / 토크나이저 로드
    model = AutoModelForSequenceClassification.from_pretrained(fold_dir)
    model.to(DEVICE)
    model.eval()

    # 데이터셋
    ds = Dataset.from_pandas(test_df).map(
        tokenize_fn_chunked, batched=True, remove_columns=list(test_df.columns)
    )
    ds.set_format("torch")

    # 예측
    trainer = Trainer(model=model, tokenizer=tokenizer)
    preds = trainer.predict(ds)
    logits = preds.predictions
    p1 = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    # 청크 단위 → 문서 피처
    df_pred = pd.DataFrame({
        "doc_id": np.array(ds["doc_id"]),
        "fname":  np.array(ds["fname"]),
        "p1":     p1,
    })
    feat_df, X_doc, order_ids = build_doc_features(df_pred)

    # 로지스틱 집계기 로드
    logreg_path = os.path.join(logreg_dir, "logreg_meta.joblib")
    if os.path.exists(logreg_path):
        clf = joblib_load(logreg_path)
        p_doc = clf.predict_proba(X_doc)[:, 1]
    else:
        # (예외) logreg 없으면 평균 점수로 대체
        p_doc = X_doc[:, 1]  # 'mean' 컬럼
    
    # === [ADD] 보정기 적용 ===
    calib_dir = os.path.join(logreg_dir, "calibrator")
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
    
    # 반환: doc_id 순서대로 확률, 해당 파일명 맵
    # (order_ids는 feat_df["doc_id"], 파일명은 test_df에서 매핑)
    id2fname = dict(zip(test_df["doc_id"].values, test_df["fname"].values))
    fnames = [id2fname[i] for i in order_ids]
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
        return np.apply_along_axis(topk_mean_arr, 0, arr, k=min(3, arr.shape[0]))
    else:  # "logit_mean" (기본)
        return np.apply_along_axis(logit_mean_arr, 0, arr)

def predict_test(
    test_dir=BASE_TEST_DIR,
    folds=(1,2,3,4,5),
    ensemble="mean",   # "logit_mean" | "mean" | "topk3"
    theta=0.5,
    save_csv=OUT_PRED_CSV, 
    calibrator="platt"
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

    # ===== [ADD] 분포도 저장 (결과 폴더에 저장) =====
    # 1) 예측 라벨 분포 막대그래프
    counts = df_out["y_pred"].value_counts().reindex([0, 1], fill_value=0)
    plt.figure()
    counts.plot(kind="bar")
    plt.xticks([0, 1], ["Human (0)", "Machine (1)"], rotation=0)
    plt.ylabel("Count")
    plt.title("Predicted class distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "pred_class_distribution.png"), dpi=150)
    plt.close()

    # 2) p(machine) 히스토그램
    plt.figure()
    plt.hist(df_out["p_machine"].values, bins=50)
    plt.xlabel("Predicted P(machine)")
    plt.ylabel("Docs")
    plt.title("P(machine) distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "p_machine_hist.png"), dpi=150)
    plt.close()

    # 3) 비율 집계/출력/저장 
    total = int(len(df_out))
    human_cnt   = int(counts.get(0, 0))
    machine_cnt = int(counts.get(1, 0))
    human_pct   = 100.0 * human_cnt / max(total, 1)
    machine_pct = 100.0 * machine_cnt / max(total, 1)

    print(f"[Summary] Predicted distribution -> "
          f"Human(0): {human_cnt} ({human_pct:.2f}%) | "
          f"Machine(1): {machine_cnt} ({machine_pct:.2f}%) | "
          f"Total: {total} | theta={theta}")

    # 텍스트로도 저장
    with open(os.path.join(RESULT_DIR, "pred_summary.txt"), "w", encoding="utf-8") as fsum:
        fsum.write(
            f"Predicted class distribution\n"
            f"- Human(0):   {human_cnt} ({human_pct:.2f}%)\n"
            f"- Machine(1): {machine_cnt} ({machine_pct:.2f}%)\n"
            f"- Total:      {total}\n"
            f"- Ensemble:   {ensemble}\n"
            f"- Theta:      {theta}\n"
        )

    # JSON 요약(추가로 확률 통계 포함)
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
    # ==============================================

    if save_csv:
        df_out.to_csv(save_csv, index=False, encoding="utf-8")
        print(f"[OK] saved predictions: {save_csv}")
    return df_out

# ============== 예시 실행 ==============
if __name__ == "__main__":
    _ = predict_test(
        test_dir=BASE_TEST_DIR,
        folds=(1,2,3,4,5),
        ensemble="mean",
        theta=0.5,                 # 보정 후 0.5 컷
        save_csv=OUT_PRED_CSV,
        calibrator="platt"         # 또는 "isotonic"
    )

    