import os, re, random, json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)

# =========================
# 0) 재현성 & 경로
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

BASE_DIR = "../../dataset/train"     # 전체 데이터 위치
MODEL_NAME = "allenai/scibert_scivocab_uncased"
OUT_DIR = "./kfold_results_scibert"
os.makedirs(OUT_DIR, exist_ok=True)

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
        label = 0 if "_0" in fname else 1
        rows.append({"fname": fname, "text": full_text, "label": label})
df = pd.DataFrame(rows).reset_index(drop=True)

print("=== 전체 라벨 분포 ===")
print(df["label"].value_counts(normalize=True).sort_index().rename({0:"Human",1:"Machine"}))

# =========================
# 3) 토크나이저
# =========================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# =========================
# 4) 지표 함수
# =========================
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
# 5) K-Fold 설정
# =========================
K = 5  # 필요에 따라 변경
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

fold_metrics = []
for fold, (train_idx, val_idx) in enumerate(skf.split(df["text"], df["label"]), start=1):
    print(f"\n========== Fold {fold}/{K} ==========")
    fold_dir = os.path.join(OUT_DIR, f"fold-{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    # Dataset 생성 & 토크나이즈
    train_ds = Dataset.from_pandas(train_df)
    val_ds   = Dataset.from_pandas(val_df)
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds   = val_ds.map(tokenize_fn, batched=True)

    # __index_level_0__ 제거 (pandas->datasets 부작용 컬럼)
    for ds in (train_ds, val_ds):
        cols_to_remove = [c for c in ["text","__index_level_0__","fname"] if c in ds.column_names]
        ds = ds.remove_columns(cols_to_remove) if cols_to_remove else ds
    # 위에서 재바인딩 필요
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c in ["text","__index_level_0__","fname"]]) if any(c in train_ds.column_names for c in ["text","__index_level_0__","fname"]) else train_ds
    val_ds   = val_ds.remove_columns([c for c in val_ds.column_names if c in ["text","__index_level_0__","fname"]]) if any(c in val_ds.column_names for c in ["text","__index_level_0__","fname"]) else val_ds

    train_ds.set_format("torch"); val_ds.set_format("torch")

    # 매 fold마다 새 모델 초기화
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=fold_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir=os.path.join(fold_dir, "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=SEED,
        data_seed=SEED,
        report_to="none",  # 필요시 wandb 등으로 변경
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

    # fold 성능 평가
    metrics = trainer.evaluate(eval_dataset=val_ds)
    print("Fold Evaluation:", metrics)

    # 상세 리포트/혼동행렬
    preds = trainer.predict(val_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)

    report = classification_report(y_true, y_pred, target_names=["Human","Machine"], output_dict=True)
    print(classification_report(y_true, y_pred, target_names=["Human","Machine"]))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human","Machine"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.savefig(os.path.join(fold_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 저장
    with open(os.path.join(fold_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(fold_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 집계용
    fold_metrics.append({
        "fold": fold,
        "accuracy": metrics.get("eval_accuracy"),
        "f1_macro": metrics.get("eval_f1_macro"),
        "recall_macro": metrics.get("eval_recall_macro"),
    })

    # 메모리 정리 (GPU 사용 시 권장)
    del trainer, model, preds
    torch.cuda.empty_cache()

# =========================
# 6) 평균/표준편차 집계
# =========================
res_df = pd.DataFrame(fold_metrics)
print("\n===== K-Fold Summary =====")
print(res_df)

def mean_std(col): 
    return f"{res_df[col].mean():.4f} ± {res_df[col].std():.4f}"

print("Accuracy :", mean_std("accuracy"))
print("F1 Macro :", mean_std("f1_macro"))
print("Recall Macro :", mean_std("recall_macro"))

res_df.to_csv(os.path.join(OUT_DIR, "kfold_summary.csv"), index=False)
