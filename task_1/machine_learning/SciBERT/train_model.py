import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def parse_tex_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    blocks ={}
    for block in ["Input","Output", "Formula", "Explanation"]:
        pattern = rf"%{block}(.*?)(?=%|$)"  
        match = re.search(pattern, text, re.S)
        if match:
            blocks[block.lower()] = match.group(1).strip()
        else:
            blocks[block.lower()] = ""
    return blocks

data = []
base_dir = '../../dataset/train'
for fname in os.listdir(base_dir):
    if fname.endswith('.tex'):
        file_path = os.path.join(base_dir, fname)
        blocks = parse_tex_file(file_path)
        full_text = " ".join(blocks.values())
        label = 0 if "_0" in fname else 1
        data.append({"text": full_text, "label": label})

df = pd.DataFrame(data)

# 데이터셋 라벨 분포 확인
print("=== 데이터셋 라벨 분포 ===")
print(f"전체 데이터 수: {len(df)}")
print("라벨 분포:")
label_counts = df['label'].value_counts().sort_index()
for label, count in label_counts.items():
    label_name = "Human" if label == 0 else "Machine"
    percentage = count / len(df) * 100
    print(f"  {label} ({label_name}): {count}개 ({percentage:.1f}%)")
print()

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=43)

# 훈련/검증 데이터셋 분포 확인
print("=== 훈련/검증 데이터셋 분포 ===")
print(f"훈련 데이터 수: {len(train_df)}")
train_label_counts = train_df['label'].value_counts().sort_index()
for label, count in train_label_counts.items():
    label_name = "Human" if label == 0 else "Machine"
    percentage = count / len(train_df) * 100
    print(f"  {label} ({label_name}): {count}개 ({percentage:.1f}%)")

print(f"\n검증 데이터 수: {len(val_df)}")
val_label_counts = val_df['label'].value_counts().sort_index()
for label, count in val_label_counts.items():
    label_name = "Human" if label == 0 else "Machine"
    percentage = count / len(val_df) * 100
    print(f"  {label} ({label_name}): {count}개 ({percentage:.1f}%)")
print()

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ===========================================
# 4. Tokenizer & Model 준비 (BERT)
# ===========================================
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)

train_dataset = train_dataset.remove_columns(["text", "__index_level_0__"])
val_dataset = val_dataset.remove_columns(["text", "__index_level_0__"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ===========================================
# 5. 학습 설정
# ===========================================
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
)

# ===========================================
# 6. Trainer 객체 생성 & 학습
# ===========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=lambda p: {
        "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean().item()
    },
)

trainer.train()

# ===========================================
# 7. 최종 평가
# ===========================================
metrics = trainer.evaluate(eval_dataset=val_dataset)
print("Final Evaluation:", metrics)

# 검증 데이터셋 예측
preds = trainer.predict(val_dataset)

# 정답 라벨과 예측 라벨
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

report = classification_report(
    y_true, y_pred, target_names=["Human", "Machine"], output_dict=True
)
print(classification_report(y_true, y_pred, target_names=["Human", "Machine"]))

# F1 & Recall 출력
print("📊 Recall (Macro):", report['macro avg']['recall'])
print("📊 F1-score (Macro):", report['macro avg']['f1-score'])

# 혼동행렬 출력
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# 시각화
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "Machine"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# 추가: 상세 리포트
print(classification_report(y_true, y_pred, target_names=["Human", "Machine"]))