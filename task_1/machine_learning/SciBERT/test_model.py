from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 최신 체크포인트 경로
model_dir = "./results/checkpoint-205"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()