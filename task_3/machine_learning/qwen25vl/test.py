# infer_repl_train.py
import os, json, re
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

TRAIN_JSON = "../train/train.json"      # 고정
IMAGES_DIR = "../train/images"          # 고정
MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT_SYSTEM = (
    "你是表格理解与计算助手。给你一张拍摄的中文表格图片和一个与表相关的问题。"
    "请先在心里从表格中定位需要的字段并进行必要的计算，但**不要展示过程**。"
    "严格要求：只输出最终答案一个字符串；不要单位；不要标点；不要解释。"
    "若是判断题，只能输出“是”或“否”。若表格无法回答，输出“无法判断”。"
)

def normalize_answer(s: str) -> str:
    s = s.strip()
    if s in ("是", "否"):
        return s
    if "无法判断" in s:
        return "无法判断"
    m = list(re.finditer(r"[-+]?\d+(?:\.\d+)?", s))
    if m:
        return m[-1].group(0)
    return s.replace(" ", "").replace("\n", "")

def load_index(jsonl_path: str):
    """JSONL을 모두 읽어 id→레코드 dict로 인덱싱(한 번만)."""
    index = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            obj = json.loads(line)
            index[str(obj["id"])] = obj
    return index

def main():
    # 0) 데이터 인덱싱
    print(f"[INFO] loading index from {TRAIN_JSON} ...")
    idx = load_index(TRAIN_JSON)
    print(f"[INFO] loaded {len(idx)} items")

    # 1) 모델/프로세서 (한 번만 로드)
    print("[INFO] loading model ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("[INFO] model ready.")

    # 2) 대화형 루프
    print("\n=== REPL ===")
    print("입력: 원하는 id (예: 1).  종료: q 또는 빈 입력")
    while True:
        try:
            s = input("id> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT]")
            break

        if not s or s.lower() == "q":
            print("[EXIT]")
            break

        if s not in idx:
            print(f"[WARN] id={s} 를 찾을 수 없습니다. 다시 입력하세요.")
            continue

        item = idx[s]
        q = item["question"]
        gt = item.get("answer")  # train에는 정답이 있음

        # 이미지 경로는 JSON 상대경로를 사용하지만, 확장자는 .jpg만 사용
        # (train.json의 image가 jpg라고 가정)
        img_name = os.path.basename(item["image"])
        img_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] 이미지가 없습니다: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")

        # 프롬프트 구성
        messages = [
            {"role": "system", "content": [{"type": "text", "text": PROMPT_SYSTEM}]},
            {"role": "user",   "content": [{"type": "image"}, {"type": "text", "text": q}]},
        ]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[chat_text],
            images=[img],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False, temperature=0.0)

        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
        gen_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        pred = normalize_answer(gen_text)

        print("\n----- RESULT -----")
        print(f"id       : {item['id']}")
        print(f"question : {q}")
        print(f"image    : {img_path}")
        print(f"raw      : {gen_text}")
        print(f"pred     : {pred}")
        if gt is not None:
            print(f"answer   : {gt}  {'✅' if str(gt)==str(pred) else '❌'}")
        print("------------------\n")

if __name__ == "__main__":
    main()
