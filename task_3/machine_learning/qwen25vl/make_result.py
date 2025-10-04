import re
import pandas as pd

in_csv  = "./fold2_test_predictions.csv"   # 입력 파일
out_csv = "./task_3_result.csv"            # 출력 파일

df = pd.read_csv(in_csv, dtype=str)        # 숫자도 문자열로 읽기(형식 보존)

# 태그만 제거(내용 보존): <calc>...</calc>, <final>...</final>
TAG_ONLY_RE = re.compile(r"</?\s*(calc|final)\s*>", flags=re.I)

def strip_tags_keep_content(s: str) -> str:
    if pd.isna(s):
        return s
    s = TAG_ONLY_RE.sub("", s)   # 태그만 삭제, 내부 텍스트는 그대로
    return s.strip()

# pred 컬럼 정리 (pred/Pred 둘 다 대응)
pred_col = "pred" if "pred" in df.columns else ("Pred" if "Pred" in df.columns else None)
if pred_col is None:
    raise ValueError("입력 CSV에 'pred' 또는 'Pred' 컬럼이 없습니다.")

df[pred_col] = df[pred_col].map(strip_tags_keep_content)

# 필요한 컬럼만 선택 + 이름 변경
id_col = "id" if "id" in df.columns else ("Id" if "Id" in df.columns else None)
if id_col is None:
    raise ValueError("입력 CSV에 'id' 또는 'Id' 컬럼이 없습니다.")

out = df[[id_col, pred_col]].copy().rename(columns={id_col: "id", pred_col: "model_answer"})
out.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"saved -> {out_csv}")
