#!/usr/bin/env python3
from pathlib import Path
import re, csv
from collections import defaultdict
from difflib import SequenceMatcher

BASE_DIR = Path("../dataset")
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR  = BASE_DIR / "test"
OUT_CSV   = Path(".pseudo_labels.csv")

ALIASES = {
    "&": " and ",
    "_": " ",
    "+": " plus ",
}

PUNCT_RE = re.compile(r"[^\w\s]")  # 영숫자/밑줄/공백만 남김
WS_RE    = re.compile(r"\s+")

def normalize_title(s: str, *, is_train: bool) -> str:
    s = s.strip()
    s = re.sub(r"\.tex$", "", s, flags=re.I)
    if is_train:
        s = re.sub(r"(_[01])$", "", s)   # train 끝 라벨 제거
    s = re.sub(r"-\d+$", "", s)          # 끝 인덱스 -<digits> 제거

    # 치환(의미 있는 기호 보존): &,_ ,+ 처리
    for k, v in ALIASES.items():
        s = s.replace(k, v)

    # 소문자화
    s = s.lower()

    # 구두점 제거(단어 경계만 남게)
    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s

def key_from_title(s: str) -> str:
    # 키는 공백 제거한 압축형
    return s.replace(" ", "")

def tokens(s: str) -> set:
    return set(s.split())

def list_files(d: Path):
    return sorted([p.name for p in d.iterdir() if p.is_file() and not p.name.startswith(".")])

def jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def best_match(test_norm: str, train_norms: dict):
    """train_norms: norm_title -> labels(set)"""
    t_tok = tokens(test_norm)
    best = (None, 0.0, 0.0)  # (norm_title, jac, ratio)
    for tr_norm in train_norms.keys():
        tr_tok = tokens(tr_norm)
        jac = jaccard(t_tok, tr_tok)

        # 1차 필터: 토큰 겹침이 거의 없으면 스킵
        if jac < 0.3:
            # 부분 포함(긴/짧은 제목 케이스) 예외 허용
            if test_norm in tr_norm or tr_norm in test_norm:
                jac = max(jac, 0.3)
            else:
                continue

        ratio = SequenceMatcher(None, test_norm, tr_norm).ratio()
        # 스코어는 (jac * 0.6 + ratio * 0.4)로 종합
        score = 0.6 * jac + 0.4 * ratio
        if score > (0.6 * best[1] + 0.4 * best[2]):
            best = (tr_norm, jac, ratio)
    return best  # (matched_norm, jac, ratio)

def main():
    if not (TRAIN_DIR.exists() and TEST_DIR.exists()):
        raise SystemExit("train/test 디렉터리를 확인하세요.")

    train_files = list_files(TRAIN_DIR)
    test_files  = list_files(TEST_DIR)

    # --- train: 표준화 제목 -> 라벨 집합
    norm2labels = defaultdict(set)
    for fn in train_files:
        stem = re.sub(r"\.tex$", "", fn, flags=re.I)
        m = re.search(r"(_([01]))$", stem)
        if not m:
            continue
        label = int(m.group(2))
        norm = normalize_title(fn, is_train=True)
        norm2labels[norm].add(label)

    rows = []
    inferred = ambiguous = unknown = 0

    for fn in test_files:
        norm_t = normalize_title(fn, is_train=False)

        # 1) 정확 일치 시도
        labels = norm2labels.get(norm_t, set())
        matched_norm = norm_t
        jac = ratio = 1.0 if labels else 0.0

        # 2) 없으면 베스트 매칭 탐색
        if not labels:
            matched_norm, jac, ratio = best_match(norm_t, norm2labels)
            labels = norm2labels.get(matched_norm, set()) if matched_norm else set()

        # 3) 임계치: 충분히 비슷해야 인정 (경험적)
        similar_enough = (jac >= 0.45 and ratio >= 0.70) or (jac >= 0.30 and ratio >= 0.85)

        if labels and similar_enough and len(labels) == 1:
            lbl = next(iter(labels))
            pseudo = 1 - lbl
            inferred += 1
            rows.append({
                "split": "test",
                "filename": fn,
                "base_key": key_from_title(norm_t),
                "inferred_label": pseudo,
                "reason": f"matched '{matched_norm}' (train labels={sorted(labels)}; jac={jac:.2f}, ratio={ratio:.2f}) → counterpart {1-lbl}"
            })
        elif labels and similar_enough and len(labels) == 2:
            ambiguous += 1
            rows.append({
                "split": "test",
                "filename": fn,
                "base_key": key_from_title(norm_t),
                "inferred_label": "",
                "reason": f"matched '{matched_norm}' but both labels in train (0/1); jac={jac:.2f}, ratio={ratio:.2f}"
            })
        else:
            unknown += 1
            mn = matched_norm if matched_norm else ""
            rows.append({
                "split": "test",
                "filename": fn,
                "base_key": key_from_title(norm_t),
                "inferred_label": "",
                "reason": f"no confident match (best='{mn}', jac={jac:.2f}, ratio={ratio:.2f})"
            })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split","filename","base_key","inferred_label","reason"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] inferred={inferred}, ambiguous={ambiguous}, unknown={unknown}")
    print(f"[OK] saved: {OUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
