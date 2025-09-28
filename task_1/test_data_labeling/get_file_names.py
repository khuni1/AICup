#!/usr/bin/env python3
from pathlib import Path

# --- 설정 ---
BASE_DIR = Path("../dataset")
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR  = BASE_DIR / "test"
OUT_FILE  = Path("fnames.txt")

def list_files(dir_path: Path):
    if not dir_path.exists():
        print(f"[WARN] Not found: {dir_path.resolve()}")
        return []
    # 파일만, 숨김 제외
    return sorted([p.name for p in dir_path.iterdir() if p.is_file() and not p.name.startswith(".")])

def main():
    train_files = list_files(TRAIN_DIR)
    test_files  = list_files(TEST_DIR)

    # 콘솔 출력
    print(f"== TRAIN ({len(train_files)}) ==")
    for name in train_files:
        print(name)
    print(f"\n== TEST ({len(test_files)}) ==")
    for name in test_files:
        print(name)

    # 파일 저장
    with OUT_FILE.open("w", encoding="utf-8") as f:
        f.write(f"== TRAIN ({len(train_files)}) ==\n")
        for name in train_files:
            f.write(f"{name}\n")
        f.write(f"\n== TEST ({len(test_files)}) ==\n")
        for name in test_files:
            f.write(f"{name}\n")

    print(f"\n[OK] Saved to {OUT_FILE.resolve()}")

main()