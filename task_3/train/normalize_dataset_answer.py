#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
normalize_dataset_answer.py

- 입력: JSONL (각 줄이 {"id": "...", "image": "...", "question": "...", "solution": "...", "answer": "..."} 형태)
- 출력: nomalized_train.json  (요청하신 파일명 그대로 씁니다)

기능:
- answer 필드가 '숫자(+과학표기) + 단위' 형태면 단위 제거
- 예) "88.4%", "1.5A", "240Ω", "33℃", "28.0cm", "3kw", "6000元" → 각각 "88.4", "1.5", "240", "33", "28.0", "3", "6000"
- "≥6.4" 처럼 비교기호가 앞에 있으면 숫자만 추출 → "6.4"
- 숫자가 아닌 답(B, 无法确定, Fe(OH)3, 纤维素和淀粉 등)은 변경하지 않음
"""

import re
import json
import argparse
from typing import Optional

# 숫자 패턴: 부호/소수/과학표기(×10^k, x10^k, e표기 일부) 까지 허용
NUM_CORE = r'[-+]?\d+(?:\.\d+)?'
SCI_TAIL = r'(?:\s*[×x]\s*10\s*(?:\^)?\s*[-+]?\d+)?'  # ×10^-16, x10^3 등
NUM_PATTERN = rf'{NUM_CORE}{SCI_TAIL}'

# answer 전체가 (선행 비교기호/공백) + 숫자(과학표기 포함) + (뒤에 단위들) 형태일 때 매칭
# 단위 목록은 영문/기호/중국어 단위의 꼬리부를 폭넓게 포괄 (필요 시 추가 가능)
UNIT_TAIL = (
    r'(?:'                         # 영문/기호
    r'[a-zA-Z％%℃°VvAaWwKkJjOoΩmMcmskgG]|'  # 단일문자 단위도 포함
    r'(?:cm|mm|km|kg|kW|KW|kw|KW|KW|KV|mv|mL|MV|Ω|ohm|Ohm|OHM|A|V|W|J|Pa)'
    r')+'
    r'|'                           # 중국어 단위들
    r'(?:元|人|天|次|辆|米|厘米|毫米|千米|千克|克|度|环|伏|安|欧姆|瓦|焦)'
)

TRAILING_UNITS = re.compile(
    rf'^\s*[≮≤≥><~≈]?\s*'          # 선행 비교/근사 기호 허용
    rf'({NUM_PATTERN})'            # 그룹1: 핵심 숫자(과학표기 포함)
    rf'\s*{UNIT_TAIL}\s*$'         # 뒤에 단위 꼬리
)

# 비교/기호가 앞에 있고 뒤엔 단위가 없어도 숫자만 있는 경우: "≥6.4", "≈30"
LEADING_SIGN_NUM = re.compile(
    rf'^\s*[≮≤≥><~≈]?\s*({NUM_PATTERN})\s*$'
)

# 문자열 어딘가에 숫자가+단위로 붙은 보편 패턴도 한 번 더 시도 (예: "28.0cm" 처럼 딱 붙은 케이스)
EMBEDDED_NUM_UNIT = re.compile(
    rf'^\s*({NUM_PATTERN})\s*{UNIT_TAIL}\s*$'
)

def normalize_answer(ans: str) -> str:
    """
    answer가 '숫자(+과학표기)+단위'인 경우 단위를 제거하여 숫자 부분만 반환.
    숫자가 아닌 텍스트(분류라벨/화학식/문장 등)는 원본 유지.
    """
    if not isinstance(ans, str):
        return ans

    s = ans.strip()

    # 완전 일치: 숫자 + 단위
    m = TRAILING_UNITS.match(s)
    if m:
        return m.group(1).replace(' ', '')

    # 임베디드: 숫자 + 단위 (공백 유무 상관 없이 딱 붙은 형태)
    m = EMBEDDED_NUM_UNIT.match(s)
    if m:
        return m.group(1).replace(' ', '')

    # 비교기호만 있고 단위는 없는 경우 → 숫자만 뽑기 (예: ≥6.4 → 6.4)
    m = LEADING_SIGN_NUM.match(s)
    if m:
        return m.group(1).replace(' ', '')

    # 그 외: 변경하지 않음 (예: "B", "Fe(OH)3", "无法确定", "纤维素和淀粉")
    return ans


def main():
    parser = argparse.ArgumentParser(description="Remove units from 'answer' field and write nomalized_train.json")
    parser.add_argument("input_jsonl", nargs="?",default="train.json", help="입력 JSONL 파일 경로")
    parser.add_argument("-o", "--output", default="nomalized_train.json",
                        help="출력 파일명 (기본: nomalized_train.json)")
    args = parser.parse_args()

    total = 0
    changed = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1

            if "answer" in obj and isinstance(obj["answer"], str):
                orig = obj["answer"]
                norm = normalize_answer(orig)
                if norm != orig:
                    changed += 1
                obj["answer"] = norm

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Done. total={total}, normalized={changed}, output={args.output}")


if __name__ == "__main__":
    main()
