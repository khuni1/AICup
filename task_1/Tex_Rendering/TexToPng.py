#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TeX → PNG batch renderer (fixed canvas)
# Prereqs (system): latexmk or pdflatex, and pdftoppm (Poppler) or magick (ImageMagick)
# Prereqs (pip): Pillow

import os
import re
import sys
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image

# -----------------------------
# Paths from your snippet
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/train/"))
OUT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../rendered_dataset/train/"))

print("Data dir:", BASE_DIR)
print("Output dir:", OUT_DIR)

# -----------------------------
# Config
# -----------------------------
DPI = 300                 # rasterization dpi
CANVAS_W, CANVAS_H = 1024, 1024
BACKGROUND = "white"      # white/black/#RRGGBB
GLOB_PATTERN = "*.tex"    # how to pick TeX files (non-recursive)

# -----------------------------
# Helpers
# -----------------------------
def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def run(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_stem(name: str) -> str:
    s = re.sub(r'[^0-9a-zA-Z._-]+', '_', name)
    return s.strip('_') or 'tex'

# -----------------------------
# LaTeX wrappers
# -----------------------------
DEFAULT_PREAMBLE = r"""
\documentclass[preview,border=2pt]{standalone}
\usepackage{amsmath,amssymb,amsfonts,mathtools}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{physics}
\usepackage{bm}
\usepackage{hyperref}
\hypersetup{hidelinks}
\begin{document}
"""
DEFAULT_END = r"""
\end{document}
"""

def wrap_as_document(body: str) -> str:
    return DEFAULT_PREAMBLE + "\n" + body + "\n" + DEFAULT_END

def compile_tex_to_pdf(tex_content: str, workdir: Path, jobname: str) -> Path:
    ensure_dir(workdir)
    tex_path = workdir / f"{jobname}.tex"
    tex_path.write_text(tex_content, encoding="utf-8")

    if which("latexmk"):
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    elif which("pdflatex"):
        cmd = ["pdflatex", "-interaction=nonstopmode", tex_path.name]
    else:
        raise RuntimeError("Neither 'latexmk' nor 'pdflatex' found in PATH.")

    code, out, err = run(cmd, cwd=str(workdir))
    if code != 0 and "latexmk" in cmd[0] and which("pdflatex"):
        code, out, err = run(["pdflatex", "-interaction=nonstopmode", tex_path.name], cwd=str(workdir))
    if code != 0:
        raise RuntimeError(f"LaTeX compile failed:\n{(out + err)[-4000:]}")

    pdf_path = workdir / f"{jobname}.pdf"
    if not pdf_path.exists():
        cand = list(workdir.glob("*.pdf"))
        if not cand:
            raise RuntimeError("Compilation seems ok but no PDF found.")
        pdf_path = cand[0]
    return pdf_path

def pdf_to_png(pdf_path: Path, out_png: Path, dpi: int) -> Path:
    ensure_dir(out_png.parent)
    if which("pdftoppm"):
        prefix = out_png.with_suffix("").as_posix()
        code, out, err = run(["pdftoppm", "-png", "-singlefile", "-r", str(dpi), pdf_path.as_posix(), prefix])
        if code != 0:
            raise RuntimeError(f"pdftoppm failed:\n{out}\n{err}")
        gen = Path(prefix + ".png")
        if gen != out_png and gen.exists():
            gen.replace(out_png)
        return out_png
    elif which("magick"):
        code, out, err = run(["magick", "-density", str(dpi), f"{pdf_path.as_posix()}[0]", "-quality", "100", out_png.as_posix()])
        if code != 0:
            raise RuntimeError(f"magick failed:\n{out}\n{err}")
        return out_png
    else:
        raise RuntimeError("Need 'pdftoppm' or 'magick' in PATH.")

def pad_to_canvas(png_path: Path, width: int, height: int, background: str,
                  allow_upscale: bool = False, interp=Image.BICUBIC) -> Path:
    """
    Aspect 유지 + 패딩(letterbox)로 width x height 캔버스에 맞춤.
    - allow_upscale=False: 작은 이미지를 키우지 않음(권장, 문자 품질 보호)
    - 반환 파일명은 기존처럼 '<stem>_WIDTHxHEIGHT.png'
    """
    img = Image.open(png_path)
    # RGBA/LA 등 알파가 있으면 투명 유지, 아니면 RGB
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")

    orig_w, orig_h = img.width, img.height
    scale = min(width / orig_w, height / orig_h)
    if not allow_upscale:
        scale = min(scale, 1.0)

    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    if (new_w, new_h) != (orig_w, orig_h):
        img = img.resize((new_w, new_h), resample=interp)

    # 캔버스 생성 및 중앙 배치
    canvas_mode = "RGBA" if img.mode == "RGBA" else "RGB"
    canvas = Image.new(canvas_mode, (width, height), background)
    x = (width - img.width) // 2
    y = (height - img.height) // 2
    # 투명 채널이 있으면 마스크로 붙임
    if img.mode == "RGBA":
        canvas.paste(img, (x, y), img)
    else:
        canvas.paste(img, (x, y))

    # 완전 불투명 배경이면 최종 RGB로 저장
    if canvas_mode == "RGBA" and background.lower() in ("white", "black", "#ffffff", "#000000"):
        canvas = canvas.convert("RGB")

    out_path = png_path.with_name(png_path.stem + f"_{width}x{height}.png")
    canvas.save(out_path)
    return out_path


def render_body_to_png(body: str, outdir: Path, basename: str) -> Path:
    jobname = safe_stem(basename)
    workdir = Path(tempfile.mkdtemp(prefix=f"texbuild_{jobname}_"))
    try:
        pdf = compile_tex_to_pdf(wrap_as_document(body), workdir, jobname)
        raw_png = outdir / f"{jobname}.png"
        pdf_to_png(pdf, raw_png, DPI)
        final_png = pad_to_canvas(raw_png, CANVAS_W, CANVAS_H, BACKGROUND)
        return final_png
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

# -----------------------------
# Main: iterate BASE_DIR/*.tex
# -----------------------------
def main():
    base = Path(BASE_DIR)
    out  = Path(OUT_DIR)
    ensure_dir(out)

    tex_files = sorted(base.glob(GLOB_PATTERN))
    if not tex_files:
        print(f"[warn] No TeX files matched {GLOB_PATTERN} under {base}")
        return

    ok, fail = 0, 0
    for tex_path in tex_files:
        try:
            body = tex_path.read_text(encoding="utf-8")
            print(f"[render] {tex_path.name}")
            out_png = render_body_to_png(body, outdir=out, basename=tex_path.stem)
            print(f"  -> {out_png}")
            ok += 1
        except Exception as e:
            print(f"[error] {tex_path.name}: {e}", file=sys.stderr)
            fail += 1

    print(f"\nDone. success={ok}, failed={fail}, outdir={out}")


main()
