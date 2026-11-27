#!/usr/bin/env python3
"""
summarize.py
- 각 씬/실험의 train_stats.json + results.json을 읽어 논문용 비교표 출력
사용법: python summarize.py [--output-dir PATH]
"""

import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description="Compare baseline and progressive 3DGS results")
parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUT_ROOT", Path(__file__).resolve().parent / "output")))
OUT_ROOT = parser.parse_args().output_dir.expanduser().resolve()
SCENES   = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
EXPS     = {
    "baseline_2k_30k":   "Baseline (2K, 30k)",
    "progressive_3stage": "Progressive (0.5K→1K→2K)",
}

# ── 유틸 ─────────────────────────────────────────────────────────────────────
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

def fmt_time(seconds):
    """초 → mm:ss 표기"""
    if not seconds:
        return "N/A"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"

def get_metrics(model_dir):
    """results.json에서 test PSNR/SSIM/LPIPS 추출"""
    results_path = model_dir / "results.json"
    data = load_json(results_path)
    if not data:
        return None, None, None

    # 3DGS results.json 구조: {"ours_30000": {"PSNR": ..., "SSIM": ..., "LPIPS": ...}}
    # iteration 키가 여러 개일 수 있으니 마지막(최고) 것 사용
    for key in sorted(data.keys(), reverse=True):
        entry = data[key]
        psnr  = entry.get("PSNR",  entry.get("psnr",  None))
        ssim  = entry.get("SSIM",  entry.get("ssim",  None))
        lpips = entry.get("LPIPS", entry.get("lpips", None))
        if psnr is not None:
            return psnr, ssim, lpips

    return None, None, None

# ── 데이터 수집 ───────────────────────────────────────────────────────────────
rows = []   # (scene, exp_key, psnr, ssim, lpips, train_sec, peak_vram)

for scene in SCENES:
    for exp_key in EXPS:
        model_dir = OUT_ROOT / f"{scene}_{exp_key}"
        if not model_dir.exists():
            continue

        stats = load_json(model_dir / "train_stats.json")
        psnr, ssim, lpips = get_metrics(model_dir)

        rows.append({
            "scene":      scene,
            "exp":        exp_key,
            "psnr":       psnr,
            "ssim":       ssim,
            "lpips":      lpips,
            "train_sec":  stats.get("train_seconds") or stats.get("total_train_seconds"),
            "peak_vram":  stats.get("peak_vram_mib"),
            "num_gaussians": stats.get("num_gaussians"),
            "ply_mb":     stats.get("ply_size_mb"),
            "stages":     stats.get("stages", []),
        })

if not rows:
    print("결과 없음 — 학습이 완료된 후 실행하세요.")
    exit(0)

# ── 씬별 출력 ────────────────────────────────────────────────────────────────
SEP = "-" * 106
HDR = f"{'Scene':<12} {'Experiment':<26} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'Time':>9} {'VRAM(MiB)':>10} {'#Gauss':>10} {'PLY(MB)':>8}"

print("=" * 106)
print(" Blender (NeRF Synthetic) — Training Comparison")
print("=" * 106)
print(HDR)
print(SEP)

prev_scene = None
for r in rows:
    if prev_scene and prev_scene != r["scene"]:
        print(SEP)
    prev_scene = r["scene"]

    psnr_s  = f"{r['psnr']:.2f}"  if r['psnr']  else "N/A"
    ssim_s  = f"{r['ssim']:.4f}"  if r['ssim']  else "N/A"
    lpips_s = f"{r['lpips']:.4f}" if r['lpips'] else "N/A"
    time_s  = fmt_time(r['train_sec'])
    vram_s  = str(r['peak_vram']) if r['peak_vram'] else "N/A"

    gauss_s = f"{int(r['num_gaussians']):,}" if r.get('num_gaussians') else "N/A"
    ply_s   = str(r['ply_mb']) if r.get('ply_mb') else "N/A"

    print(f"{r['scene']:<12} {EXPS[r['exp']]:<26} {psnr_s:>7} {ssim_s:>7} {lpips_s:>7} {time_s:>9} {vram_s:>10} {gauss_s:>10} {ply_s:>8}")

    # Progressive: 단계별 VRAM 상세
    if r["stages"]:
        for s in r["stages"]:
            sname = s.get("name","")
            st    = fmt_time(s.get("train_seconds"))
            sv    = s.get("peak_vram_mib","N/A")
            print(f"{'':12}   └ {sname:<22} {'':>7} {'':>7} {'':>7} {st:>9} {sv:>10}")

print(SEP)

# ── 씬 평균 출력 ─────────────────────────────────────────────────────────────
print("\n[Per-Experiment Average across all scenes]")
print(f"{'Experiment':<26} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'Time':>9} {'VRAM(MiB)':>10} {'#Gauss':>10} {'PLY(MB)':>8}")
print("-" * 78)

for exp_key, exp_label in EXPS.items():
    exp_rows = [r for r in rows if r["exp"] == exp_key]
    if not exp_rows:
        continue

    def avg(key):
        vals = [r[key] for r in exp_rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    avg_psnr  = avg("psnr")
    avg_ssim  = avg("ssim")
    avg_lpips = avg("lpips")
    avg_time  = avg("train_sec")
    avg_vram  = avg("peak_vram")
    avg_gauss = avg("num_gaussians")
    avg_ply   = avg("ply_mb")

    print(f"{exp_label:<26} "
          f"{f'{avg_psnr:.2f}' if avg_psnr else 'N/A':>7} "
          f"{f'{avg_ssim:.4f}' if avg_ssim else 'N/A':>7} "
          f"{f'{avg_lpips:.4f}' if avg_lpips else 'N/A':>7} "
          f"{fmt_time(avg_time):>9} "
          f"{f'{avg_vram:.0f}' if avg_vram else 'N/A':>10} "
          f"{f'{avg_gauss:.0f}' if avg_gauss else 'N/A':>10} "
          f"{f'{avg_ply:.1f}' if avg_ply else 'N/A':>8}")

print("=" * 90 + "\n")

# ── Progressive 단계별 VRAM 요약 ────────────────────────────────────────────
prog_rows = [r for r in rows if r["exp"] == "progressive_3stage" and r["stages"]]
if prog_rows:
    print("[Progressive — Stage-wise VRAM & Time (avg across scenes)]")
    print(f"{'Stage':<20} {'Time':>9} {'VRAM(MiB)':>10}")
    print("-" * 42)
    stage_keys = [s["name"] for s in prog_rows[0]["stages"]]
    for sk in stage_keys:
        times = []
        vrams = []
        for r in prog_rows:
            for s in r["stages"]:
                if s["name"] == sk:
                    if s.get("train_seconds"): times.append(s["train_seconds"])
                    if s.get("peak_vram_mib"): vrams.append(s["peak_vram_mib"])
        at = sum(times)/len(times) if times else None
        av = sum(vrams)/len(vrams) if vrams else None
        print(f"{sk:<20} {fmt_time(at):>9} {f'{av:.0f}' if av else 'N/A':>10}")
    print()
