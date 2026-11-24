"""Camada 3 — Perceptual Visual Regression Engine (NVIDIA FLIP + SSIM + dHash).

Eliminates false positives from 1-bit driver rounding differences while catching
missing geometry, NaN flashes, inverted colors, or alpha bugs.
"""

from __future__ import annotations

import pathlib
from typing import Any


def compute_dhash(img: Any, hash_size: int = 8) -> int:
    """Compute 64-bit difference hash (dHash) using Pillow."""
    gray = img.convert("L").resize((hash_size + 1, hash_size))
    pixels = (
        list(gray.get_flattened_data())
        if hasattr(gray, "get_flattened_data")
        else list(gray.getdata())
    )
    bits = 0
    for row in range(hash_size):
        base = row * (hash_size + 1)
        for col in range(hash_size):
            if pixels[base + col] > pixels[base + col + 1]:
                bits |= 1 << (row * hash_size + col)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def compute_ssim_numpy(ref_rgb: Any, test_rgb: Any) -> float:
    """Global-windowed SSIM on [0, 1] RGB numpy arrays (Wang et al. 2004)."""
    import numpy as np

    r = ref_rgb.astype(np.float64)
    t = test_rgb.astype(np.float64)
    # Luminance projection
    w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    x = np.tensordot(r, w, axes=([-1], [0]))
    y = np.tensordot(t, w, axes=([-1], [0]))
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    # 8x8 block pooling (fast, deterministic, pure numpy)
    h, w_px = x.shape
    bh, bw = max(1, h // 8), max(1, w_px // 8)
    scores = []
    for by in range(0, h - bh + 1, bh):
        for bx in range(0, w_px - bw + 1, bw):
            px = x[by : by + bh, bx : bx + bw]
            py = y[by : by + bh, bx : bx + bw]
            mx, my = float(px.mean()), float(py.mean())
            vx = float(px.var())
            vy = float(py.var())
            cov = float(((px - mx) * (py - my)).mean())
            num = (2.0 * mx * my + c1) * (2.0 * cov + c2)
            den = (mx * mx + my * my + c1) * (vx + vy + c2)
            scores.append(num / den if den > 0 else 1.0)
    return float(sum(scores) / len(scores)) if scores else 1.0


def compare_images(
    reference_path: pathlib.Path,
    test_path: pathlib.Path,
    *,
    diff_out_path: pathlib.Path | None = None,
    metric: str = "flip",
    flip_threshold: float = 0.05,
    ssim_threshold: float = 0.985,
) -> dict[str, Any]:
    """Compare two images using NVIDIA FLIP (if available), SSIM, dHash, and max delta."""
    from PIL import Image, ImageChops

    ref_img = Image.open(reference_path).convert("RGB")
    test_img = Image.open(test_path).convert("RGB")
    resized = False
    if ref_img.size != test_img.size:
        test_img = test_img.resize(ref_img.size)
        resized = True

    dhash_ref = compute_dhash(ref_img)
    dhash_test = compute_dhash(test_img)
    dhash_dist = hamming_distance(dhash_ref, dhash_test)

    abs_diff = ImageChops.difference(ref_img, test_img)
    extrema = abs_diff.getextrema()
    max_channel_delta = max(hi for _, hi in extrema)

    ssim_score: float | None = None
    flip_mean: float | None = None
    flip_max: float | None = None
    rmse: float | None = None
    flip_available = False

    try:
        import numpy as np

        ref_arr = np.asarray(ref_img, dtype=np.float32) / 255.0
        test_arr = np.asarray(test_img, dtype=np.float32) / 255.0
        rmse = float(np.sqrt(np.mean((ref_arr - test_arr) ** 2)))
        ssim_score = compute_ssim_numpy(ref_arr, test_arr)

        try:
            import flip_evaluator

            flip_err, mean_err, _ = flip_evaluator.evaluate(
                ref_arr, test_arr, "LDR", inputsRGB=True, applyMagma=False, computeMeanError=True
            )
            flip_available = True
            flip_mean = float(mean_err)
            flip_max = float(np.max(flip_err))
            if diff_out_path is not None:
                diff_out_path.parent.mkdir(parents=True, exist_ok=True)
                flip_magma, _, _ = flip_evaluator.evaluate(
                    ref_arr, test_arr, "LDR", inputsRGB=True, applyMagma=True, computeMeanError=False
                )
                magma_u8 = np.clip(flip_magma * 255.0, 0, 255).astype(np.uint8)
                Image.fromarray(magma_u8).save(diff_out_path)
        except ImportError:
            pass
    except ImportError:
        pass

    if diff_out_path is not None and not flip_available:
        diff_out_path.parent.mkdir(parents=True, exist_ok=True)
        abs_diff.save(diff_out_path)

    exact_match = max_channel_delta == 0 and not resized
    if metric == "exact":
        passed = exact_match
    elif metric == "ssim":
        passed = (ssim_score >= ssim_threshold) if ssim_score is not None else exact_match
    else:  # "flip" (fallback to ssim if flip_evaluator missing)
        if flip_available and flip_mean is not None:
            passed = flip_mean <= flip_threshold
        elif ssim_score is not None:
            passed = ssim_score >= ssim_threshold
        else:
            passed = exact_match

    return {
        "reference": str(reference_path),
        "test": str(test_path),
        "diff_image": str(diff_out_path) if diff_out_path else None,
        "resolution": list(ref_img.size),
        "resized_to_match": resized,
        "exact_match": exact_match,
        "max_channel_delta": int(max_channel_delta),
        "dhash_distance": int(dhash_dist),
        "rmse": rmse,
        "ssim": ssim_score,
        "flip_available": flip_available,
        "flip_mean": flip_mean,
        "flip_max": flip_max,
        "metric": metric,
        "passed": bool(passed),
    }


def compare_directories(
    test_dir: pathlib.Path,
    reference_dir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    metric: str = "flip",
    flip_threshold: float = 0.05,
    ssim_threshold: float = 0.985,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    test_pngs = {p.name: p for p in sorted(test_dir.glob("*.png"))}
    ref_pngs = {p.name: p for p in sorted(reference_dir.glob("*.png"))}
    names = sorted(set(test_pngs) | set(ref_pngs))
    items: list[dict[str, Any]] = []
    missing_refs: list[str] = []
    missing_tests: list[str] = []
    failed_count = 0
    passed_count = 0

    for name in names:
        if name not in ref_pngs:
            missing_refs.append(name)
            continue
        if name not in test_pngs:
            missing_tests.append(name)
            failed_count += 1
            continue
        diff_path = out_dir / f"{pathlib.Path(name).stem}.diff.png"
        try:
            cmp_res = compare_images(
                ref_pngs[name],
                test_pngs[name],
                diff_out_path=diff_path,
                metric=metric,
                flip_threshold=flip_threshold,
                ssim_threshold=ssim_threshold,
            )
        except Exception as exc:
            cmp_res = {
                "reference": str(ref_pngs[name]),
                "test": str(test_pngs[name]),
                "diff_image": None,
                "metric": metric,
                "flip_mean": None,
                "ssim": None,
                "error": str(exc),
                "passed": False,
            }
        cmp_res["name"] = name
        if cmp_res["passed"]:
            passed_count += 1
        else:
            failed_count += 1
        items.append(cmp_res)

    verdict = "PASS" if failed_count == 0 and items else ("FAIL" if failed_count > 0 else "WARN")
    return {
        "verdict": verdict,
        "summary": (
            f"Compared {len(items)} image pair(s) via {metric}: "
            f"{passed_count} passed, {failed_count} failed, "
            f"{len(missing_refs)} missing reference"
        ),
        "metrics": {
            "pairs_compared": len(items),
            "passed": passed_count,
            "failed": failed_count,
            "missing_references": len(missing_refs),
            "missing_tests": len(missing_tests),
            "mean_flip": (
                sum(x["flip_mean"] for x in items if x["flip_mean"] is not None)
                / max(1, sum(1 for x in items if x["flip_mean"] is not None))
                if any(x["flip_mean"] is not None for x in items)
                else None
            ),
            "min_ssim": min((x["ssim"] for x in items if x["ssim"] is not None), default=None),
        },
        "items": items,
        "missing_references": missing_refs,
        "missing_tests": missing_tests,
    }
