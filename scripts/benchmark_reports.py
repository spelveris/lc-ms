#!/usr/bin/env python3
"""Benchmark deconvolution output against reference PDF reports.

This script:
1. Parses each report PDF for the deconvolution time window and component table.
2. Runs the current app deconvolution on the paired .D folder using that window.
3. Compares detected masses to report masses and prints match metrics.

By default, matching is "reference-wise": each report component is compared to its
nearest detected mass (a detected mass may satisfy multiple nearby references).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyMuPDF is required. Install with: pip install pymupdf") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))

from analysis import deconvolute_protein_local_lcms_machine_like, detect_singly_charged, sum_spectra_in_range
from data_reader import read_sample


WINDOW_RE = re.compile(
    r"Deconvolution of Spectrum #\s*\d+\s*@\s*([0-9]+\.[0-9]+)\s*-\s*([0-9]+\.[0-9]+)\s*min"
)
ROW_RE = re.compile(
    r"^\s*([A-Z])\s+([0-9]+\.[0-9]+)\s+([0-9]+)\s+([0-9]+\.[0-9]+)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class SampleSpec:
    name: str
    data_path: Path
    report_path: Path


@dataclass(frozen=True)
class ReferenceComponent:
    label: str
    mass: float
    absolute: int
    relative: float


@dataclass(frozen=True)
class PredictedComponent:
    mass: float
    intensity: float
    source: str  # "deconv" or "z1"


@dataclass(frozen=True)
class RefMatch:
    component: ReferenceComponent
    nearest: PredictedComponent | None
    abs_error: float
    matched: bool


DEFAULT_SAMPLES = [
    SampleSpec(
        name="LA_SUMOliraglutide_wt",
        data_path=Path("/Volumes/chab_loc_lang_s1/Dominykas Spelveris/9. LC-MS/LA_SUMOliraglutide_wt.D"),
        report_path=Path("/Volumes/chab_loc_lang_s1/Dominykas Spelveris/9. LC-MS/LA_SUMOliraglutide_wt.pdf"),
    ),
    SampleSpec(
        name="VW mGold test BCBIV",
        data_path=Path("/Volumes/chab_loc_lang_s1/Dominykas Spelveris/9. LC-MS/VW mGold test BCBIV.D"),
        report_path=Path("/Volumes/chab_loc_lang_s1/Dominykas Spelveris/9. LC-MS/VW mGold test BCBIV.pdf"),
    ),
    SampleSpec(
        name="VW mOrange test BCBIV",
        data_path=Path("/Volumes/chab_loc_lang_s1/Dominykas Spelveris/9. LC-MS/VW mOrange test BCBIV.D"),
        report_path=Path("/Volumes/chab_loc_lang_s1/Dominykas Spelveris/9. LC-MS/VW mOrange test BCBIV.pdf"),
    ),
]


def resolve_sample_path(path: Path) -> Path:
    """Resolve sample/report path across legacy and comparison-folder layouts."""
    if path.exists():
        return path

    parent = path.parent
    alt = parent / "Comparison files" / path.name
    if alt.exists():
        return alt

    return path


def parse_report(report_path: Path) -> tuple[tuple[float, float], list[ReferenceComponent]]:
    report_path = resolve_sample_path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    with fitz.open(report_path) as doc:
        text = "\n".join((doc.load_page(i).get_text("text") or "") for i in range(doc.page_count))

    win_match = WINDOW_RE.search(text)
    if not win_match:
        raise ValueError(f"Could not parse deconvolution window from {report_path}")
    time_window = (float(win_match.group(1)), float(win_match.group(2)))

    components = []
    for label, mass, absolute, relative in ROW_RE.findall(text):
        components.append(
            ReferenceComponent(
                label=label,
                mass=float(mass),
                absolute=int(absolute),
                relative=float(relative),
            )
        )

    if not components:
        raise ValueError(f"Could not parse component table from {report_path}")

    return time_window, components


def dedupe_by_mass(components: Iterable[PredictedComponent], tolerance_da: float) -> list[PredictedComponent]:
    kept: list[PredictedComponent] = []
    for comp in components:
        if all(abs(comp.mass - existing.mass) > tolerance_da for existing in kept):
            kept.append(comp)
    return kept


def run_prediction(
    sample_path: Path,
    time_window: tuple[float, float],
    args: argparse.Namespace,
) -> list[PredictedComponent]:
    sample_path = resolve_sample_path(sample_path)
    sample = read_sample(str(sample_path))
    if sample.error:
        raise RuntimeError(f"Failed to read sample {sample_path}: {sample.error}")

    mz, intensity = sum_spectra_in_range(sample, time_window[0], time_window[1])
    if len(mz) == 0:
        raise RuntimeError(f"No MS spectrum points for {sample_path} in window {time_window}")

    deconv = deconvolute_protein_local_lcms_machine_like(
        mz,
        intensity,
        min_charge=args.min_charge,
        max_charge=args.max_charge,
        min_peaks=args.min_peaks,
        noise_cutoff=args.noise_cutoff,
        abundance_cutoff=args.abundance_cutoff,
        mw_agreement=args.mw_agreement,
        mw_assign_cutoff=args.mw_assign_cutoff,
        envelope_cutoff=args.envelope_cutoff,
        pwhh=args.pwhh,
        low_mw=args.low_mw,
        high_mw=args.high_mw,
        contig_min=args.contig_min,
        use_mz_agreement=args.use_mz_agreement,
        use_monoisotopic_proton=args.use_monoisotopic_proton,
    )

    predicted = [
        PredictedComponent(mass=float(r["mass"]), intensity=float(r["intensity"]), source="deconv")
        for r in deconv
        if args.low_mw <= float(r["mass"]) <= args.high_mw
    ]

    if args.include_singly_charged:
        z1 = detect_singly_charged(
            mz,
            intensity,
            noise_cutoff=args.noise_cutoff,
            min_intensity_pct=args.z1_min_intensity_pct,
            low_mw=args.low_mw,
            high_mw=args.high_mw,
            pwhh=args.pwhh,
            exclude_mz_ranges=None,
            use_monoisotopic_proton=args.use_monoisotopic_proton,
        )
        predicted.extend(
            PredictedComponent(mass=float(r["mass"]), intensity=float(r["intensity"]), source="z1")
            for r in z1
        )

    predicted.sort(key=lambda c: c.intensity, reverse=True)
    predicted = dedupe_by_mass(predicted, tolerance_da=args.pred_dedupe_da)
    return predicted[: args.top_n]


def match_reference_wise(
    references: list[ReferenceComponent],
    predicted: list[PredictedComponent],
    tolerance_da: float,
) -> list[RefMatch]:
    if not predicted:
        return [RefMatch(component=r, nearest=None, abs_error=float("inf"), matched=False) for r in references]

    out: list[RefMatch] = []
    for ref in references:
        nearest = min(predicted, key=lambda p: abs(p.mass - ref.mass))
        err = abs(nearest.mass - ref.mass)
        out.append(RefMatch(component=ref, nearest=nearest, abs_error=err, matched=(err <= tolerance_da)))
    return out


def match_unique_greedy(
    references: list[ReferenceComponent],
    predicted: list[PredictedComponent],
    tolerance_da: float,
) -> int:
    if not references or not predicted:
        return 0

    # Match highest-abundance references first.
    ordered_refs = sorted(references, key=lambda r: r.relative, reverse=True)
    unused = set(range(len(predicted)))
    matched = 0

    for ref in ordered_refs:
        if not unused:
            break
        best_idx = min(unused, key=lambda idx: abs(predicted[idx].mass - ref.mass))
        if abs(predicted[best_idx].mass - ref.mass) <= tolerance_da:
            matched += 1
            unused.remove(best_idx)
    return matched


def format_mass(component: PredictedComponent | None) -> str:
    if component is None:
        return "n/a"
    return f"{component.mass:.2f} ({component.source})"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark LC-MS deconvolution against PDF report components.")
    p.add_argument("--relative-threshold", type=float, default=5.0, help="Minimum report relative abundance (%%).")
    p.add_argument(
        "--match-tolerance-da",
        type=float,
        default=5.0,
        help="Mass match tolerance in Da (default 5.0 matches current manual comparison style).",
    )
    p.add_argument("--top-n", type=int, default=20, help="Number of predicted masses to keep after dedupe.")
    p.add_argument(
        "--pred-dedupe-da",
        type=float,
        default=0.05,
        help="Deduplicate predicted masses within this absolute Da tolerance.",
    )

    # Local LC-MS machine-like defaults
    p.add_argument("--low-mw", type=float, default=500.0)
    p.add_argument("--high-mw", type=float, default=50000.0)
    p.add_argument("--min-charge", type=int, default=5)
    p.add_argument("--max-charge", type=int, default=50)
    p.add_argument("--min-peaks", type=int, default=3)
    p.add_argument("--noise-cutoff", type=float, default=1000.0)
    p.add_argument("--abundance-cutoff", type=float, default=0.10)
    p.add_argument("--mw-agreement", type=float, default=0.0005)
    p.add_argument("--mw-assign-cutoff", type=float, default=0.40)
    p.add_argument("--envelope-cutoff", type=float, default=0.50)
    p.add_argument("--pwhh", type=float, default=0.6)
    p.add_argument("--contig-min", type=int, default=3)
    p.add_argument("--use-mz-agreement", action="store_true")
    p.add_argument("--use-monoisotopic-proton", action="store_true")

    # Optional z=1 merge path for future comparisons.
    p.add_argument("--include-singly-charged", action="store_true")
    p.add_argument("--z1-min-intensity-pct", type=float, default=1.0)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    total_refs = 0
    total_refwise_matches = 0
    total_unique_matches = 0

    print("Benchmark: Deconvolution vs PDF Report")
    print(f"Settings: rel>={args.relative_threshold:.1f}% | tol={args.match_tolerance_da:.2f} Da | top_n={args.top_n}")
    print(
        "Method: Local LC-MS machine-like"
        + (" + singly-charged merge" if args.include_singly_charged else "")
    )
    print()

    for spec in DEFAULT_SAMPLES:
        time_window, all_refs = parse_report(spec.report_path)
        refs = [r for r in all_refs if r.relative >= args.relative_threshold]
        predicted = run_prediction(spec.data_path, time_window, args)
        ref_matches = match_reference_wise(refs, predicted, args.match_tolerance_da)

        refwise_matched = sum(1 for m in ref_matches if m.matched)
        unique_matched = match_unique_greedy(refs, predicted, args.match_tolerance_da)

        total_refs += len(refs)
        total_refwise_matches += refwise_matched
        total_unique_matches += unique_matched

        print(f"Sample: {spec.name}")
        print(f"  Window: {time_window[0]:.3f}-{time_window[1]:.3f} min")
        print(f"  References (rel>={args.relative_threshold:.1f}%): {len(refs)}")
        print(f"  Reference-wise matches: {refwise_matched}/{len(refs)} ({(100.0 * refwise_matched / len(refs)):.1f}%)")
        print(f"  Unique greedy matches: {unique_matched}/{len(refs)} ({(100.0 * unique_matched / len(refs)):.1f}%)")

        matched_errors = [m.abs_error for m in ref_matches if m.matched]
        if matched_errors:
            print(
                f"  Matched abs error: mean={sum(matched_errors)/len(matched_errors):.3f} Da, "
                f"max={max(matched_errors):.3f} Da"
            )
        else:
            print("  Matched abs error: n/a")

        print("  Reference -> nearest predicted")
        for m in refs:
            rm = next(x for x in ref_matches if x.component.label == m.label)
            status = "MATCH" if rm.matched else "MISS"
            sign = "+" if rm.nearest and (rm.nearest.mass - rm.component.mass) >= 0 else "-"
            if rm.nearest is None:
                delta_text = "n/a"
            else:
                delta = abs(rm.nearest.mass - rm.component.mass)
                delta_text = f"{sign}{delta:.2f}"
            print(
                f"    {rm.component.label}: ref={rm.component.mass:.2f} ({rm.component.relative:.2f}%) "
                f"-> {format_mass(rm.nearest)}  d={delta_text}  {status}"
            )

        print("  Predicted masses")
        if predicted:
            for idx, comp in enumerate(predicted, start=1):
                print(f"    {idx:>2}. {comp.mass:.4f} ({comp.source})")
        else:
            print("    none")
        print()

    if total_refs == 0:
        print("No reference components found after filtering.")
        return 1

    print("Overall")
    print(
        f"  Reference-wise: {total_refwise_matches}/{total_refs} "
        f"({(100.0 * total_refwise_matches / total_refs):.1f}%)"
    )
    print(
        f"  Unique greedy:  {total_unique_matches}/{total_refs} "
        f"({(100.0 * total_unique_matches / total_refs):.1f}%)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
