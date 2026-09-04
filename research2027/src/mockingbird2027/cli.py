"""Command-line entry point for the Research2027 scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .data.manifest import manifest_report, write_json, write_parquet
from .data.ravdess import build_ravdess_manifest
from .data.splits import build_ravdess_splits, split_report


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="mockingbird2027",
        description="Research2027 speech representation research scaffold.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("smoke", help="run the synthetic scaffold smoke check")

    manifest_parser = subparsers.add_parser("manifest", help="build a validated dataset manifest")
    manifest_subparsers = manifest_parser.add_subparsers(dest="dataset", required=True)
    ravdess_parser = manifest_subparsers.add_parser("ravdess", help="build the RAVDESS manifest")
    ravdess_parser.add_argument("root", type=Path, help="root containing official RAVDESS WAV files")
    ravdess_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts"), help="artifact output directory"
    )
    ravdess_parser.add_argument("--speaker-folds", type=int, default=6)
    ravdess_parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="permit a development subset instead of the complete 1,440-file corpus",
    )

    extract_parser = subparsers.add_parser("extract", help="extract frozen encoder representations")
    extract_subparsers = extract_parser.add_subparsers(dest="model", required=True)
    wavlm_parser = extract_subparsers.add_parser(
        "wavlm", help="extract microsoft/wavlm-base-plus hidden states"
    )
    wavlm_parser.add_argument("manifest", type=Path, help="validated RAVDESS Parquet manifest")
    wavlm_parser.add_argument(
        "--cache-root", type=Path, default=Path("artifacts/embeddings")
    )
    wavlm_parser.add_argument("--revision", default="main")
    wavlm_parser.add_argument("--device", default="auto")
    wavlm_parser.add_argument("--batch-size", type=int, default=4)
    wavlm_parser.add_argument("--smoke-size", type=int, default=16)
    wavlm_parser.add_argument(
        "--smoke-only", action="store_true", help="validate only the smoke cache"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        print("Research2027 scaffold OK")
    elif args.command == "manifest" and args.dataset == "ravdess":
        rows = build_ravdess_manifest(args.root, require_complete=not args.allow_incomplete)
        assignments = build_ravdess_splits(rows, speaker_folds=args.speaker_folds)

        manifest_path = args.output_dir / "manifests" / "ravdess.parquet"
        splits_path = args.output_dir / "manifests" / "ravdess_splits.parquet"
        report_path = args.output_dir / "reports" / "ravdess_manifest_report.json"
        write_parquet(rows, manifest_path)
        write_parquet(assignments, splits_path)
        report = manifest_report(rows)
        report["splits"] = split_report(assignments)
        write_json(report, report_path)
        print(f"Wrote {len(rows)} validated utterances to {manifest_path}")
        print(f"Wrote split metadata to {splits_path}")
        print(f"Wrote manifest report to {report_path}")
    elif args.command == "extract" and args.model == "wavlm":
        from .models.extract import run_m2_extraction

        smoke_result, full_result = run_m2_extraction(
            args.manifest,
            args.cache_root,
            smoke_size=args.smoke_size,
            smoke_only=args.smoke_only,
            revision=args.revision,
            device=args.device,
            batch_size=args.batch_size,
        )
        if smoke_result is None:
            print("Smoke extraction skipped because a valid full cache already exists.")
        else:
            print(
                f"Smoke cache validated: {smoke_result.path} "
                f"({smoke_result.layer_count} layers, "
                f"{smoke_result.utterance_count}x{smoke_result.hidden_size})"
            )
        if full_result is not None:
            print(
                f"Full cache validated: {full_result.path} "
                f"({full_result.layer_count} layers, "
                f"{full_result.utterance_count}x{full_result.hidden_size})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

