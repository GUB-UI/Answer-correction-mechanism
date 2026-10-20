from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from src.adapters import (
    ManualCorrectionOCRAdapter,
    create_diagnosis_adapter,
    create_ocr_adapter,
)
from src.adapters.base import DiagnosisRunContext, OCRRunContext
from src.config import load_config
from src.models import AnswerRecord, ProblemRecord, utc_now_iso
from src.storage import Storage, validate_anonymized_id


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cmd_smoke(_: argparse.Namespace) -> int:
    root = _project_root()
    config = load_config(root)
    storage = Storage(config)

    print("=== config ===")
    print(f"LLM URL: {config.lmstudio.base_url}")
    print(f"data dir: {config.data_dir}")
    print(f"service mode: {config.services.mode}")
    print(f"OCR provider: {config.ocr.provider} ({config.ocr.model_name})")
    print(
        f"Diagnosis provider: {config.diagnosis.provider} "
        f"({config.diagnosis.model_name})"
    )

    problem = ProblemRecord(
        problem_id=storage.generate_id("prob"),
        title="二次方程式",
        problem_text="x^2 + 5x + 6 = 0 を解きなさい。",
        correct_answer="x = -2, -3",
        rubric="因数分解または解の公式で正解を導ければ満点。",
        unit="二次関数",
        difficulty="basic",
        created_at=utc_now_iso(),
    )
    storage.save_problem(problem)
    print(f"saved problem: {problem.problem_id}")

    image = Image.new("RGB", (200, 120), color="white")
    image_path = config.images_dir / f"{storage.generate_id('img')}.png"
    image.save(image_path)
    answer = AnswerRecord(
        answer_id=storage.generate_id("ans"),
        problem_id=problem.problem_id,
        image_path=config.to_stored_path(image_path),
        student_anonymized_id=validate_anonymized_id("student_001"),
        created_at=utc_now_iso(),
    )
    storage.save_answer(answer)
    print(f"saved answer: {answer.answer_id}")

    ocr_adapter = create_ocr_adapter("mock", config)
    ocr_result = ocr_adapter.run(
        OCRRunContext(
            answer_id=answer.answer_id,
            image_path=config.resolve_path(answer.image_path),
            ocr_engine=config.ocr.model_name,
        )
    )
    ocr_result.ocr_id = storage.generate_id("ocr")
    storage.save_ocr_result(ocr_result)
    print(f"saved OCR: {ocr_result.ocr_id}")
    print(f"  raw_text: {ocr_result.raw_text!r}")
    print(f"  used_text: {ocr_result.used_text!r}")

    manual = ManualCorrectionOCRAdapter(storage)
    corrected = manual.apply_correction(
        ocr_result.ocr_id,
        "x^2 - 5x + 6 = 0\n(x - 2)(x - 3) = 0\nx = 2, 3",
        ocr_suspect=True,
        correction_note="OCRが -5x を +5x と誤読していた",
    )
    print(f"corrected OCR used_text: {corrected.used_text!r}")

    diagnosis_adapter = create_diagnosis_adapter("mock", config)
    diagnosis_context = DiagnosisRunContext(
        answer_id=answer.answer_id,
        problem_id=problem.problem_id,
        ocr_id=corrected.ocr_id,
        problem=problem,
        used_text=corrected.used_text,
        model_name=config.diagnosis.model_name,
        prompt_type="basic_scoring",
    )
    diagnosis_raw = diagnosis_adapter.run(diagnosis_context)
    diagnosis = diagnosis_adapter.to_diagnosis_result(
        diagnosis_context,
        diagnosis_raw,
        diagnosis_id=storage.generate_id("diag"),
        created_at=utc_now_iso(),
    )
    storage.save_diagnosis_result(diagnosis)
    print(f"saved diagnosis: {diagnosis.diagnosis_id}")
    print(f"  possible_ocr_issue: {diagnosis.possible_ocr_issue}")

    reloaded_problems = storage.list_problems()
    reloaded_answers = storage.list_answers()
    reloaded_ocr = storage.list_ocr_results()
    reloaded_diagnosis = storage.list_diagnosis_results()
    print("=== reload counts ===")
    print(
        json.dumps(
            {
                "problems": len(reloaded_problems),
                "answers": len(reloaded_answers),
                "ocr_results": len(reloaded_ocr),
                "diagnosis_results": len(reloaded_diagnosis),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def cmd_gpu_profile(args: argparse.Namespace) -> int:
    from src.runtime.envfile import update_env_file
    from src.runtime.gpu import detect_profile

    gpu, profile = detect_profile()
    _print_json({"gpu": None if gpu is None else gpu.__dict__, "profile": profile.as_dict()})
    for note in profile.notes:
        print(f"* {note}")
    if args.write_env:
        env_path = Path(args.write_env)
        update_env_file(env_path, profile.env())
        print(f"wrote {len(profile.env())} keys to {env_path}")
    return 0


def cmd_services(args: argparse.Namespace) -> int:
    from src.runtime.services import get_service_manager

    config = load_config(_project_root())
    manager = get_service_manager(config)
    if args.action == "status":
        _print_json({"mode": manager.mode, "services": manager.status()})
    elif args.action == "start":
        if not manager.managed:
            print("SERVICE_MODE=managed のときだけ起動できます", file=sys.stderr)
            return 2
        manager.ensure(args.name)
        _print_json(manager.status())
    elif args.action == "stop":
        stopped = [args.name] if args.name and manager.stop(args.name) else []
        if not args.name:
            stopped = manager.stop_all()
        _print_json({"stopped": stopped})
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from src.batch_pipeline import BatchOptions, run_batch

    config = load_config(_project_root())
    storage = Storage(config)
    answers = storage.list_answers()
    if args.answers:
        wanted = set(args.answers)
        answers = [a for a in answers if a.answer_id in wanted]
    if args.problem_id:
        answers = [a for a in answers if a.problem_id == args.problem_id]
    if not answers:
        print("対象の答案がありません", file=sys.stderr)
        return 2
    options = BatchOptions(
        ocr_provider=args.ocr_provider or config.ocr.provider,
        diagnosis_provider=args.diagnosis_provider or config.diagnosis.provider,
        prompt_type=args.prompt_type,
        reuse_existing_ocr=not args.rerun_ocr,
        skip_existing_diagnosis=not args.rerun_diagnosis,
        stop_services_after=args.stop_after,
    )

    def progress(phase: str, index: int, total: int, answer_id: str) -> None:
        print(f"[{phase}] {index}/{total} {answer_id}", flush=True)

    report = run_batch(storage, config, answers, options, progress=progress)
    _print_json(report.as_dict())
    return 0 if all(not item.errors for item in report.items) else 1


def cmd_import_teacher(args: argparse.Namespace) -> int:
    from src.teacher_compare import derive_score_allocation, parse_teacher_csv

    config = load_config(_project_root())
    storage = Storage(config)
    path = Path(args.csv)
    grades, issues = parse_teacher_csv(
        path.read_bytes(),
        storage.list_problems(),
        source_file=path.name,
        default_problem_id=args.default_problem_id,
        id_factory=lambda: storage.generate_id("tg"),
    )
    for issue in issues:
        print(f"row {issue.row_number}: {issue.message}", file=sys.stderr)
    if not grades:
        return 2
    storage.save_teacher_grades(grades)
    print(f"imported {len(grades)} teacher grades")
    if args.apply_allocation:
        for problem_id, allocation in derive_score_allocation(grades).items():
            problem = storage.get_problem(problem_id)
            if problem is not None and not problem.score_allocation:
                storage.update_problem(problem.model_copy(update={"score_allocation": allocation}))
                print(f"set score_allocation for {problem_id}: {allocation}")
    return 0 if not issues else 1


def cmd_compare(args: argparse.Namespace) -> int:
    import csv

    from src.teacher_compare import (
        build_comparisons,
        compute_metrics,
        log_discrepancies,
        metrics_by_question,
        run_recheck,
    )

    config = load_config(_project_root())
    storage = Storage(config)
    ratio = args.ratio_threshold if args.ratio_threshold is not None else config.comparison.ratio_threshold
    abs_threshold = (
        args.abs_threshold if args.abs_threshold is not None else config.comparison.abs_threshold
    )
    rows = build_comparisons(
        storage.list_teacher_grades(),
        storage.list_answers(),
        storage.list_diagnosis_results(),
        storage.list_ocr_results(),
        model_name=args.model_name,
    )
    metrics = compute_metrics(rows, ratio, abs_threshold)
    discrepancies = log_discrepancies(
        storage, rows, ratio_threshold=ratio, abs_threshold=abs_threshold
    )
    payload: dict[str, object] = {
        "metrics": metrics.as_dict(),
        "by_question": {k: v.as_dict() for k, v in metrics_by_question(rows, ratio, abs_threshold).items()},
        "discrepancies": [d.model_dump() for d in discrepancies],
    }
    if args.recheck and discrepancies:
        from src.adapters import create_diagnosis_adapter

        provider = args.diagnosis_provider or config.diagnosis.provider
        adapter = create_diagnosis_adapter(provider, config)
        mode = args.mode or config.comparison.recheck_mode
        model_name = config.diagnosis.model_name if provider != "mock" else "mock"
        rechecked = []
        for discrepancy in discrepancies:
            if discrepancy.status != "open" and not args.force_recheck:
                continue
            print(f"[recheck] {discrepancy.comparison_key}", flush=True)
            rechecked.append(
                run_recheck(
                    storage,
                    adapter,
                    discrepancy,
                    model_name=model_name,
                    mode=mode,
                    ratio_threshold=ratio,
                    abs_threshold=abs_threshold,
                ).model_dump()
            )
        payload["rechecked"] = rechecked
    _print_json(payload)
    if args.out:
        out = Path(args.out)
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            records = [row.as_dict() for row in rows]
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["comparison_key"])
            writer.writeheader()
            writer.writerows(records)
        print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Math answer diagnosis CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("smoke", help="Run end-to-end mock smoke test")

    gpu = subparsers.add_parser("gpu-profile", help="GPU を検出し推奨設定を表示（.env へ書き込み可）")
    gpu.add_argument("--write-env", default=None, help="書き込む .env のパス")

    services = subparsers.add_parser("services", help="GPU モデルサーバーの状態確認・起動・停止")
    services.add_argument("action", choices=["status", "start", "stop"])
    services.add_argument("name", nargs="?", choices=["glm_ocr", "unimernet", "llm"])

    batch = subparsers.add_parser("batch", help="OCR → UniMERNet → 診断LLM を一括で逐次実行")
    batch.add_argument("--answers", nargs="*", default=None)
    batch.add_argument("--problem-id", default=None)
    batch.add_argument("--ocr-provider", default=None)
    batch.add_argument("--diagnosis-provider", default=None)
    batch.add_argument("--prompt-type", default="step_by_step_check")
    batch.add_argument("--rerun-ocr", action="store_true")
    batch.add_argument("--rerun-diagnosis", action="store_true")
    batch.add_argument("--stop-after", action="store_true", help="終了後に全サーバーを停止")

    imp = subparsers.add_parser("import-teacher", help="教師採点 CSV を取り込む")
    imp.add_argument("csv")
    imp.add_argument("--default-problem-id", default=None)
    imp.add_argument("--apply-allocation", action="store_true", help="配点列を問題の配点として設定")

    compare = subparsers.add_parser("compare", help="教師採点とシステム採点を比較")
    compare.add_argument("--ratio-threshold", type=float, default=None)
    compare.add_argument("--abs-threshold", type=float, default=None)
    compare.add_argument("--model-name", default=None, help="比較する診断モデル名で絞り込む")
    compare.add_argument("--recheck", action="store_true", help="差の大きい項目を自動再チェック")
    compare.add_argument("--force-recheck", action="store_true", help="再チェック済みの項目も再実行")
    compare.add_argument("--mode", choices=["blind", "informed"], default=None)
    compare.add_argument("--diagnosis-provider", default=None)
    compare.add_argument("--out", default=None, help="比較表 CSV の出力先")

    args = parser.parse_args(argv)
    handlers = {
        "smoke": cmd_smoke,
        "gpu-profile": cmd_gpu_profile,
        "services": cmd_services,
        "batch": cmd_batch,
        "import-teacher": cmd_import_teacher,
        "compare": cmd_compare,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
