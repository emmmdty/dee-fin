from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from sarge.data.jsonl import write_jsonl
from sarge.data.loader import V2DatasetDocument, load_documents
from sarge.data.schema import DatasetSchema, load_schema
from sarge.data.staging import stage_dataset
from sarge.evaluation.export import export_predictions
from sarge.generation.diagnostics import aggregate_parse_diagnostics, generation_diagnostic_fields
from sarge.generation.parser import candidate_set_to_dict, parse_getm_output
from sarge.generation.prompt import build_getm_prompt_result, normalize_prompt_baseline_mode
from sarge.selection.ranker import default_rule_based_model
from sarge.selection.selector import select_candidate_rows
from sarge.slot_planning.baseline import TrainPriorPlanner
from sarge.slot_planning.plan import slot_plan_to_dict
from sarge.surface_memory.builder import build_surface_memory, surface_candidate_to_dict, surface_memory_to_dict
from sarge.surface_memory.candidate_builder import build_surface_memories

DEFAULT_MODEL_ENV_NAMES = ("DEEPSEEK_MODEL_FLASH", "DEEPSEEK_MODEL_PRO")
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_PROMPT_OPTIONS = {
    "max_surface_candidates": 20,
    "candidate_context_chars": 0,
    "candidate_render_mode": "compact",
    "enable_candidate_filtering": True,
    "max_candidates_per_type": 6,
    "dedupe_surface_candidates": True,
    "drop_low_value_company_fragments": True,
    "use_response_prefix": False,
    "response_prefix": "",
    "prompt_delimiter": "### RESPONSE_JSON",
    "output_format": "minimal_text",
}
RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, APIError)


@dataclass(frozen=True)
class PromptJob:
    document: V2DatasetDocument
    prompt: str
    prompt_metadata: dict[str, Any]
    selected_surface_candidates: list[dict[str, Any]]
    surface_memory: Any
    slot_plan: Any
    candidate_index: int = 0


@dataclass(frozen=True)
class ApiCallResult:
    doc_id: str
    candidate_index: int
    raw_output: str
    stopped_output: str
    metadata: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SARGE inference through DeepSeek's OpenAI-compatible API.")
    parser.add_argument("--dataset", default="DuEE-Fin-dev500")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--processed", default="data")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--limit", type=int, default=5, help="docs to run; default is a cheap smoke")
    parser.add_argument("--slot-train-limit", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--baseline-mode", default="role_safe_surface_memory")
    parser.add_argument("--model", action="append", default=[], help="DeepSeek model name; can be repeated")
    parser.add_argument(
        "--model-env",
        action="append",
        default=[],
        help="environment variable that contains a model name; can be repeated",
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url-env", default="DEEPSEEK_BASE_URL")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    load_dotenv(args.env_file, override=False)
    baseline_mode = normalize_prompt_baseline_mode(args.baseline_mode)
    models = _resolve_models(args.model, args.model_env)
    if not models:
        raise SystemExit("no DeepSeek models configured; pass --model or set DEEPSEEK_MODEL_FLASH/DEEPSEEK_MODEL_PRO")

    project_root = Path(args.project_root).resolve() if args.project_root else Path.cwd()
    api_key = _required_env(args.api_key_env)
    base_url = _env_value(args.base_url_env, DEFAULT_BASE_URL).rstrip("/")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "limit": args.limit,
                "models": models,
                "base_url": base_url,
                "api_key_set": True,
                "concurrency": args.concurrency,
                "uses_gpu": False,
                "baseline_mode": baseline_mode,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    roots = asyncio.run(_run_all_models(args, project_root, api_key, base_url, models, baseline_mode))
    print("[deepseek] completed run roots:", flush=True)
    for root in roots:
        print(root, flush=True)
    return 0


async def _run_all_models(
    args: argparse.Namespace,
    project_root: Path,
    api_key: str,
    base_url: str,
    models: list[str],
    baseline_mode: str,
) -> list[Path]:
    roots: list[Path] = []
    for model in models:
        root = await _run_one_model(
            args=args,
            project_root=project_root,
            api_key=api_key,
            base_url=base_url,
            model=model,
            baseline_mode=baseline_mode,
        )
        roots.append(root)
    return roots


async def _run_one_model(
    *,
    args: argparse.Namespace,
    project_root: Path,
    api_key: str,
    base_url: str,
    model: str,
    baseline_mode: str,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = _run_id(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        model=model,
        baseline_mode=baseline_mode,
        concurrency=args.concurrency,
        stamp=stamp,
    )
    run_root = Path(args.out) / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    diagnostics_dir = run_root / "diagnostics"
    getm_dir = run_root / "intermediate" / "getm"
    mrs_dir = run_root / "intermediate" / "mrs"
    prediction_dir = run_root / "predictions" / args.dataset
    for directory in (diagnostics_dir, getm_dir, mrs_dir, prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sarge_deepseek_") as tmp:
        staging = Path(tmp)
        stage_dataset(
            dataset=args.dataset,
            processed_root=args.processed,
            output_root=staging,
            splits=("train",),
            limit=args.slot_train_limit,
        )
        stage_dataset(
            dataset=args.dataset,
            processed_root=args.processed,
            output_root=staging,
            splits=(args.split,),
            limit=args.limit,
        )
        schema = load_schema(args.dataset, data_root=staging)
        documents = load_documents(args.dataset, args.split, data_root=staging, mode="predict", limit=args.limit)
        train_docs = load_documents(args.dataset, "train", data_root=staging, mode="train")
        prompt_jobs, surface_rows, slot_rows = _build_prompt_jobs(
            dataset=args.dataset,
            split=args.split,
            schema=schema,
            documents=documents,
            train_docs=train_docs,
            baseline_mode=baseline_mode,
        )

        write_jsonl(getm_dir / f"prompts.{args.split}.jsonl", _prompt_rows(args.dataset, args.split, prompt_jobs))
        write_jsonl(run_root / "intermediate" / "surface_memory.jsonl", surface_rows)
        write_jsonl(run_root / "intermediate" / "slot_plan.jsonl", slot_rows)

        print(
            f"[deepseek] model={model} docs={len(documents)} concurrency={args.concurrency} run_root={run_root}",
            flush=True,
        )
        t0 = time.monotonic()
        call_results = await _call_model_for_jobs(
            api_key=api_key,
            base_url=base_url,
            model=model,
            jobs=prompt_jobs,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            baseline_mode=baseline_mode,
        )
        api_seconds = time.monotonic() - t0
        parsed_rows, raw_rows = _parse_results(
            dataset=args.dataset,
            split=args.split,
            model=model,
            schema=schema,
            prompt_jobs=prompt_jobs,
            call_results=call_results,
            baseline_mode=baseline_mode,
            max_tokens=args.max_tokens,
        )
        write_jsonl(getm_dir / f"raw_outputs.{args.split}.jsonl", raw_rows)
        write_jsonl(getm_dir / f"parsed_candidates.{args.split}.jsonl", parsed_rows)
        parse_diagnostics = aggregate_parse_diagnostics(
            parsed_rows,
            dataset=args.dataset,
            split=args.split,
            k=1,
            generation_metadata=_generation_metadata(
                model=model,
                baseline_mode=baseline_mode,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            ),
        )
        _write_json(getm_dir / f"parse_diagnostics.{args.split}.json", parse_diagnostics)

        selection = select_candidate_rows(
            candidates=parsed_rows,
            documents=documents,
            schema=schema,
            model=default_rule_based_model(),
            surface_memories={row["doc_id"]: row for row in surface_rows},
            slot_plans={row["doc_id"]: row for row in slot_rows},
        )
        write_jsonl(mrs_dir / f"selector_scores.{args.split}.jsonl", selection.score_rows)
        write_jsonl(mrs_dir / f"selected_candidates.{args.split}.jsonl", selection.selected_rows)
        prediction_path = prediction_dir / f"{args.split}.canonical.pred.jsonl"
        export_predictions(selection.canonical_predictions, prediction_path, schema=schema)

    summary = _pipeline_summary(
        run_id=run_id,
        run_root=run_root,
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        model=model,
        baseline_mode=baseline_mode,
        concurrency=args.concurrency,
        api_seconds=api_seconds,
        raw_rows=raw_rows,
        parsed_rows=parsed_rows,
        prediction_path=prediction_path,
    )
    _write_json(diagnostics_dir / "pipeline_summary.json", summary)
    _write_json(
        run_root / "run_manifest.json",
        {
            "run_id": run_id,
            "dataset": args.dataset,
            "split": args.split,
            "limit": args.limit,
            "seed": args.seed,
            "backend": "deepseek_api",
            "model": model,
            "base_url": base_url,
            "api_key_env": args.api_key_env,
            "api_key_recorded": False,
            "uses_gpu": False,
            "concurrency": args.concurrency,
            "baseline_mode": baseline_mode,
            "prompt_options": _prompt_options(baseline_mode),
            "created_at_utc": stamp,
        },
    )

    if not args.skip_eval:
        _run_eval(
            run_root=run_root,
            dataset=args.dataset,
            split=args.split,
            processed_root=Path(args.processed),
            project_root=project_root,
            python=args.python,
        )
    return run_root


def _build_prompt_jobs(
    *,
    dataset: str,
    split: str,
    schema: DatasetSchema,
    documents: list[V2DatasetDocument],
    train_docs: list[V2DatasetDocument],
    baseline_mode: str,
) -> tuple[list[PromptJob], list[dict[str, Any]], list[dict[str, Any]]]:
    planner = TrainPriorPlanner.fit(schema, train_docs)
    slot_plans = {plan.doc_id: plan for plan in planner.predict(documents)}
    memories = {memory.doc_id: memory for memory in build_surface_memories(documents)}
    jobs: list[PromptJob] = []
    surface_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    for document in documents:
        memory = memories.get(document.doc_id) or build_surface_memory(document.input)
        slot_plan = slot_plans.get(document.doc_id)
        surface_rows.append(surface_memory_to_dict(memory))
        if slot_plan is not None:
            slot_rows.append(slot_plan_to_dict(slot_plan))
        prompt_result = build_getm_prompt_result(
            dataset=dataset,
            schema=schema,
            document=document.input,
            surface_candidates=list(memory.candidates),
            slot_plan=slot_plan,
            **_prompt_options(baseline_mode),
        )
        jobs.append(
            PromptJob(
                document=document,
                prompt=prompt_result.prompt,
                prompt_metadata=dict(prompt_result.prompt_metadata),
                selected_surface_candidates=list(prompt_result.selected_surface_candidates),
                surface_memory=memory,
                slot_plan=slot_plan,
            )
        )
    return jobs, surface_rows, slot_rows


async def _call_model_for_jobs(
    *,
    api_key: str,
    base_url: str,
    model: str,
    jobs: list[PromptJob],
    concurrency: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
    max_retries: int,
    baseline_mode: str,
) -> list[ApiCallResult]:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0, http_client=http_client)
    semaphore = asyncio.Semaphore(concurrency)
    try:
        tasks = [
            _call_one(
                client=client,
                semaphore=semaphore,
                model=model,
                job=job,
                max_tokens=max_tokens,
                temperature=temperature,
                max_retries=max_retries,
                baseline_mode=baseline_mode,
            )
            for job in jobs
        ]
        return list(await asyncio.gather(*tasks))
    finally:
        await client.close()


async def _call_one(
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    job: PromptJob,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    baseline_mode: str,
) -> ApiCallResult:
    doc_id = job.document.doc_id
    t0 = time.monotonic()
    last_error = ""
    async with semaphore:
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": job.prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                choice = response.choices[0] if response.choices else None
                message = choice.message if choice else None
                content = str(message.content if message else "" or "")
                reasoning_content = _message_extra_text(message, "reasoning_content")
                usage = response.usage
                metadata = _token_metadata(
                    model=model,
                    baseline_mode=baseline_mode,
                    max_tokens=max_tokens,
                    raw_output=content,
                    reasoning_content=reasoning_content,
                    latency_sec=time.monotonic() - t0,
                    retry_count=attempt,
                    finish_reason=_choice_finish_reason(choice),
                    request_id=str(response.id or ""),
                    usage=usage,
                )
                return ApiCallResult(doc_id, job.candidate_index, content, content, metadata)
            except RETRYABLE_ERRORS as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= max_retries:
                    break
                await asyncio.sleep(min(2**attempt, 20))
            except Exception as exc:  # pragma: no cover - network/provider dependent
                last_error = f"{type(exc).__name__}: {exc}"
                break
    fallback = '{"events": []}'
    metadata = _token_metadata(
        model=model,
        baseline_mode=baseline_mode,
        max_tokens=max_tokens,
        raw_output=fallback,
        reasoning_content="",
        latency_sec=time.monotonic() - t0,
        retry_count=max_retries,
        finish_reason="api_error",
        request_id="",
        usage=None,
    )
    metadata["api_error"] = last_error
    return ApiCallResult(doc_id, job.candidate_index, fallback, fallback, metadata)


def _parse_results(
    *,
    dataset: str,
    split: str,
    model: str,
    schema: DatasetSchema,
    prompt_jobs: list[PromptJob],
    call_results: list[ApiCallResult],
    baseline_mode: str,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result_by_key = {(result.doc_id, result.candidate_index): result for result in call_results}
    parsed_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for job in prompt_jobs:
        result = result_by_key[(job.document.doc_id, job.candidate_index)]
        candidate_id = f"{job.document.doc_id}:getm:{job.candidate_index}"
        generation_metadata = {
            **_generation_metadata(model=model, baseline_mode=baseline_mode, max_tokens=max_tokens, temperature=0.0),
            **job.prompt_metadata,
        }
        parsed = parse_getm_output(
            result.stopped_output,
            doc_id=job.document.doc_id,
            candidate_id=candidate_id,
            schema=schema,
            prompt=job.prompt,
            surface_candidate_count=len(job.selected_surface_candidates),
            generation_metadata=generation_metadata,
            token_metadata=result.metadata,
            output_format="minimal_text",
            response_prefix_used=False,
        )
        parsed_row = candidate_set_to_dict(parsed)
        parsed_rows.append(parsed_row)
        raw_rows.append(
            {
                "candidate_id": candidate_id,
                "doc_id": job.document.doc_id,
                "candidate_index": job.candidate_index,
                "backend": "DeepSeekApiBackend",
                "model": model,
                "raw_output": result.raw_output,
                "stopped_output": result.stopped_output,
                "stop_reason": result.metadata.get("finish_reason"),
                "balanced_stop_applied": False,
                "api_error": result.metadata.get("api_error"),
                "latency_sec": result.metadata.get("latency_sec"),
                "prompt_token_count": result.metadata.get("prompt_token_count"),
                "generated_token_count": result.metadata.get("generated_token_count"),
                "total_tokens": result.metadata.get("total_tokens"),
                **generation_diagnostic_fields(parsed_row["diagnostics"]),
            }
        )
    return parsed_rows, raw_rows


def _run_eval(
    *,
    run_root: Path,
    dataset: str,
    split: str,
    processed_root: Path,
    project_root: Path,
    python: str,
) -> None:
    cmd = [
        python,
        "-B",
        "scripts/eval_three_tracks.py",
        "--run-root",
        str(run_root),
        "--dataset",
        dataset,
        "--split",
        split,
        "--processed-root",
        str(processed_root),
        "--project-root",
        str(project_root),
        "--python",
        python,
    ]
    proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
    (run_root / "eval_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_root / "eval_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"eval_three_tracks failed rc={proc.returncode}: {run_root}")


def _pipeline_summary(
    *,
    run_id: str,
    run_root: Path,
    dataset: str,
    split: str,
    limit: int | None,
    model: str,
    baseline_mode: str,
    concurrency: int,
    api_seconds: float,
    raw_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, Any]],
    prediction_path: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_root": str(run_root),
        "dataset": dataset,
        "split": split,
        "limit": limit,
        "document_count": len(parsed_rows),
        "backend": "deepseek_api",
        "model": model,
        "uses_gpu": False,
        "baseline_mode": baseline_mode,
        "concurrency": concurrency,
        "api_seconds": round(api_seconds, 3),
        "api_error_count": sum(1 for row in raw_rows if row.get("api_error")),
        "parse_error_count": sum(1 for row in parsed_rows if (row.get("diagnostics") or {}).get("parse_error")),
        "accepted_event_count": sum(int((row.get("diagnostics") or {}).get("accepted_event_count") or 0) for row in parsed_rows),
        "prompt_tokens": sum(int(row.get("prompt_token_count") or 0) for row in raw_rows),
        "completion_tokens": sum(int(row.get("generated_token_count") or 0) for row in raw_rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in raw_rows),
        "prediction_path": str(prediction_path),
    }


def _token_metadata(
    *,
    model: str,
    baseline_mode: str,
    max_tokens: int,
    raw_output: str,
    reasoning_content: str,
    latency_sec: float,
    retry_count: int,
    finish_reason: str | None,
    request_id: str,
    usage: Any | None,
) -> dict[str, Any]:
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    metadata = {
        "api_backend": "deepseek",
        "api_model": model,
        "request_id": request_id,
        "finish_reason": finish_reason,
        "latency_sec": round(latency_sec, 3),
        "retry_count": int(retry_count),
        "prompt_token_count": prompt_tokens,
        "generated_token_count": completion_tokens,
        "total_tokens": total_tokens,
        "max_new_tokens": int(max_tokens),
        "raw_output": raw_output,
        "stopped_output": raw_output,
        "reasoning_content_char_count": len(reasoning_content),
        "response_prefix_used": False,
        "response_prefix": "",
        "output_format": "minimal_text",
        "baseline_mode": baseline_mode,
    }
    details = _usage_value(usage, "completion_tokens_details")
    if details is not None:
        metadata["completion_tokens_details"] = _json_ready(details)
    return metadata


def _generation_metadata(*, model: str, baseline_mode: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    return {
        "backend_kind": "deepseek_api",
        "api_backend": "deepseek",
        "api_model": model,
        "max_new_tokens": int(max_tokens),
        "temperature": float(temperature),
        "do_sample": False,
        "response_prefix_used": False,
        "response_prefix": "",
        "output_format": "minimal_text",
        "baseline_mode": baseline_mode,
        **_prompt_options(baseline_mode),
    }


def _prompt_rows(dataset: str, split: str, jobs: Iterable[PromptJob]) -> Iterable[dict[str, Any]]:
    for job in jobs:
        memory = job.surface_memory
        yield {
            "doc_id": job.document.doc_id,
            "dataset": dataset,
            "split": split,
            "prompt": job.prompt,
            "surface_candidates": [surface_candidate_to_dict(candidate) for candidate in memory.candidates],
            "prompt_surface_candidates": list(job.selected_surface_candidates),
            "prompt_metadata": dict(job.prompt_metadata),
        }


def _prompt_options(baseline_mode: str) -> dict[str, Any]:
    return {**DEFAULT_PROMPT_OPTIONS, "baseline_mode": baseline_mode}


def _resolve_models(model_args: list[str], model_envs: list[str]) -> list[str]:
    names = list(model_args)
    env_names = model_envs or list(DEFAULT_MODEL_ENV_NAMES)
    for env_name in env_names:
        value = _env_value(env_name, "")
        if value:
            names.append(value)
    deduped: list[str] = []
    for name in names:
        clean = str(name).strip()
        if clean and clean not in deduped:
            deduped.append(clean)
    return deduped


def _run_id(
    *,
    dataset: str,
    split: str,
    limit: int | None,
    model: str,
    baseline_mode: str,
    concurrency: int,
    stamp: str,
) -> str:
    limit_label = f"limit{limit}" if limit is not None else "full"
    return (
        f"sarge_deepseek_api_{_slug(dataset)}_{split}_{limit_label}_"
        f"{_slug(model)}_{baseline_mode}_c{concurrency}_{stamp}"
    )


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _required_env(name: str) -> str:
    value = _env_value(name, "")
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def _env_value(name: str, default: str) -> str:
    import os

    return str(os.environ.get(name) or default).strip()


def _usage_value(usage: Any | None, name: str) -> Any:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(name)
    return getattr(usage, name, None)


def _choice_finish_reason(choice: Any | None) -> str | None:
    if choice is None:
        return None
    value = getattr(choice, "finish_reason", None)
    if value is not None:
        return str(value)
    if isinstance(choice, dict):
        value = choice.get("finish_reason")
        return str(value) if value is not None else None
    extra = getattr(choice, "model_extra", None)
    if isinstance(extra, dict):
        value = extra.get("finish_reason")
        return str(value) if value is not None else None
    return None


def _message_extra_text(message: Any | None, name: str) -> str:
    if message is None:
        return ""
    value = getattr(message, name, None)
    if value:
        return str(value)
    if isinstance(message, dict):
        return str(message.get(name) or "")
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        return str(extra.get(name) or "")
    return ""


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_ready(value.model_dump())
    return str(value)


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    raise SystemExit(main())
