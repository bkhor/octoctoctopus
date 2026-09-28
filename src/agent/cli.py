import argparse
import json
import sys
from pathlib import Path

from . import db, memory, orchestrator
from .config import load_config
from .embedding_client import BgeEmbeddingClient
from .llm_client import OllamaClient
from .models import ApplicationState
from .profile import load_profile


def cmd_submit(conn, url: str) -> int:
    app_id = db.create_application(conn, url)
    print(app_id)
    return app_id


def cmd_status(conn) -> None:
    for state in ApplicationState:
        for row in db.list_applications_by_state(conn, state.value):
            print(f"{row['id']}\t{row['url']}\t{row['state']}")


def cmd_review(conn, app_id: int) -> None:
    app = db.get_application(conn, app_id)
    print(f"url: {app['url']}")
    print(f"state: {app['state']}")
    for row in db.list_fields(conn, app_id):
        marker = " (needs input)" if row["needs_input"] else ""
        print(f"  {row['label']}: {row['resolved_value']}{marker}")
        options_raw = dict(row).get("options")
        if row["needs_input"] and row["field_type"] in ("select", "radio") and options_raw:
            for i, opt in enumerate(json.loads(options_raw), start=1):
                print(f"    {i}. {opt}")
    if app["cover_letter_draft"]:
        print("cover letter:")
        print(app["cover_letter_draft"])


NON_TERMINAL_STATES = (
    ApplicationState.QUEUED.value,
    ApplicationState.FETCHING.value,
    ApplicationState.CLASSIFYING.value,
    ApplicationState.RESOLVING_FORMAL.value,
    ApplicationState.RESOLVING_INFORMAL.value,
    ApplicationState.DRAFTING_COVER_LETTER.value,
    ApplicationState.FILLING.value,
)

STOPPING_STATES = (
    ApplicationState.AWAITING_HITL.value,
    ApplicationState.READY_FOR_REVIEW.value,
    ApplicationState.MANUAL_FALLBACK.value,
    ApplicationState.DONE.value,
)


def cmd_run(conn, llm, embedder, fetcher, filler, profile: dict) -> int:
    app_ids = []
    for state in NON_TERMINAL_STATES:
        for row in db.list_applications_by_state(conn, state):
            app_ids.append(row["id"])
    failures = 0
    for app_id in app_ids:
        try:
            state = db.get_application(conn, app_id)["state"]
            while state not in STOPPING_STATES:
                state = orchestrator.step(
                    conn, app_id, llm=llm, embedder=embedder,
                    fetcher=fetcher, filler=filler, profile=profile,
                )
            app = db.get_application(conn, app_id)
            print(f"{app_id}\t{app['url']}\t{state}")
        except Exception as exc:
            failures += 1
            print(f"{app_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return failures


def cmd_hitl(conn, llm, embedder, field_id: int, category: str, answer: str) -> None:
    field_row = conn.execute(
        "SELECT application_id FROM application_fields WHERE id = ?", (field_id,)
    ).fetchone()
    if field_row is None:
        raise KeyError(f"no field with id {field_id}")
    orchestrator.resume_after_hitl(
        conn, field_row["application_id"], llm=llm, embedder=embedder,
        field_id=field_id, category=category, answer_text=answer,
    )


def cmd_approve(conn, app_id: int, filler) -> str:
    return orchestrator.approve(conn, app_id, filler=filler)


def cmd_remove(conn, app_id: int) -> None:
    db.delete_application(conn, app_id)


def cmd_rerun(conn, app_id: int) -> None:
    db.get_application(conn, app_id)
    db.delete_fields_for_application(conn, app_id)
    db.reset_application(conn, app_id)


IGNORE_KEYWORDS = ("skip", "n/a", "na", "ignore")


def cmd_hitl_shell(conn, llm, embedder, app_id: int, input_fn=input) -> None:
    fields = db.list_fields(conn, app_id)
    pending_file_fields = [row for row in fields if row["needs_input"] and row["field_type"] == "file"]
    answerable = [row for row in fields if row["needs_input"] and row["field_type"] != "file"]
    answered = 0
    skipped = 0
    ignored = 0
    for row in answerable:
        print(row["label"])
        options_raw = dict(row).get("options")
        options = json.loads(options_raw) if options_raw else []
        is_select = row["field_type"] in ("select", "radio") and options
        if is_select:
            for i, opt in enumerate(options, start=1):
                print(f"  {i}. {opt}")
        raw = input_fn("> ").strip()
        if not raw:
            skipped += 1
            continue
        if raw.lower() in IGNORE_KEYWORDS:
            db.update_field(conn, row["id"], resolved_value="", needs_input=False, resolved_from="ignored")
            ignored += 1
            continue
        if is_select:
            try:
                choice = int(raw)
            except ValueError:
                choice = 0
            if choice < 1 or choice > len(options):
                print("invalid choice, skipped")
                skipped += 1
                continue
            answer = options[choice - 1]
        else:
            answer = raw
        category = memory.categorize(llm, row["label"])
        orchestrator.resume_after_hitl(
            conn, app_id, llm=llm, embedder=embedder,
            field_id=row["id"], category=category, answer_text=answer,
        )
        answered += 1
    print(f"{answered} answered, {skipped} skipped, {ignored} ignored")
    for row in pending_file_fields:
        print(f"file field still needs input, add to profile.yaml: {row['label']}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="octo")
    parser.add_argument("--config", default="./config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("url")

    subparsers.add_parser("run")
    subparsers.add_parser("status")

    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("app_id", type=int)

    hitl_parser = subparsers.add_parser("hitl")
    hitl_parser.add_argument("field_id", type=int)
    hitl_parser.add_argument("--category", required=True)
    hitl_parser.add_argument("--answer", required=True)

    approve_parser = subparsers.add_parser("approve")
    approve_parser.add_argument("app_id", type=int)

    remove_parser = subparsers.add_parser("remove")
    remove_parser.add_argument("app_id", type=int)

    rerun_parser = subparsers.add_parser("rerun")
    rerun_parser.add_argument("app_id", type=int)

    hitl_shell_parser = subparsers.add_parser("hitl-shell")
    hitl_shell_parser.add_argument("app_id", type=int)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.config != "./config.yaml" and not Path(args.config).exists():
        raise FileNotFoundError(f"config file not found: {args.config}")
    settings = load_config(args.config)
    conn = db.connect(settings["db_path"])
    db.init_schema(conn)

    if args.command == "submit":
        cmd_submit(conn, args.url)
    elif args.command == "status":
        cmd_status(conn)
    elif args.command == "review":
        cmd_review(conn, args.app_id)
    elif args.command == "run":
        from playwright.sync_api import sync_playwright

        from .browser import PlaywrightFormFiller, PlaywrightPageFetcher

        profile = load_profile(settings["profile_path"])
        llm = OllamaClient(base_url=settings["ollama_base_url"], model=settings["ollama_model"])
        embedder = BgeEmbeddingClient(model_name=settings["bge_model"])
        pw = sync_playwright().start()
        try:
            browser = pw.chromium.launch()
            try:
                fetcher = PlaywrightPageFetcher(browser)
                filler = PlaywrightFormFiller(browser)
                failures = cmd_run(conn, llm, embedder, fetcher, filler, profile)
            finally:
                browser.close()
        finally:
            pw.stop()
        if failures:
            sys.exit(1)
    elif args.command == "hitl":
        llm = OllamaClient(base_url=settings["ollama_base_url"], model=settings["ollama_model"])
        embedder = BgeEmbeddingClient(model_name=settings["bge_model"])
        cmd_hitl(conn, llm, embedder, args.field_id, args.category, args.answer)
    elif args.command == "remove":
        cmd_remove(conn, args.app_id)
    elif args.command == "rerun":
        cmd_rerun(conn, args.app_id)
    elif args.command == "hitl-shell":
        llm = OllamaClient(base_url=settings["ollama_base_url"], model=settings["ollama_model"])
        embedder = BgeEmbeddingClient(model_name=settings["bge_model"])
        cmd_hitl_shell(conn, llm, embedder, args.app_id)
    elif args.command == "approve":
        from playwright.sync_api import sync_playwright

        from .browser import PlaywrightFormFiller

        pw = sync_playwright().start()
        try:
            browser = pw.chromium.launch()
            try:
                filler = PlaywrightFormFiller(browser)
                final_state = cmd_approve(conn, args.app_id, filler)
                print(final_state)
            finally:
                browser.close()
        finally:
            pw.stop()


if __name__ == "__main__":
    main()
