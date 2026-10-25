"""Interfejs wiersza poleceń: `uv run rag <komenda>` (Makefile tylko to opakowuje)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table

from rag.config import CHUNKING_STRATEGIES, get_settings

console = Console()


def cmd_ingest(args: argparse.Namespace) -> None:
    from rag.ingest import ingest

    stats = ingest(
        [Path(p) for p in (args.paths or [get_settings().data_dir])],
        get_settings(),
        strategy=args.chunking,
        collection=args.collection,
        recreate=args.recreate,
    )
    console.print(
        f"✅ Zaindeksowano {stats['chunks']} chunków z {stats['files']} plików do "
        f"kolekcji [bold]{stats['collection']}[/] w {stats['total_s']}s "
        f"(embedding: {stats['embed_s']}s)"
    )


def cmd_ask(args: argparse.Namespace) -> None:
    from rag.config import parse_filters
    from rag.generate import answer_question, best_quote, render_answer

    result = answer_question(
        args.question,
        get_settings(),
        k=args.k,
        collection=args.collection,
        filters=parse_filters(",".join(args.filter)),
    )
    console.print(Markdown(render_answer(result)))
    if result.answered_from_context:
        console.print("\n[bold]Źródła cytowane:[/]")
        for n in result.cited_numbers:
            hit = result.hits[n - 1]
            quote = best_quote(hit.text, result.answer.answer)
            console.print(f"  [{n}] {hit.label()}  [dim](score {hit.score:.3f})[/]")
            console.print(f"      [italic]„{quote}”[/]")
        if not result.has_inline_citations:
            console.print("[yellow]Uwaga: brak znaczników [n] w tekście odpowiedzi.[/]")
    else:
        console.print(
            "\n[yellow]Model nie zacytował żadnego fragmentu – kontekst nie zawiera odpowiedzi.[/]"
        )
    if retrieved_only := result.retrieved_only():
        console.print("[bold]Pobrane (retrieved), niecytowane:[/]")
        for i, hit in retrieved_only:
            console.print(f"  [dim][{i}] {hit.label()}  (score {hit.score:.3f})[/]")
    console.print(
        f"[dim]retrieval {result.retrieve_s}s · generacja {result.generate_s}s · "
        f"structured output: {result.mode}[/]"
    )
    if args.show_context:
        for i, hit in enumerate(result.hits, start=1):
            console.rule(f"[{i}] {hit.label()}")
            console.print(hit.text)


def cmd_agent(args: argparse.Namespace) -> None:
    from rag.agent import AgentStep, run_agent

    def show(step: AgentStep) -> None:
        if step.kind == "tool_call":
            console.print(f"[cyan]🔧 {step.arguments}[/]")
            console.print(f"[dim]{step.content[:300]}…[/]")
        else:
            console.print(Markdown(step.content))

    settings = get_settings()
    if args.question:
        run_agent(args.question, settings, on_step=show)
        return
    console.print("Tryb interaktywny agenta (pusta linia kończy).")
    while question := console.input("[bold]> [/]").strip():
        run_agent(question, settings, on_step=show)


# Krótkie nagłówki, żeby tabela mieściła się w wąskim terminalu (legenda pod tabelą).
SHORT_HEADERS = {
    "mrr": "MRR",
    "ctx_chars": "ctx",
    "refusal_ok": "odm.",
    "citation_hit": "cyt.",
    "inline_cited": "[n]",
    "latency_s": "t[s]",
}


def cmd_eval(args: argparse.Namespace) -> None:
    from rag.eval import evaluate, format_cell, load_golden, save_results

    settings = get_settings()
    items = load_golden(Path(args.golden))
    results = evaluate(
        items, settings, generate=args.generate, collection=args.collection, level=args.level
    )
    summary = results["summary"]
    # Kolumny = metryki policzone w tym przebiegu (bez --generate nie ma metryk odpowiedzi).
    metric_cols = [c for c in summary if c != "questions" and summary[c] is not None]
    table = Table(
        title=f"Eval: {args.golden} · {settings.llm_model} · {args.collection or settings.collection}"
    )
    table.add_column("id", no_wrap=True, min_width=7)
    # Kolumna z pytaniem dostaje to, co zostaje po metrykach (pełne pytania są w pliku .md).
    question_width = max(10, min(50, console.width - 10 - 8 * len(metric_cols)))
    table.add_column("pytanie", width=question_width, no_wrap=True, overflow="ellipsis")
    for col in metric_cols:
        header = SHORT_HEADERS.get(col, col.replace("recall@", "R@"))
        # escape: „[n]” i „t[s]” rich wziąłby za znaczniki formatowania.
        table.add_column(
            escape(header), justify="right", no_wrap=True, min_width=max(4, len(header))
        )
    for row in results["rows"]:
        table.add_row(row["id"], row["question"], *(format_cell(row.get(c)) for c in metric_cols))
    table.add_section()
    table.add_row(
        "ŚREDNIA",
        f"{summary['questions']} pytań",
        *(format_cell(summary.get(c)) for c in metric_cols),
    )
    console.print(table)
    legend = "R@k = recall@k · MRR = mean reciprocal rank · ctx = znaki kontekstu w top-k"
    if args.generate:
        legend += (
            " · odm. = poprawna odmowa/odpowiedź · cyt. = cytat z oczekiwanego źródła"
            " · [n] = znaczniki źródeł w tekście"
        )
    console.print(f"[dim]{escape(legend)}[/]")

    label = f"{Path(args.golden).stem}_{args.collection or settings.collection}"
    if args.level != "source":
        label += f"_{args.level}"
    if args.generate:
        label += f"_{settings.llm_model.replace(':', '-').replace('/', '-')}"
    results |= {
        "golden": args.golden,
        "collection": args.collection or settings.collection,
        "level": args.level,
        "generate": args.generate,
    }
    path = save_results(results, settings, Path("eval/results"), label)
    console.print(f"[dim]Zapisano: {path}[/]")


def cmd_check(_: argparse.Namespace) -> None:
    from rag.check import run_checks

    if not run_checks(get_settings()):
        sys.exit(1)


def cmd_reset_cache(_: argparse.Namespace) -> None:
    from rag.structured import CACHE_PATH, reset_cache

    if reset_cache():
        console.print(f"Usunięto {CACHE_PATH} – tryb structured output zostanie wykryty od nowa.")
    else:
        console.print(f"Brak cache ({CACHE_PATH}).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="Starter RAG – kurs IDM (WEiTI PW)")
    parser.add_argument("-v", "--verbose", action="store_true", help="więcej logów")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="zaindeksuj dokumenty")
    p.add_argument("paths", nargs="*", help="pliki lub katalogi (domyślnie DATA_DIR z .env)")
    p.add_argument("--chunking", choices=CHUNKING_STRATEGIES, help="strategia chunkingu")
    p.add_argument("--collection", help="nazwa kolekcji (domyślnie COLLECTION z .env)")
    p.add_argument("--recreate", action="store_true", help="usuń i utwórz kolekcję od nowa")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="zadaj pytanie (RAG)")
    p.add_argument("question")
    p.add_argument("-k", type=int, help="liczba chunków w kontekście (domyślnie TOP_K)")
    p.add_argument("--collection")
    p.add_argument("--show-context", action="store_true", help="pokaż pełny tekst chunków")
    p.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="POLE=WARTOŚĆ",
        help="filtr metadanych, np. --filter doc_type=regulamin (można powtarzać)",
    )
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("agent", help="agent z narzędziem search_docs")
    p.add_argument("question", nargs="?")
    p.set_defaults(func=cmd_agent)

    p = sub.add_parser("eval", help="ewaluacja na złotym zbiorze")
    p.add_argument("--golden", default="eval/golden_sample.jsonl")
    p.add_argument("--generate", action="store_true", help="oceniaj też odpowiedzi (wolniej)")
    p.add_argument(
        "--level",
        choices=("source", "paragraph"),
        default="source",
        help="granulacja source_ids: jak w złotym zbiorze albo na poziomie § (paragraph)",
    )
    p.add_argument("--collection")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("check", help="smoke test środowiska")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("reset-cache", help="usuń cache wykrytego trybu structured output")
    p.set_defaults(func=cmd_reset_cache)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(show_path=False, show_time=False)],
    )
    # Wykrycie trybu structured output pokazujemy zawsze (jedna linia na INFO) – to ważna
    # informacja dydaktyczna. Szczegóły każdej próby: `rag -v ...` (DEBUG).
    logging.getLogger("rag.structured").setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        args.func(args)
    except (RuntimeError, ValueError) as err:
        console.print(f"[red]Błąd:[/] {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
