"""Pobiera dokumenty korpusu wskazane w data/raw/manifest.json (`make fetch-data`).

PDF-y nie są w gicie (licencje, rozmiar) – w repo jest tylko manifest z adresami URL.

Użycie: uv run python scripts/fetch_data.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import httpx

from rag.ingest import load_manifest

DATA_DIR = Path("data/raw")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="pobierz ponownie istniejące pliki")
    args = parser.parse_args()

    for doc in load_manifest(DATA_DIR).values():
        path = DATA_DIR / doc["file"]
        if path.exists() and not args.force:
            print(f"[ok]       {doc['file']} (już pobrany)")
        else:
            print(f"[pobieram] {doc['file']} ← {doc['url']}")
            response = httpx.get(doc["url"], follow_redirects=True, timeout=60)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise SystemExit(
                    f"{doc['file']}: odpowiedź nie jest PDF-em – sprawdź URL w manifeście"
                )
            path.write_bytes(response.content)
        expected = doc.get("sha256")
        actual = sha256(path)
        if expected and expected != actual:
            print("  ⚠ suma SHA-256 różni się od manifestu – dokument na BIP mógł zostać zmieniony")
        print(f"  sha256 {actual}  ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
