# idm-rag-starter

Starter repo do projektów z kursu **Inżynieria dużych modeli językowych** (WEiTI PW).
Minimalny, czytelny pipeline RAG, który uruchomisz lokalnie na laptopie:

```
 ingest (offline)                                        query (online)
 ─────────────────                                       ───────────────
 PDF/MD/TXT → normalizacja → chunking → embedding ─┐     pytanie → embedding → retrieval top-k
                                                   ▼                                  │
                                              Qdrant (wektory + metadane) ◄───────────┘
                                                                                      ▼
                                       odpowiedź z cytowaniem źródeł ◄── LLM (structured output)
```

- LLM i embeddingi przez SDK `openai` z `base_url` z `.env` – lokalna **Ollama** albo
  **Gemini API** (zmiana tylko w `.env`). Bez LangChaina/LlamaIndexa: cały przepływ jest
  w kilkunastu krótkich modułach w `src/rag/`.
- Wektory w **Qdrancie** (Docker) albo w trybie lokalnym bez Dockera.
- Agent z tool callingiem (pętla agenta napisana jawnie), ewaluacja retrievalu (recall@k, MRR),
  smoke test środowiska (`make check`).
- Miejsca **TODO** na kolejne techniki: hybrid search (BM25), reranking, metadata filtering,
  query rewriting – z opisem, co i jak zrobić.

## Spis treści

1. [Wymagania sprzętowe](#wymagania-sprzętowe)
2. [Instalacja](#instalacja)
3. [Okno kontekstu Ollamy – koniecznie przeczytaj](#okno-kontekstu-ollamy--koniecznie-przeczytaj)
4. [Szybki start](#szybki-start)
5. [Przykład: Regulamin studiów PW](#przykład-regulamin-studiów-pw-gałąź-exampleregulamin)
6. [Konfiguracja (.env)](#konfiguracja-env)
7. [Structured output: trzy tryby](#structured-output-trzy-tryby)
8. [Słabszy laptop: Gemini API](#słabszy-laptop-gemini-api)
9. [Ewaluacja](#ewaluacja)
10. [Co dalej (TODO)](#co-dalej-todo)
11. [Struktura repozytorium](#struktura-repozytorium)
12. [Troubleshooting](#troubleshooting)

## Wymagania sprzętowe

Wybierz profil pasujący do Twojego laptopa i przepisz jego ustawienia do `.env`
(`OLLAMA_CONTEXT_LENGTH` ustawiasz w systemie – patrz [okno kontekstu](#okno-kontekstu-ollamy--koniecznie-przeczytaj)).

| Profil | RAM | `LLM_MODEL` | `OLLAMA_CONTEXT_LENGTH` | `TOP_K` / `CHUNK_SIZE` | Uwagi |
|--------|-----|-------------|-------------------------|------------------------|-------|
| **S** | 8 GB | `qwen3.5:4b` (awaryjnie `qwen3.5:2b`) | 8192 | 4 / 600 | zamknij przeglądarkę i IDE z wtyczkami AI; Qdrant w trybie `local:` oszczędza ~0,5 GB |
| **M** | 16 GB | `qwen3.5:9b` | 16384 | 5 / 800 | wyraźnie lepsze odpowiedzi, ok. 1,5–2× wolniej niż 4b |
| **L** | 24 GB+ | `qwen3.5:9b` (porównania z 4b) | 16384 | 5 / 800 | oba modele mieszczą się naraz w pamięci – wygodne do eksperymentów |
| **API** | dowolny | Gemini | – | 5 / 800 | gdy laptop nie uciągnie modelu – patrz [niżej](#słabszy-laptop-gemini-api) |

Pamięć (okno 16384, `ollama ps`): `qwen3.5:4b` ≈ 4 GB, `qwen3.5:9b` ≈ 7,6 GB, embeddingi ≈ 0,6 GB.
Dysk: 5 GB (4b) lub 9 GB (9b) na modele i ~1 GB na zależności Pythona.

## Instalacja

Potrzebujesz trzech rzeczy: **uv** (Python i zależności), **Ollama** (modele lokalnie)
i opcjonalnie **Docker** (Qdrant). Nic nie instalujemy globalnie poza tymi narzędziami –
`uv` sam pobierze Pythona 3.12 do swojego katalogu i utworzy `.venv` w repozytorium.

### macOS

```bash
brew install uv ollama
brew services start ollama          # albo aplikacja Ollama.app
# Docker: OrbStack (lekki, polecany) lub Docker Desktop
brew install --cask orbstack
```

Na Macu Ollama musi działać **natywnie**, nie w Dockerze – kontener nie ma dostępu do GPU (Metal),
więc model działałby kilka-kilkanaście razy wolniej.

### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://ollama.com/install.sh | sh     # instaluje też usługę systemd
# Docker: pakiet docker.io / docker-ce + plugin compose, użytkownik w grupie docker
```

### Windows

```powershell
winget install --id=astral-sh.uv -e
winget install --id=Ollama.Ollama -e
winget install --id=Docker.DockerDesktop -e     # opcjonalnie
```

`make` na Windowsie zwykle nie jest dostępny – zamiast `make ask Q="..."` wpisz
`uv run rag ask "..."` (każda komenda w `Makefile` to tylko taki wrapper; `make help` pokazuje listę).

### Bez Dockera

Ustaw w `.env`: `QDRANT_URL=local:./.qdrant`. Qdrant działa wtedy jako biblioteka w procesie
Pythona i trzyma dane w katalogu `.qdrant/`. Wystarczy do projektu, ale: tylko jeden proces naraz
(nie uruchamiaj równolegle `ingest` i `ask`), brak dashboardu, wolniej przy dużych korpusach.

## Okno kontekstu Ollamy – koniecznie przeczytaj

**Okno kontekstu** to liczba tokenów, które model widzi naraz (instrukcje, chunki, pytanie,
odpowiedź). Ollama domyślnie ustawia małe okno (często 2048–4096) i dłuższy prompt **obcina po
cichu, bez błędu** – model nie widzi części chunków, a Ty szukasz błędu w retrievalu, którego nie ma.
Pięć chunków po 800 znaków to już ok. 2000 tokenów, dlatego ustawiamy okno jawnie:

| System | Jak ustawić (potem zrestartuj Ollamę) |
|--------|---------------------------------------|
| macOS (aplikacja) | `launchctl setenv OLLAMA_CONTEXT_LENGTH 16384` |
| macOS (brew)      | `brew services stop ollama`, potem `OLLAMA_CONTEXT_LENGTH=16384 ollama serve` w osobnym terminalu (albo dopisz zmienną do plist usługi) |
| Linux (systemd)   | `sudo systemctl edit ollama` → `[Service]` `Environment="OLLAMA_CONTEXT_LENGTH=16384"` → `sudo systemctl restart ollama` |
| Windows           | `setx OLLAMA_CONTEXT_LENGTH 16384`, zamknij Ollamę z zasobnika i uruchom ponownie |

Na laptopie z 8 GB RAM użyj `8192` – większe okno to większy KV cache w pamięci.
Sprawdzenie: `ollama ps` (kolumna CONTEXT) albo `make check` (krok „Okno kontekstu”
ostrzega, gdy okno < 8192).

## Szybki start

```bash
make setup     # uv sync + utworzenie .env z .env.example
make pull      # ollama pull modeli z .env (raz, kilka GB)
make up        # Qdrant w Dockerze (pomiń, jeśli QDRANT_URL=local:...)
make check     # smoke test – wszystko powinno być ✅
make ingest    # indeksuje data/ (przykładowe pliki w data/sample/)
make ask Q="Dlaczego trzeba ustawić okno kontekstu w Ollamie?"
make agent Q="Jaka jest różnica między chunkingiem fixed-size a recursive?"
make eval      # recall@k i MRR na eval/golden_sample.jsonl
```

Na gałęzi `example/regulamin` `.env.example` wskazuje korpus regulaminu – zamiast `make ingest`/`make eval`
użyj komend z sekcji [Przykład](#przykład-regulamin-studiów-pw-gałąź-exampleregulamin).

Własne dokumenty (PDF/MD/TXT) wrzuć do `data/raw/` i uruchom `make ingest ARGS=--recreate`.
`make ingest` indeksuje katalog `DATA_DIR` z `.env` (domyślnie `data/`); jednorazowo inny katalog:
`make ingest DATA=data/raw`.
Przydatne flagi: `make ask Q="..." ARGS="--show-context -k 8"`,
`make ingest ARGS="--chunking fixed --collection docs_fixed"`, `uv run rag --help`.

Przykładowy wynik `make ask Q="Czym jest chunking?"`:

```
Chunking to proces dzielenia długich tekstów na mniejsze fragmenty (chunki), aby zmniejszyć obciążenie
okna kontekstu i poprawić precyzję odpowiedzi. Domyślną strategią jest 'recursive', która dzieli tekst
po akapitach, liniach i zdaniach, a następnie skleja małe kawałki do docelowej długości [2].
Alternatywnie stosuje się 'fixed-size', tnący tekst na stałą liczbę znaków z nakładką (overlap) [5].

Źródła cytowane:
  [2] chunking.md › Recursive  (score 0.743)
      „Chunking recursive próbuje dzielić tekst po naturalnych granicach: najpierw po akapitach, ...”
  [5] chunking.md › Fixed-size  (score 0.657)
      „Chunking fixed-size tnie tekst na kawałki o stałej liczbie znaków.”
Pobrane (retrieved), niecytowane:
  [1] chunking.md › Chunking  (score 0.853)
  [3] chunking.md › Chunking po strukturze  (score 0.712)
  [4] chunking.md › Po co dzielić dokumenty  (score 0.696)
retrieval 0.13s · generacja 7.15s · structured output: prompt
```

Znaczniki `[n]` w tekście wstawia model; listę źródeł i cytaty („…”) buduje kod z metadanych
i treści chunków – model nie może wymyślić nazwy pliku ani cytatu. „Pobrane, niecytowane” to chunki,
które retrieval znalazł, ale model ich nie użył (warto je oglądać przy debugowaniu retrievalu).

## Przykład: Regulamin studiów PW (gałąź `example/regulamin`)

RAG nad Regulaminem studiów w Politechnice Warszawskiej z chunkingiem po strukturze dokumentu
(rozdział → § → ustęp) i odpowiedziami cytującymi paragrafy („§ 12 ust. 3”).

```bash
git checkout example/regulamin
cp .env.example .env            # DATA_DIR=data/raw, COLLECTION=regulamin_structure, CHUNKING=structure
make fetch-data                 # pobiera PDF-y z BIP PW wg data/raw/manifest.json (nie są w gicie)
make up && make ingest-regulamin  # kolekcje regulamin_structure i regulamin_fixed
make ask Q="Ile razy mogę podchodzić do egzaminu poprawkowego?"
make ask Q="..." ARGS="--collection regulamin_fixed"   # ten sam retrieval na chunkach fixed-size
make ask Q="..." ARGS="--filter doc_type=przepisy_przejsciowe"  # metadata filtering
make eval-regulamin ARGS=--generate                    # obie kolekcje, złoty zbiór regulaminu
```

**Korpus** (`data/raw/manifest.json`): regulamin obowiązujący od 1.10.2026 i przepisy przejściowe
od 1.10.2026 (oba `version=aktualna`). Metadane z manifestu (`version`, `valid_from`, `doc_type`,
`label`) trafiają do każdego chunka, więc można po nich filtrować (`--filter pole=wartość`).

**Przepisy przejściowe** na BIP to PDF złożony z samych obrazów (tekst polski i angielski, bez warstwy
tekstowej). Zamiast OCR ingest czyta ręczną transkrypcję polskiej części z
`data/transcriptions/przepisy_przejsciowe_2026.txt` (pole `text_file` w manifeście).

**Chunking po strukturze** (`src/rag/legal.py`): 1 chunk = 1 ustęp z nagłówkiem kontekstowym
(`Regulamin › Rozdział III … › § 12 ust. 3`). Chunki fixed-size dostają `source_ids` ustępów, na które
zachodzą, więc obie strategie porównujemy tymi samymi metrykami (`--level paragraph` liczy je per §).

Wyniki porównania (fixed vs po paragrafach, qwen3.5:4b vs 9b): [eval/results/comparison.md](eval/results/comparison.md).
Demo krok po kroku (komendy, pytania, wyniki, wnioski): [docs/demo.md](docs/demo.md).

**Znane ograniczenia ekstrakcji PDF**: część wyjustowanych linii ma rozstrzelone litery bez wyraźnych
granic słów (np. „orazopisane”, „Praktyki za w od owe”); normalizacja (`src/rag/normalize.py`) naprawia
większość przypadków, ale nie wszystkie.

## Konfiguracja (.env)

Wszystkie ustawienia są w `.env` (szablon z komentarzami: `.env.example`), wczytuje je
`src/rag/config.py`. Najważniejsze:

| Zmienna | Domyślnie | Opis |
|---------|-----------|------|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Ollama, `qwen3.5:4b` | serwer zgodny z OpenAI |
| `LLM_REASONING_EFFORT` | `none` | `none` wyłącza tryb thinking (patrz niżej) |
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | model embeddingów; `EMBED_BASE_URL` domyślnie = `LLM_BASE_URL` |
| `DATA_DIR` | `data` | katalog z dokumentami dla `make ingest` |
| `QDRANT_URL`, `COLLECTION` | `http://localhost:6333`, `docs` | vector store |
| `TOP_K`, `CHUNKING`, `CHUNK_SIZE`, `CHUNK_OVERLAP` | 5, `recursive`, 800, 120 | parametry RAG (długości w znakach) |
| `LLM_TEMPERATURE` | `0` | temperatura generacji (0 = powtarzalne odpowiedzi i eval) |
| `EMBED_QUERY_INSTRUCTION` | instrukcja dla qwen3-embedding | doklejana do zapytań (nie do dokumentów); dla innych modeli pusta |
| `STRUCTURED_OUTPUT` | `auto` | `auto` / `native` / `tool` / `prompt` |

Zmienne z powłoki mają pierwszeństwo przed `.env`: `LLM_MODEL=qwen3.5:9b make eval ARGS=--generate`.

**Thinking.** qwen3.5 domyślnie „myśli” przed odpowiedzią: „Ile to 17*23?” trwa z thinkingiem
**26,8 s**, z `LLM_REASONING_EFFORT=none` **0,2 s**. W RAG odpowiedź jest w kontekście – zostaw `none`.

**Zmiana modelu embeddingów** wymaga ponownego ingestu (`make ingest ARGS=--recreate`):
wektory z różnych modeli są nieporównywalne. Program wykryje niezgodność i podpowie, co zrobić.

## Structured output: trzy tryby

Odpowiedź RAG to JSON zgodny ze schematem `Answer` (`src/rag/generate.py`): `answer` (tekst ze
znacznikami `[n]`) i `citations` (numery fragmentów; pusta lista = odmowa). `src/rag/structured.py`
obsługuje trzy tryby:

1. **native** – `response_format` z JSON Schema; serwer ogranicza dekodowanie, więc JSON zawsze jest poprawny.
2. **prompt** – schemat w prompcie + walidacja Pydantic + jedno ponowienie z opisem błędu.
3. **tool** – schemat jako parametry narzędzia `submit_answer` (tylko jawnie: `STRUCTURED_OUTPUT=tool`).

`auto` (domyślnie) próbuje native, a gdy serwer go nie obsługuje – prompt. Wykryty tryb trafia do
`.cache/structured_output.json` (reset: `make reset-cache`; `make check` wykrywa go zawsze od nowa).
Na Macu Ollama uruchamia qwen3.5 runnerem **MLX**, który (Ollama 0.40.10) zwraca na `response_format`
błąd `501` – stąd linia `structured output: native niedostępny → prompt`. Na Windows/Linux (GGUF)
i w Gemini native powinien działać; to może się zmienić w nowszych wersjach Ollamy. Małe modele
zwracają czasem `"citations": "[2], [3]"` – walidator `Answer.repair_citations` naprawia to przed
walidacją. Szczegóły każdej próby: `uv run rag -v ask "..."`.

## Słabszy laptop: Gemini API

Odkomentuj blok Gemini w `.env` (szablon w `.env.example`): klucz z
[Google AI Studio](https://aistudio.google.com/apikey), `EMBED_MODEL=gemini-embedding-001`, pusta
`EMBED_QUERY_INSTRUCTION`, potem `make ingest ARGS=--recreate`. Qdrant nadal działa lokalnie.

**Prywatność:** lokalnie (Ollama + Qdrant) dane nie opuszczają komputera. W darmowym tierze API treść
promptów może być czytana przez ludzi i używana do trenowania – **nie wysyłaj** danych osobowych,
danych z pracy ani niczego objętego NDA. Klucz trzymaj tylko w `.env` (jest w `.gitignore`).

## Ewaluacja

Bez pomiaru nie wiesz, czy zmiana coś poprawiła. Złoty zbiór to plik JSONL:

```json
{"id": "s1", "question": "Czym jest RAG?", "answer": "...", "source_ids": ["rag_wprowadzenie.md#Czym jest RAG"], "tags": ["easy"]}
```

`source_ids` to identyfikatory źródeł **niezależne od chunkingu**: `plik.pdf#p3` (strona),
`plik.md#Nagłówek` (sekcja), `plik.txt`. Każdy chunk ma w metadanych listę `source_ids`,
które obejmuje, więc metryki są porównywalne między strategiami chunkingu. Pusta lista oznacza
pytanie spoza dokumentów (oczekujemy odmowy).

```bash
make eval                                   # recall@1/3/5, MRR (sam retrieval, szybko)
make eval ARGS=--generate                   # + poprawność odmów, trafność cytatów, czas odpowiedzi
make eval ARGS="--golden eval/moj_zbior.jsonl"
```

Generacja działa z `temperature=0` (powtarzalność). Wyniki: tabela w terminalu oraz
`eval/results/<czas>_<zbiór>_<kolekcja>[_<model>].json` i `.md`.

Kolumna `ctx` to średnia liczba znaków kontekstu w top-k. Porównując strategie chunkingu, patrz
na recall razem z nią: większe chunki „łapią” więcej źródeł przy tym samym k, ale dają modelowi
więcej tekstu (wolniej, więcej szumu). Uczciwe porównanie to podobny budżet znaków, nie to samo k.

## Co dalej (TODO)

W `src/rag/retrieve.py` są szkielety z opisem krok po kroku – po każdej zmianie uruchom `make eval`:

| TODO | Gdzie | Idea |
|------|-------|------|
| `TODO(hybrid)` | `hybrid_search()`, `store.py` | dense + sparse BM25 (FastEmbed `Qdrant/bm25`), fuzja RRF w Qdrant Query API (`uv add fastembed`) |
| `TODO(reranking)` | `rerank()` | cross-encoder `BAAI/bge-reranker-v2-m3` na top-20 kandydatach |
| `TODO(metadata-filtering)` | `build_filter()` | jest równość (`--filter pole=wartość`); dopisz `MatchAny`, zakresy (`valid_from`), indeksy |
| `TODO(query-rewriting)` | `rewrite_query()` | przepisanie pytania / multi-query / HyDE |

## Struktura repozytorium

```
src/rag/   jeden moduł = jeden krok pipeline'u (opis w docstringu modułu; mapa w AGENTS.md)
data/      sample/ – przykładowe dokumenty · raw/ – Twoje dokumenty (poza gitem)
eval/      złote zbiory (JSONL) · results/ – wyniki (poza gitem)
tests/     testy jednostkowe bez sieci (make test)
scripts/   pomocnicze skrypty (pobieranie modeli)
```

## Troubleshooting

| Objaw | Przyczyna / rozwiązanie |
|-------|-------------------------|
| `❌ Ollama działa` | Ollama nie jest uruchomiona: `ollama serve` / aplikacja / `brew services start ollama`. Sprawdź `curl localhost:11434/api/version`. |
| `❌ Modele pobrane` | `make pull` albo `ollama pull <model>`. Nazwa w `.env` musi zgadzać się z `ollama list`. |
| `❌ Qdrant działa` | `make up` (czy Docker/OrbStack działa? `docker ps`). Alternatywa: `QDRANT_URL=local:./.qdrant`. |
| `Storage folder ./.qdrant is already accessed` | tryb lokalny pozwala na jeden proces – zamknij inne `uv run rag ...`. |
| `Kolekcja ... nie istnieje` | najpierw `make ingest`. |
| `Kolekcja zindeksowana modelem X` / wymiar się nie zgadza | zmieniłeś `EMBED_MODEL` – `make ingest ARGS=--recreate`. |
| odpowiedzi ignorują część kontekstu | za małe okno kontekstu – patrz [sekcja o oknie](#okno-kontekstu-ollamy--koniecznie-przeczytaj). |
| bardzo wolne odpowiedzi (dziesiątki sekund) | thinking włączony – `LLM_REASONING_EFFORT=none`; albo model nie mieści się w RAM (swap) – weź mniejszy. |
| `structured output: native niedostępny → prompt` | oczekiwane na Macu (MLX) – patrz [Structured output](#structured-output-trzy-tryby). |
| po aktualizacji Ollamy wciąż używany stary tryb | `make reset-cache`. |
| `żaden tryb structured output nie zadziałał` | model za słaby na JSON albo kontekst ucięty; spróbuj `qwen3.5:9b`, mniejszego `TOP_K` lub większego okna. |
| PDF daje pusty tekst | skan bez warstwy tekstowej – potrzebny OCR (poza zakresem startera). |
| `make: command not found` (Windows) | używaj `uv run rag <komenda>`. |
