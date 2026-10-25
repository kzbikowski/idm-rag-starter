# Demo: RAG nad Regulaminem studiów PW

Na kilku pytaniach do Regulaminu studiów PW zobaczysz kolejno:
- RAG, który działa;
- typowe sposoby, w jakie RAG zawodzi;
- wpływ chunkingu i wpływ modelu na odpowiedź.

Każdy krok to jedna komenda, pod nią wynik i wniosek. Wyniki pochodzą z przebiegu z qwen3.5:4b/9b na
Ollamie, `TOP_K=5`, `temperature=0`. U Ciebie sformułowania mogą się nieco różnić, a przy innym
modelu lub providerze także wnioski. Pełne tabele są w
[eval/results/comparison.md](../eval/results/comparison.md).

## Przygotowanie

Komendy z sekcji [Przykład w README](../README.md#przykład-regulamin-studiów-pw-gałąź-exampleregulamin)
(`cp .env.example .env` nadpisuje `.env` z `main`; `make setup` kopiuje tylko, gdy `.env` nie istnieje),
a do kroku 5 dodatkowo `ollama pull qwen3.5:9b`. Domyślna kolekcja: `regulamin_structure` (1 chunk = 1 ustęp).

## 1. Korpus i chunki

```bash
cat data/raw/manifest.json
```

- PDF-y nie są w gicie. Repo trzyma manifest z adresami URL, sumami SHA-256 i metadanymi
  (`version`, `valid_from`, `doc_type`), które ingest zapisuje w payloadzie chunków.
- Przepisy przejściowe to PDF z samym obrazem, bez warstwy tekstowej. Ingest czyta ręczną transkrypcję
  z `data/transcriptions/` (`text_file` w manifeście).

W dashboardzie Qdranta (http://localhost:6333/dashboard → `regulamin_structure` → *Points*) obejrzyj
payload chunka. Zawiera:
- `text` z nagłówkiem kontekstowym „Regulamin › Rozdział › § 36 ust. 2”;
- `source_ids`, `paragraph`, `ustep`, `version`.

**Wniosek:** ekstrakcja i normalizacja tekstu to pełnoprawna część pipeline'u. `source_ids` nie
zależą od chunkingu, więc ten sam złoty zbiór porównuje obie kolekcje.

## 2. Sukces

```bash
make ask Q="Jak obliczana jest ocena ze studiów?"
```

**Wynik:** „0,6 średniej ważonej … 0,3 oceny za pracę dyplomową … 0,1 oceny za egzamin dyplomowy
[§ 32 ust. 2]”, a pod spodem cytat z fragmentu.

- **Źródła cytowane vs pobrane (retrieved):** model dostał 5 fragmentów, a zacytował 1.
- **Numery `[n]` zwraca model.** Zamianę na „§ 32 ust. 2” i dociągnięcie cytatu robi kod, nie model.
- **`structured output: prompt`:** na Ollamie z MLX natywny `response_format` nie działa (README).
- **`ARGS=--show-context`** pokazuje dokładnie to, co trafiło do promptu.

## 3. Porażki

**a) Pytanie spoza dokumentu**

```bash
make ask Q="Kto jest obecnie rektorem Politechniki Warszawskiej?"
```

**Wynik:** poprawna odmowa („W podanym kontekście nie znajduje się informacji…”) bez cytatów. Mimo to
retrieval zwrócił 5 fragmentów (§ 1, przypis przepisów przejściowych…) ze score ~0,6.

**Wniosek:** wyszukiwanie wektorowe **zawsze** zwraca top-k. O tym, że odpowiedzi nie ma, musi
zdecydować model albo próg podobieństwa.

**b) Pytanie wymagające kilku paragrafów**

```bash
make ask Q="Nie zapłaciłem za powtarzanie przedmiotu. Jakie mogą być konsekwencje?"
```

**Wynik:** błędna odmowa („nie ma informacji o sankcjach”). Retrieval znalazł § 21 ust. 1 (obowiązek
opłaty), ale nie § 37 ust. 2 ani § 33 ust. 2 (skreślenie za niewniesienie opłat). Recall@5 = 0,33.

**Wniosek:** pytanie mówi „nie zapłaciłem”, a przepis „niewniesienie opłat”. Pierwszy krok
rozumowania trafia, drugiego podobieństwo semantyczne już nie łapie.

**c) To samo pytanie zadane agentowi**

```bash
make agent Q="Nie zapłaciłem za powtarzanie przedmiotu. Jakie mogą być konsekwencje?"
```

**Wynik (4b):** agent wywołuje `search_docs` raz, z zniekształconym zapytaniem, i odpowiada na podstawie
§ 21 ust. 1 i § 15 ust. 7. Dorzuca też § 38 (kary dyscyplinarne), którego nie było w kontekście.

**Wniosek:** tool calling nie naprawia słabego retrievalu, a mały model chętnie „uzupełnia” z pamięci.

## 4. Wpływ chunkingu

```bash
make ask Q="Na jak długi okres można maksymalnie otrzymać urlop od zajęć?"
make ask Q="Na jak długi okres można maksymalnie otrzymać urlop od zajęć?" ARGS="--collection regulamin_fixed"
make ask Q="Na jak długi okres można maksymalnie otrzymać urlop od zajęć?" ARGS="-k 10"
```

| wariant | wynik |
|---|---|
| `regulamin_structure`, k=5 | ✗ „okres nie jest precyzowany”; pobrane § 36 ust. 1, 6, 5, 4, 9, **bez ust. 2** |
| `regulamin_fixed`, k=5 | ✓ „dwa następujące bezpośrednio po sobie semestry”, ale cytat tylko jako „s. 44” |
| `regulamin_structure`, k=10 | ✓ ust. 2 na pozycji 6, odpowiedź „[§ 36 ust. 2]” |

**Wnioski:**
- § 36 ust. 2 to jedno krótkie zdanie („Urlop może być udzielony na okres nie dłuższy niż dwa…”).
  Sąsiednie ustępy o urlopie są do pytania bardziej podobne niż on sam.
- Duży chunk fixed-size obejmuje kilka ustępów naraz, więc trafia. Płaci za to ~2× dłuższym promptem
  i mniej precyzyjnym cytatem.
- Zwiększenie k to najprostsza naprawa. Lepsze są **reranking** i **hybrid search**: BM25 złapie
  dosłowne „okres nie dłuższy niż”.

## 5. Wpływ modelu

```bash
make ask Q="Zostałem skreślony z listy studentów w 2024 roku. Czy w 2027 roku mogę złożyć wniosek o wznowienie studiów?"
LLM_MODEL=qwen3.5:9b make ask Q="Zostałem skreślony z listy studentów w 2024 roku. Czy w 2027 roku mogę złożyć wniosek o wznowienie studiów?" ARGS="--collection regulamin_fixed"
```

- **qwen3.5:4b** odpowiada „Nie, … upłynął okres 5 lat”. Cytuje **poprawnie** przepisy przejściowe
  pkt 2a i § 15 ust. 8, ale z 2024 + 5 = 2029 wyciąga błędny wniosek.
- **qwen3.5:9b** odpowiada „2027 mieści się w okresie 5 lat, możesz złożyć wniosek”. Ten sam
  kontekst daje poprawny wniosek, ale odpowiedź trwa ok. 2× dłużej.

**Wniosek:** trafne cytowania nie gwarantują poprawnej odpowiedzi. Recall i MRR są tu idealne, a
błąd widać dopiero w ręcznej ocenie (sekcja 3 w `comparison.md`). Wyniki bywają też niestabilne:
9b na `regulamin_structure` przytacza regułę, ale wniosku nie wyciąga.

## 6. Ewaluacja

```bash
make eval-regulamin                      # recall@k / MRR / ctx dla obu kolekcji, bez LLM
make eval-regulamin ARGS=--generate      # + odpowiedzi modelu (kilka minut)
```

Przy tym samym k fixed-size ma wyższy recall (R@5 0,87 vs 0,77), ale kolumna `ctx` pokazuje ~2× więcej
tekstu; przy równym budżecie kontekstu wygrywa chunking po ustępach. Szczegóły: `eval/results/comparison.md`.

## Co dalej

TODO w `src/rag/retrieve.py`. Po każdej zmianie uruchom `make eval-regulamin` i porównaj wyniki.
- **hybrid search** (BM25 + fuzja RRF): pomoże w krokach 3b i 4;
- **reranking** (`bge-reranker-v2-m3`): przestawi § 36 ust. 2 wyżej;
- **query rewriting**: „nie zapłaciłem” → „niewniesienie opłaty”;
- **metadata filtering** (`build_filter()`): jest `--filter pole=wartość`; dopisz `MatchAny` i zakresy dat.
