# Regulamin PW – porównanie: chunking fixed vs po paragrafach, qwen3.5:4b vs qwen3.5:9b

Korpus: Regulamin studiów PW (od 1.10.2026) + przepisy przejściowe (transkrypcja),
`DEFAULT_FILTERS=version=aktualna`. Złoty zbiór: `eval/golden_regulamin.jsonl` (15 pytań).
Środowisko: Apple Silicon 24 GB, Ollama 0.40.10 (MLX, okno 16384), qwen3-embedding:0.6b, `TOP_K=5`,
`CHUNK_SIZE=800`, `CHUNK_OVERLAP=120`, temperature=0, structured output: prompt. Data: 2026-10-25.

| kolekcja | chunków | chunking |
|---|---|---|
| regulamin_structure | 249 | 1 chunk = 1 ustęp (+ nagłówek: dokument › rozdział › § ust.) |
| regulamin_fixed | 104 | okno 800 znaków, overlap 120, na tekście bez layoutu |

## 1. Generacja (`make eval-regulamin ARGS=--generate`)

| kolekcja | model | R@1 | R@3 | R@5 | MRR | kontekst [zn.] | odmowy/odpowiedzi OK | cytat z właściwego § | znaczniki [n] | śr. czas [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| regulamin_fixed | qwen3.5:4b | 0.47 | 0.76 | 0.87 | 0.89 | 3977 | 1.00 | 0.69 | 0.92 | 6.18 |
| regulamin_fixed | qwen3.5:9b | 0.47 | 0.76 | 0.87 | 0.89 | 3977 | 1.00 | 1.00 | 0.92 | 13.11 |
| regulamin_structure | qwen3.5:4b | 0.55 | 0.70 | 0.77 | 0.89 | 2057 | 0.87 | 0.85 | 0.92 | 5.27 |
| regulamin_structure | qwen3.5:9b | 0.55 | 0.70 | 0.77 | 0.89 | 2057 | 1.00 | 0.85 | 1.00 | 8.71 |

Metryki retrievalu są powtarzalne. Metryki generacji między przebiegami różnią się o pojedyncze
pytania, mimo `temperature=0`. Przy 15 pytaniach jedna zmiana to ±0,07, więc różnic poniżej ~0,1
nie należy interpretować.

Czasy (Apple Silicon 24 GB): ingest obu kolekcji ~17 s, eval samego retrievalu ~3 s, eval z generacją
(30 pytań) ~3 min dla 4b i ~5,5 min dla 9b. Pierwsze pytanie po zmianie modelu obejmuje jego ładowanie.

„odmowy/odpowiedzi OK” = model cytuje źródła przy pytaniach z regulaminu i nie cytuje przy pytaniach
spoza niego (odmowa = brak cytowań). „cytat z właściwego §” liczony na poziomie identyfikatorów ze złotego
zbioru (ustęp); chunk fixed-size obejmuje kilka ustępów, więc ta metryka faworyzuje fixed.

## 2. Retrieval – poziom ustępu vs paragrafu, równy budżet kontekstu

| kolekcja | top-k | kontekst [zn.] | R@5 (ustęp) | R@5 (§) | R@10 (ustęp) | R@10 (§) | MRR (§) |
|---|---|---|---|---|---|---|---|
| regulamin_fixed | 5 | 3977 | 0.87 | 0.91 | – | – | 0.96 |
| regulamin_structure | 5 | 2057 | 0.77 | 0.85 | – | – | 0.96 |
| regulamin_structure | 10 | 3806 | – | – | 0.95 | 0.95 | 0.96 |

Przy tym samym k chunki fixed-size dają modelowi ~2× więcej tekstu i „zahaczają” o kilka ustępów naraz,
co podbija recall. Przy porównywalnym budżecie kontekstu (~3,8 tys. znaków) chunking po paragrafach
wygrywa: R@10 = 0,95 vs R@5 = 0,87 (ustęp) / 0,91 (§).

## 3. Poprawność odpowiedzi

Metryki automatyczne nie sprawdzają, czy wniosek jest poprawny. Przykład: w r10 („skreślony w 2024,
wniosek w 2027?”) qwen3.5:4b cytuje właściwe przepisy (przepisy przejściowe pkt 2a, § 15 ust. 8),
a mimo to odpowiada „nie” – choć 2024 + 5 lat = 2029. Recall, MRR i trafność cytatów są tu idealne.
Pytania łączące kilka przepisów albo wymagające rachunku dat warto oceniać ręcznie – np. czytając
odpowiedzi zapisane w `eval/results/*.json` po `make eval-regulamin ARGS=--generate`.

## Wnioski

- Retrieval działa dobrze w obu wariantach (MRR ≈ 0,9); błędy są głównie po stronie **rozumowania modelu**:
  qwen3.5:4b myli się w pytaniach wymagających połączenia przepisów i rachunku dat (r10, r12), 9b radzi sobie
  wyraźnie lepiej, ale ok. 1,7–2× wolniej (8,7–13,1 s vs 5,3–6,2 s na odpowiedź).
- Chunking po paragrafach: precyzyjne cytowania (§ … ust. …), mniejszy kontekst, szybsze odpowiedzi; fixed-size:
  wyższy recall przy tym samym k kosztem 2× dłuższego promptu.
- Wykrywanie odmowy po pustych cytowaniach jest niedoskonałe (r15, 4b structure: „brak informacji” + cytat).
