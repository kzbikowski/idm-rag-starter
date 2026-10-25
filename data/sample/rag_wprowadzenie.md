# Wprowadzenie do RAG

Ten plik jest przykładowym dokumentem dla startera. Zastąp go własnym korpusem.

## Czym jest RAG

Retrieval-Augmented Generation (RAG) to technika, w której model językowy odpowiada na pytanie
na podstawie fragmentów dokumentów dostarczonych w prompcie. Najpierw system wyszukuje (retrieval)
fragmenty pasujące do pytania, a następnie model generuje odpowiedź, cytując te fragmenty.
RAG pozwala korzystać z wiedzy, której model nie widział podczas trenowania, np. z wewnętrznych
regulaminów, bez kosztownego fine-tuningu.

## Etapy pipeline'u

Typowy pipeline RAG ma dwie fazy. Faza offline (ingest) obejmuje wczytanie dokumentów,
normalizację tekstu, chunking, policzenie embeddingów i zapis do bazy wektorowej.
Faza online obejmuje embedding pytania, retrieval top-k chunków, zbudowanie promptu z kontekstem
i wygenerowanie odpowiedzi. W tym repozytorium bazą wektorową jest Qdrant, a embeddingi liczy
model qwen3-embedding:0.6b uruchomiony w Ollamie.

## Okno kontekstu

Okno kontekstu to maksymalna liczba tokenów, które model przetwarza naraz: prompt systemowy,
chunki, pytanie i odpowiedź. Ollama domyślnie używa małego okna i po cichu obcina zbyt długi prompt,
dlatego w tym kursie ustawiamy zmienną OLLAMA_CONTEXT_LENGTH na co najmniej 8192 tokeny.
Obcięty prompt nie powoduje błędu – model po prostu nie widzi części chunków.

## Halucynacje

Halucynacja to odpowiedź, która brzmi wiarygodnie, ale nie wynika z kontekstu ani z faktów.
RAG zmniejsza ryzyko halucynacji, ale go nie eliminuje: model może źle połączyć fragmenty
albo odpowiedzieć z własnej wiedzy, gdy retrieval nie znajdzie właściwego chunka.
Dlatego prosimy model o cytowanie źródeł i o jawną odmowę, gdy kontekst nie zawiera odpowiedzi.
