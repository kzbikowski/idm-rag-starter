"""Normalizacja tekstu po ekstrakcji z PDF.

Ekstrakcja tekstu z PDF jest stratna: pojawiają się ligatury, łamanie wyrazów na końcu linii,
podwójne spacje, a czasem spacje WEWNĄTRZ słów (np. "Usta wę", "term inami") – wynik tego,
jak PDF pozycjonuje pojedyncze glify. Takie artefakty psują zarówno embeddingi, jak i BM25,
więc naprawiamy je przed chunkingiem.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}

# Słowo = litery (w tym polskie znaki). Cyfr i interpunkcji nie sklejamy.
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# Fragment liter + spacja, po których następuje kolejny fragment liter (kandydat na rozbite
# słowo). Prawy fragment jest w lookahead, żeby nie „zjadał” go mecz poprzedniej pary.
_SPLIT_PAIR = re.compile(r"\b([^\W\d_]+) (?=([^\W\d_]+)\b)", re.UNICODE)

# Krótkie polskie słowa, które są samodzielne – nigdy nie doklejamy ich do sąsiada.
# fmt: off
_STANDALONE = {
    "a", "i", "o", "u", "w", "z", "we", "ze", "na", "do", "od", "po", "za", "przy", "nie",
    "się", "lub", "albo", "oraz", "jest", "są", "to", "ten", "ta", "jak", "ich", "go",
    "dla", "pod", "nad", "przez", "być", "ma", "mają", "może", "mogą", "tym", "tej", "też",
    "już", "gdy", "czy", "jej", "nr", "pkt", "ust", "lit", "art", "dni", "rok", "ani", "bez",
    "co", "je", "jako", "aby", "tak", "lat", "dnia", "roku", "lecz", "tylko",
}
# fmt: on

_LETTER_SPACED = re.compile(r"\b(?:[^\W\d_] ){3,}[^\W\d_]\b", re.UNICODE)
_PREFIX_LEN = 5


def _frequent_prefixes(vocab: Counter[str]) -> set[str]:
    """Rdzenie (pierwsze 5 liter), które łącznie, we wszystkich formach, występują ≥ 3 razy."""
    stems: Counter[str] = Counter()
    for word, count in vocab.items():
        if len(word) >= _PREFIX_LEN:
            stems[word[:_PREFIX_LEN]] += count
    return {stem for stem, count in stems.items() if count >= 3}


def fix_letter_spaced_line(line: str) -> str:
    """Naprawia linię wyjustowaną rozstrzelonymi literami.

    Przykład z PDF: "cz a s i e  ( s e m e s t rz e  l u b  r o k u" – w takich liniach
    granicą słów są PODWÓJNE spacje, a pojedyncze spacje rozdzielają litery jednego słowa.
    Rozpoznajemy je po tym, że większość „słów” ma 1–2 znaki, i sklejamy litery w słowa.
    """
    tokens = line.split()
    if len(tokens) < 8 or line.count("  ") < 3:
        return line
    if sum(len(t) <= 2 for t in tokens) / len(tokens) < 0.7:
        return line
    # Numer ustępu/punktu na początku linii („5. ”, „3) ”) zostawiamy osobno.
    enumerator = re.match(r"^\s*(\d+[a-z]?[.)])\s+", line)
    prefix = enumerator.group(1) + " " if enumerator else ""
    rest = line[enumerator.end() :] if enumerator else line
    words = [w.replace(" ", "") for w in re.split(r" {2,}", rest.strip())]
    return prefix + " ".join(w for w in words if w)


def basic_clean(text: str) -> str:
    """Normalizacja Unicode, ligatury, łamanie wyrazów i zbędne spacje."""
    text = unicodedata.normalize("NFC", text)
    for lig, repl in LIGATURES.items():
        text = text.replace(lig, repl)
    text = text.replace(" ", " ").replace("­", "")  # twarda spacja, miękki dywiz
    # Najpierw linie rozstrzelone (zanim sklejanie dywizów połączy je z następną linią).
    text = text.replace("\t", " ")
    text = "\n".join(fix_letter_spaced_line(line) for line in text.split("\n"))
    # "kształ-\ncenia", "organi -\nzacyjnych" -> jedno słowo (dywiz na końcu linii przed małą literą)
    text = re.sub(r"(\w) ?-\n(?=[a-ząćęłńóśźż])", r"\1", text)
    text = re.sub(r"[ ]+", " ", text)
    # Druk rozstrzelony: "t y g o d n i" -> "tygodni" (≥ 4 pojedyncze litery pod rząd).
    text = _LETTER_SPACED.sub(lambda m: m.group(0).replace(" ", ""), text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_counts(text: str) -> Counter[str]:
    """Licznik słów (lowercase) – używany jako „słownik” dokumentu."""
    return Counter(w.lower() for w in _WORD.findall(text))


def join_split_words(text: str, vocab: Counter[str]) -> str:
    """Skleja słowa rozbite spacją, korzystając ze słownika zbudowanego z samego korpusu.

    Heurystyka: "Usta wę" zamieniamy na "Ustawę", jeżeli forma sklejona ("ustawę") występuje
    gdzieś w korpusie, a przynajmniej jeden z fragmentów NIE jest samodzielnym słowem
    (występuje rzadziej niż forma sklejona i nie jest na liście krótkich słów).
    Nie potrzebujemy zewnętrznego słownika – dokument „naprawia się” sam sobą.

    Args:
        text: tekst po `basic_clean`.
        vocab: licznik słów z całego dokumentu (`word_counts`).
    """
    prefixes = _frequent_prefixes(vocab)

    def is_standalone(fragment: str, joined_count: int) -> bool:
        f = fragment.lower()
        return f in _STANDALONE or vocab.get(f, 0) > joined_count

    def is_orphan(fragment: str, max_count: int) -> bool:
        """Fragment, który (prawie) nigdy nie występuje jako samodzielne słowo."""
        f = fragment.lower()
        return f not in _STANDALONE and vocab.get(f, 0) <= max_count

    def repl(match: re.Match[str]) -> str:
        """Zwraca lewy fragment bez spacji (sklejenie) albo bez zmian."""
        left, right = match.group(1), match.group(2)
        joined = left + right
        joined_count = vocab.get(joined.lower(), 0)
        if joined_count > 0:
            # Reguła 1: forma sklejona istnieje w korpusie.
            if is_standalone(left, joined_count) and is_standalone(right, joined_count):
                return match.group(0)
            return left
        # Reguła 2: forma sklejona nie występuje (inna końcówka fleksyjna, np. "ustawę" vs
        # "ustawy"). Sklejamy tylko, gdy spacja przecina rdzeń częstego słowa: lewy fragment
        # jest krótszy niż rdzeń ("Usta|wę" -> rdzeń "ustaw"), oba fragmenty są
        # „sierotami”, a rdzeń sklejonego słowa występuje w korpusie ("ustawa", "ustawy").
        if (
            len(left) < _PREFIX_LEN
            and is_orphan(left, max_count=5)
            and is_orphan(right, max_count=1)
            and joined.lower()[:_PREFIX_LEN] in prefixes
        ):
            return left
        return match.group(0)

    # Dwa przebiegi: łapie też słowa rozbite na 3 części ("kształ cen ia").
    for _ in range(2):
        text = _SPLIT_PAIR.sub(repl, text)
    return text
