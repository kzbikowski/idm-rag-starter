from rag.normalize import basic_clean, join_split_words, word_counts


def normalize(text: str) -> str:
    """Jak ingest: czyszczenie, potem sklejanie ze słownikiem z tego samego tekstu."""
    cleaned = basic_clean(text)
    return join_split_words(cleaned, word_counts(cleaned))


def test_basic_clean_ligatures_hyphenation_whitespace():
    assert basic_clean("ﬁzyka  i kształ-\ncenie") == "fizyka i kształcenie"


def test_join_when_joined_form_exists_in_corpus():
    corpus = "Dziekan decyduje. Wniosek do dziekana. Decyzja dzi ekana jest ostateczna."
    assert "decyzja dziekana" in normalize(corpus).lower()


def test_join_by_stem_when_inflected_form_missing():
    # "ustawę" nie występuje w korpusie, ale rdzeń "ustaw" – tak (ustawa, ustawy, ustawie).
    corpus = "Ustawa mówi. Zgodnie z ustawy. W ustawie jest. Ustawa - Usta wę z dnia 27 lipca."
    assert "Ustawę z dnia" in normalize(corpus)


def test_does_not_join_real_neighbouring_words():
    corpus = "studia niestacjonarne " * 3 + "studianiestacjonarne"
    text = "studia niestacjonarne trwają dłużej"
    assert join_split_words(text, word_counts(corpus)) == text


def test_does_not_join_standalone_short_words():
    corpus = "w ramach programu " * 5 + "wramach"
    assert join_split_words("w ramach", word_counts(corpus)) == "w ramach"


def test_letter_spaced_text_is_joined():
    assert basic_clean("nie krócej niż 6 t y g o d n i , a") == "nie krócej niż 6 tygodni , a"
    assert basic_clean("i w z") == "i w z"  # krótkie ciągi zostają


def test_letter_spaced_line_with_double_space_word_gaps():
    line = (
        "cz a s i e  ( s e m e s t rz e  l u b  r o k u  a k a d e m i c k i m ) ,  p o  k t ó ry m"
    )
    assert basic_clean(line) == "czasie (semestrze lub roku akademickim), po którym"


def test_normal_line_is_not_treated_as_letter_spaced():
    line = "Student ma  prawo do  jednego egzaminu  poprawkowego w sesji."
    assert basic_clean(line) == "Student ma prawo do jednego egzaminu poprawkowego w sesji."


def test_hyphenation_with_space_before_dash():
    assert (
        basic_clean("jednostek organi -\nzacyjnych Uczelni") == "jednostek organizacyjnych Uczelni"
    )
    assert basic_clean("od 3,49 -\nDostateczny") == "od 3,49 -\nDostateczny"


def test_letter_spaced_line_keeps_leading_enumerator():
    line = "5. W  p r o g r a m i e  s t u d i ó w  w  j ę z y k u  p o l s k i m"
    assert basic_clean(line) == "5. W programie studiów w języku polskim"
