# Chunking

## Po co dzielić dokumenty

Embedding całego długiego dokumentu uśrednia znaczenie wielu tematów, więc słabo pasuje do
konkretnego pytania. Dlatego dzielimy dokumenty na chunki – fragmenty o długości zwykle od
kilkuset do około tysiąca znaków. Za małe chunki tracą kontekst, za duże rozmywają znaczenie
i zajmują dużo miejsca w oknie kontekstu.

## Fixed-size

Chunking fixed-size tnie tekst na kawałki o stałej liczbie znaków. Aby zdanie przecięte na
granicy nie zginęło, sąsiednie chunki nakładają się (overlap), np. o 10–20% długości.
Zaletą jest prostota, wadą – cięcie w środku zdań, list i tabel.

## Recursive

Chunking recursive próbuje dzielić tekst po naturalnych granicach: najpierw po akapitach,
potem po liniach, zdaniach i słowach, aż fragment zmieści się w limicie. Małe kawałki są potem
sklejane do docelowej długości. To domyślna strategia w tym starterze (CHUNKING=recursive).

## Chunking po strukturze

Dokumenty o wyraźnej strukturze, jak akty prawne czy regulaminy, warto dzielić według ich
jednostek: rozdziałów, paragrafów i ustępów. Każdy chunk dostaje wtedy metadane, np. numer
paragrafu, co pozwala precyzyjnie cytować źródło i filtrować wyniki (metadata filtering).
