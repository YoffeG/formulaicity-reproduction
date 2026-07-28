# Data files

This directory contains the complete local inputs used by Figures 3--6.

- `TOTHT_corpus_full` is the serialized Python corpus containing the
  morphological and surface Hebrew tokens. It is a trusted pickle input and
  should not be replaced by an untrusted download.
- `Genesis_Lists_Narrative.csv` supplies the Genesis genealogical-list labels
  used by Figure 3.
- `Exodus_P-nonP_Roemer_AB.csv` supplies the Exodus P/non-P labels used by
  Figure 4.
- `Leviticus _PH_14-3-2022.csv` supplies the Leviticus P/H labels used by
  Figures 5 and 6.
- `Genesis_P-nonP_Roemer_AB.csv` is retained with the source bundle for the
  alternative Genesis P/non-P analysis, although it is not read by the four
  default reproduction commands.

SHA-256 checksums:

```text
c8829d41643e66f4f92d7a68a312fb3f5b493750672d6599aa9ce1df90203bcf  Exodus_P-nonP_Roemer_AB.csv
5e3ed953358a813aa2f0d54c01bce13790b2830afa222c793747404cd1a91573  Genesis_Lists_Narrative.csv
a7c7019b12ad984d8fbc22e39540f90f834ff79cc88287f25116cb23581811ac  Genesis_P-nonP_Roemer_AB.csv
5f261acc33f57264294418ba684ec7dab227a621b782638069480e08dc788313  Leviticus _PH_14-3-2022.csv
7044a7a9ffb550fc15254bbaae909358e31e9ecb845394daf63fcfac0fe5081a  TOTHT_corpus_full
```

Before making a public GitHub repository, confirm that the corpus and expert
annotation tables may be redistributed under the repository's chosen license.
