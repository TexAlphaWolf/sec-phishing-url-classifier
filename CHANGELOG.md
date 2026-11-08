# Changelog

## 1.0.0

First release. Scores a URL for phishing from the text of its origin (scheme, user info, host, port): hand-made lexical features plus hashed host trigrams, a logistic regression stored as plain JSON and scored in pure Python. `urlclf score`, `train`, `synth`, `serve` (local JSON API). Training splits by site, picks the threshold on a validation part for a 1% false-positive budget, and reports on a held-out test part with confidence intervals and the comparison models. The bundled model was trained on the PhiUSIIL dataset (see docs/evaluation.md for the figures and for what they do not show).
