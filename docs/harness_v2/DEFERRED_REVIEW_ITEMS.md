# Deferred review items

Recorded from Sol re-reviews. These are not fixed.

## Scope

`harness_v2/recognition/skeleton.py` only needs to measure notation used in Japanese high-school Math I/A. That is square roots, fractions, exponents, absolute values, inequalities, plus-minus, set notation (including complement written with an overline), nPr and nCr, factorial, trigonometric functions, degrees, pi, and the symbols for angle, triangle, parallel, and perpendicular.

Cube roots, fourth roots, indexed roots, exotic Unicode math styles, and contrived edge cases are outside that scope. Skeleton findings outside it are recorded here and not fixed. Sol r5 skeleton minors 6–10 are reclassified under that rule below. Sol r5 minors 4 and 5 are challenge-report findings and stay open.

## Still open

### Sol r5 — PR #12 minor 4

- File: `harness_v2/evaluation/challenge.py` around line 101 (`_grading_issue`); the nested read is in `_cell` around line 49
- A grade whose items are the wrong shape stays comparable. Non-dict items make the cell incomplete, but that does not add a report reason, so the CLI still exits 0.
- Repro: with otherwise matching provenance, `grading={"status":"validated","items":["bad"]}`. The report is `comparable: true` with no reasons.

### Sol r5 — PR #12 minor 5

- File: `harness_v2/evaluation/challenge.py` around line 54
- A score integer that does not fit in a float crashes aggregation. `math.isfinite` raises `OverflowError` instead of recording a reason.
- Repro: a JSON score of `10**1000` on an otherwise valid graded item. Building the report raises `OverflowError`.

## Out of Math I/A scope

Skeleton behavior. Recorded, not fixed.

### Sol r5 — PR #12 minor 6

Indexed roots are outside Math I/A scope.

- File: `harness_v2/recognition/skeleton.py` around line 300
- A square-root index that is only grouped as `(2)` is not treated as two on the first pass, so normalization is not idempotent.
- Repro: `skeleton(r"\sqrt[{2}]{x}")` is `root(2,x)`, and `skeleton` of that result is `sqrt(x)`. `root((2),x)` does the same.

### Sol r5 — PR #12 minor 7

Exotic Unicode math styles are outside Math I/A scope.

- File: `harness_v2/recognition/skeleton.py` around line 544
- A bold glyph emitted as `mathbf(v)` is glued onto a preceding command name. The Unicode root and the LaTeX command therefore differ.
- Repro: `\sqrt` plus U+1D42F (mathematical bold small v) is `\sqrtmathbf⁣(v)`. `√` plus the same glyph is `sqrt(mathbf(v))`. `\bar` plus that glyph is `\barmathbf⁣(v)`.

### Sol r5 — PR #12 minor 8

Exotic Unicode math styles are outside Math I/A scope.

- File: `harness_v2/recognition/skeleton.py` around line 532
- A spacing compatibility mark after a bold glyph is attached to a single folded character and corrupts the decoration that the bold glyph already emitted.
- Repro: U+1D42F plus U+00AF is `mathbf(v\overline⁣())`. The same glyph plus combining U+0304 is `overline(mathbf(v))`.

### Sol r5 — PR #12 minor 9

Compatibility accents outside the Math I/A notation list are out of scope.

- File: `harness_v2/recognition/skeleton.py` around line 465, applied again from `skeleton` around line 563
- Compatibility accents other than the four spacing marks already handled still change on the second pass.
- Repro: `x` plus U+00A8 is `ẍ`, then `ẍ`. `a` plus U+00B4 is `á`, then `á`.

### Sol r5 — PR #12 minor 10

Exotic Unicode math styles are outside Math I/A scope.

- File: `harness_v2/recognition/skeleton.py` around line 544
- Adjacent bold glyphs and a LaTeX bold span do not share a canonical form. Each Unicode glyph is its own command.
- Repro: U+1D42F U+1D430 (`𝐯𝐰`) is `mathbf(v)mathbf(w)`. `\mathbf{vw}` is `mathbf(vw)`.

### Sol r6 — PR #12 major

A bracket radicand followed by a barred factor, and the same shape on cube and fourth roots, is outside Math I/A scope. Introduced by the Sol r5 skeleton commits and left unchanged.

- File: `harness_v2/recognition/skeleton.py` around line 481
- A combining overline on the factor after a bracket radicand is dropped, so the barred and plain forms match. Cube and fourth roots do the same.
- Repro: `√[x]y̅` and `√[x]y` both become `sqrt([x])y`. `∛[x]y̅` and `∛[x]y` both become `root(3,[x])y`. `∜[x]y̅` and `∜[x]y` both become `root(4,[x])y`.

### Sol r6 — PR #12 minor

A parenthesized combining stroke on a square-root radicand is a contrived edge of the root-line rule and is outside Math I/A scope. Introduced by the Sol r5 skeleton commits and left unchanged.

- File: `harness_v2/recognition/skeleton.py` around line 481
- A combining overline inside parentheses or a `sqrt(...)` call is kept, while the same stroke on a bare or braced radicand is treated as the root line.
- Repro: `√(x̅)` and `sqrt(x̅)` become `sqrt(overline(x))`. `√x̅` and `\sqrt{x̅}` become `sqrt(x)`.
