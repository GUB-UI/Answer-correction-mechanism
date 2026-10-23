All 12 findings are fixed, with regression coverage:

| Finding | Status / coverage |
|---|---|
| 2 | Fixed — score blinding preserves student math |
| 4 | Fixed — explicit free-tier opt-in, pinned AI Studio endpoint, recorded policy |
| 5 | Fixed — invalid returned maxima rejected |
| 6 | Fixed — malformed rows and invalid rubrics rejected |
| 9 | Fixed — scoped RPC output and explicit request refusals |
| 10 | Fixed — nonblocking reads/writes, deadlines and cancellation |
| 12 | Fixed — text allowlist and image-content rejection |
| 13 | Fixed — exact OCR bytes or traceable derived text with both hashes |
| New MAJOR: instructions | Fixed — trusted grading instructions and response schema |
| New MAJOR: acceptance | Fixed — completed, valid review and item-level agreement required |
| New MAJOR: exceptions | Fixed — completed grading persists after reviewer failure |
| New MINOR: provenance | Fixed — CPU OCR metadata persists |

Added **136 regression cases**. Required combined suite: **311 passed**. External connections were blocked; existing localhost test fixtures ran.

Both change logs are updated. [commit_plan.json](/workspace/pr10_review/commit_plan.json) contains six groups, assigning each changed file once. No git writes or legacy-app changes were made.
