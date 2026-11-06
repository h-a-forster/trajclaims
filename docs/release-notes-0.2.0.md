# trajclaims 0.2.0

Corrects the study after two independent audits. The headline figures and two findings change.
Findings that stand as limits, or were rejected, are in [DESIGN.md](DESIGN.md).

## Findings that changed

1,388 terminus-2 closings were wrongly dropped as having no text: the code ignored a step that sent
`mark_task_complete` beside commands. Another 384 were placeholders counted as silence, and are
now `lost`. These closings claim done far less often than other terminus-2 closings, so leaving
them out made terminus-2 look like an outlier.

| Figure | 0.1.0 | 0.2.0 |
|---|---|---|
| FCR over readable closings | 73.8% | 71.2% |
| Calibrated | 70.1% | 65.9% |
| Rogan-Gladen | 74.8% | 70.3% |
| terminus-2 FCR, raw | 86.8% | 78.7% |
| terminus-2 FCR, calibrated | 79.1% | 68.1% |
| Scaffold share, model share of cell variance | 66.5%, 12.8% | 39.9%, 34.3% |
| Detector AUROC, closing message | 0.74 | 0.75 |

"The scaffold carries more than the model" no longer holds on readable closings, where the
difference is 5.5 points (-14.9 to 30.8). "terminus-2 is the outlier" no longer holds after
calibration.

## Data and labels

- A completion signal beside commands now keeps its step's text, marked `with_command`. Extraction
  schema 2; `labels.jsonl` gains the column.
- The validation sample grows from 573 to 653 messages: 80 from new `terminus-2+command` strata,
  labelled blind with the same prompts. The original 573 redrew identically. `rules-1` is
  unchanged. The new `trajclaims sample --extend` adds strata.
- `study` refuses a sample that differs from a fresh draw, before writing anything.
- Codex `apply_patch` and judy `replace_file` count as edits, which moves 14 `verified_after_edit`
  values.

## Estimators and reporting

- Each closing group has its own correction rates, and Rogan-Gladen is clipped after aggregating,
  not per cell.
- The README reports the pooled direct FCR, the sample's rule-rate gap, precision and recall on
  failed runs, honest failure on adjudicated labels, and the scaffold mix.
- Variance shares use a bias-corrected interval, and the scaffold-minus-model difference has its
  own. Ranges over cells use a simultaneous band, which can reach zero.
- The verification regressions are repeated on tasks with at least 50 trials.

## Reproducibility and tests

- Detector scores are rounded on both the fresh and cached paths, so `summary.json` regenerates
  byte for byte. Their cache is keyed on a hash of the inputs, so a label change cannot reuse
  stale scores.
- `labellers._parse` reads the last JSON object, because a model that miscopies an id corrects
  itself after it.
- CLI paths no longer depend on the working directory.
- Tests now check rates, corrections, the bridge, shares, odds-ratio intervals, validation and the
  tables against hand counts. Each audit mutation fails a test.
