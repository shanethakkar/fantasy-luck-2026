# 2026 Fantasy Luck Explorer

An interactive look at which NFL players have outproduced — or underproduced — their opportunity so far in the 2026 season, using a transparent expected-PPR (xPPR / xFP) model and a simpler expected-touchdown (xTD) model. Both are built from public [nflverse](https://github.com/nflverse) play-by-play. A league-average play in the same situation is the baseline. This is not a player-talent rating.

Open [`index.html`](index.html) in a browser (or [`output/fantasy_luck_2026.html`](output/fantasy_luck_2026.html)) to use the explorer. It opens with the five players running hottest and coldest, then a luck-vs-opportunity chart (or actual vs expected), a player panel that splits each player's luck into TD / yards / catch / INT parts with a week-by-week chart, and a sortable leaderboard. Filters cover position, PPR or touchdowns, a minimum-opportunity slider and name search. It works on phones (the player panel opens as a bottom sheet) and has light and dark themes.

The page is built by `build_explorer.py`, which embeds `output/explorer_data.json` into [`templates/explorer.html`](templates/explorer.html). Edit the template to change the layout or styling, then run `python build_explorer.py`.

GitHub Pages is not enabled on this repository. You can turn it on later if you want the explorer served from the repo.

## Data coverage

**2026 regular season, Weeks 1–10, 149 games, through Sunday, November 15, 2026.** Week 10 Monday Night Football (Los Angeles Chargers at Baltimore Ravens, November 16) is not included. Chicago, Denver, Philadelphia, and Tampa Bay were on bye in Week 10, so those teams are through Week 9 only.

The nflverse play-by-play snapshot used for the published outputs was last updated **November 16, 2026, 1:32 PM America/Chicago** (`2026-11-16 14:32:22 EST`). The ffopportunity comparison file timestamp is `2026-11-16 14:00:30` (timezone not specified in that file).

## Scoring

Applied to both actual and expected stats:

- Rushing / receiving (PPR): **0.1 per yard, 6 per touchdown, 1 per reception**
- Passing: **0.04 per passing yard, 4 per passing touchdown, −2 per interception**
- Quarterback rushing is scored with the rush model and the PPR rush rules
- Expected interceptions are included
- Fumbles and two-point conversions are omitted (two-point plays are dropped; fumbles are not modeled)

Luck components are constructed so they add to `FP − xFP`:

- **TD luck:** 6 points per rushing or receiving TD over expected, 4 per passing TD
- **Yards luck:** 0.1 per rush or receiving yard over expected, 0.04 per passing yard
- **Catch luck:** 1 PPR point per reception over expected
- **INT luck:** −2 per interception over expected (fewer INTs than expected is positive luck)

## Models

### Expected touchdowns (first pass)

`xtd_first_pass.py` estimates TD probability from 2023–2025 regular-season historical rates, then applies those rates to 2026 opportunities.

- **Rushes:** P(rush TD) by yard line. Exact yard line for the 1–20; longer rushes are pooled into distance bins. Sparse bins are partially pooled toward a coarser parent bin.
- **Targets:** Split into end-zone targets (`air_yards >= yardline_100`) and all other targeted throws (sacks excluded). Non-end-zone throws use the same yard-line scheme as rushes; end-zone targets use coarser bins because deep end-zone shots are rare.

This is a simple, fully auditable yard-line / end-zone-target model. It does not use down, distance, or pass/run location.

### Expected PPR (full model)

`xfp_model.py` trains histogram gradient-boosting models (scikit-learn `HistGradientBoostingRegressor` / `Classifier`) on 2023–2025 regular-season plays and applies them to 2026.

**Rushes** (kneels and two-point tries excluded):

- Outcomes: expected rushing yards, P(rush TD)
- Features: `yardline_100`, down, distance (`ydstogo`), goal-to-go, shotgun, scramble flag, run location, run gap

**Throws** (non-sack pass attempts; spikes and two-point tries excluded):

- Outcomes: P(catch), expected receiving yards (0 if incomplete), P(TD), P(interception)
- Features: air yards, `yardline_100`, end-zone-target flag, down, distance, goal-to-go, shotgun, pass location, pass length
- Throws with no listed receiver (mostly throwaways) still count for the quarterback but do not create a receiving opportunity

The production models were trained on 2023–25. A 2023–24 train / 2025 holdout is the sanity check reported below. Play-level yardage R² is deliberately low: situation does not explain a single play’s result, and the leftover is the luck (and talent) the explorer is meant to surface. Calibration of means, and player-season correlation of xFP with actual PPR, are the relevant checks.

## Validation

From `output/xfp_validation.json`, produced by the published run.

**2025 holdout** (trained on 2023–24 only):

- Play-level predicted means are close to actual means (rush yards 4.43 vs 4.52; receiving yards 7.12 vs 7.04; catch rate 65.0% vs 64.6%; rush TD 3.53% vs 3.60%; pass TD 4.51% vs 4.67%; INT 2.25% vs 2.19%). Decile calibration gaps for the classification models are about 0.003–0.012.
- Player-season correlation of xFP vs actual PPR: **0.986** overall. Qualified players: QB **0.978** (min 150 pass attempts), RB **0.984** (min 40 looks), WR **0.962** (min 25 targets + rushes), TE **0.965** (min 25).
- Caveat on that holdout: 2025 tight ends ran ahead of a 2023–24-trained model (mean 112.6 actual PPR vs 101.0 expected). The production model includes 2025.

**2026 league totals**, actual vs expected (production model, not a holdout):

| Stat | Actual | Expected |
| --- | ---: | ---: |
| Rush yards | 33,775 | 34,273 |
| Rush TDs | 282 | 289.0 |
| Receiving yards | 69,778 | 69,701 |
| Receptions | 6,274 | 6,312 |
| Passing TDs | 458 | 437.0 |
| Interceptions | 219 | 221.1 |

Passing touchdowns in 2026 are running a bit ahead of the 2023–25 baseline (about 5%). Rush yards are about 1.5% behind.

**Agreement with ffopportunity** expected PPR on 2026 (534 skill-position players): correlation of our xFP with ffopportunity xFP is **0.997**; luck residuals (`FP − xFP`) correlate **0.923**; our xFP vs actual FP correlates **0.975**. Mean absolute gap between the two xFP series is **2.6** PPR points.

The first-pass xTD model’s (TD − xTD) rankings also agree closely with ffopportunity’s xTD residuals (correlation about 0.96 / 0.95 / 0.92 for qualified RBs, WR/TEs, and QBs).

## Caveats

- Nine or ten games is a small sample. Touchdown variance is large. Over- or under-performance is a tendency, not a prediction.
- The models use situation only (field position, down, distance, location, and similar). They do not use player identity, coverage, blocking, or quarterback quality. A “lucky” star can still be genuinely better at finishing.
- Usage is not luck. A large number of end-zone targets is opportunity earned.
- Two-point conversions and fumbles are omitted from both actual and expected PPR, so totals will not match a box-score PPR source that includes them.
- 2026 passing touchdowns are running slightly hot relative to 2023–25, so the luckiest quarterbacks are somewhat inflated by the league environment.
- Week 10 is incomplete until Monday Night Football is in the file. Role changes (for example a midseason quarterback move) are labeled with both teams when they appear in the play-by-play.
- Opportunity filters matter. The explorer defaults to 150 pass attempts for quarterbacks, 40 looks for running backs, and 25 targets + rushes for wide receivers and tight ends.

## How to rebuild

Python 3.11+ is recommended. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_data.py            # public nflverse + ffopportunity files into data/
python xtd_first_pass.py           # yard-line xTD model and leaderboards
python xfp_model.py                # xPPR models, validation, explorer_data.json
python build_explorer.py           # writes output/fantasy_luck_2026.html and index.html
```

`download_data.py` skips files that already exist. Pass `--force` to re-download. Raw parquet play-by-play is gitignored (those files are larger than this repo and are redistributed by nflverse). The committed `output/` CSVs, `xfp_validation.json`, `run_meta.json`, and HTML reflect the November 16, 2026 snapshot described above.

## Data credit and license

Code in this repository is released under the [MIT License](LICENSE).

Play-by-play, rosters, and related files come from [nflverse](https://github.com/nflverse) / [nflfastR](https://www.nflfastr.com/), published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The comparison expected-PPR series comes from [ffverse / ffopportunity](https://github.com/ffverse/ffopportunity). If you use or republish these numbers, please credit those projects and this repo as appropriate.

NFL, team names, and player names are used for identification. This project is not affiliated with the National Football League.
