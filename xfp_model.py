#!/usr/bin/env python3
"""
Expected PPR (xFP) model for 2026 fantasy luck.

Trains explainable gradient-boosting models on 2023-2025 nflverse
regular-season play-by-play, then applies them to 2026 Weeks 1-10.

Play models
-----------
Rushes (play_type=run, excluding kneels and two-point tries):
  - E[rushing yards], P(rush TD)
  - Features: yardline_100, down, ydstogo, goal_to_go, shotgun,
    qb_scramble, run_location, run_gap

Targeted throws (non-sack pass attempts; sacks/spikes/two-point excluded):
  - P(catch), E[receiving yards] (0 if incomplete), P(receiving/passing TD),
    P(interception)
  - Features: air_yards, yardline_100, end-zone-target flag
    (air_yards >= yardline_100), down, ydstogo, goal_to_go, shotgun,
    pass_location, pass_length
  Throws with no listed receiver (mostly throwaways) still count for the
  quarterback (yards / TD / INT) but do not create a receiving opportunity.

Scoring (applied to both actual and expected)
---------------------------------------------
  Skill (RB/WR/TE, and any receiving/rushing by a QB):
    0.1 * rush yds + 0.1 * rec yds + 6 * (rush TD + rec TD) + 1 * reception
  Quarterback passing:
    0.04 * pass yds + 4 * pass TD + (-2) * INT
  Fumbles and two-point conversions are omitted (two-point plays are
  dropped; fumbles are not modeled).

Validation
----------
  Holdout: train on 2023-24, evaluate play-level and player-season 2025.
  Production: train on 2023-25, apply to 2026; league-total calibration
  and correlation with ffopportunity PPR xFP.

Data: nflverse/nflfastR play-by-play, CC BY 4.0.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

PBP_COLS = [
    "season", "season_type", "week", "game_id", "game_date", "play_id",
    "play_type", "posteam", "defteam", "yardline_100", "down", "ydstogo",
    "goal_to_go", "shotgun", "no_huddle", "qb_kneel", "qb_spike",
    "qb_scramble", "two_point_attempt", "sack", "rush_attempt",
    "pass_attempt", "complete_pass", "interception", "rush_touchdown",
    "pass_touchdown", "air_yards", "yards_after_catch", "rushing_yards",
    "receiving_yards", "passing_yards", "run_location", "run_gap",
    "pass_location", "pass_length", "rusher_player_id", "rusher_player_name",
    "receiver_player_id", "receiver_player_name", "passer_player_id",
    "passer_player_name",
]

RUSH_NUM = ["yardline_100", "ydstogo", "goal_to_go", "shotgun", "qb_scramble"]
RUSH_CAT = ["down", "run_location", "run_gap"]
PASS_NUM = ["yardline_100", "ydstogo", "goal_to_go", "shotgun", "air_yards", "ez_target"]
PASS_CAT = ["down", "pass_location", "pass_length"]

REG_RUSH = dict(
    max_depth=5, max_iter=300, learning_rate=0.06, min_samples_leaf=80,
    l2_regularization=0.2, early_stopping=True, validation_fraction=0.15,
    n_iter_no_change=20, random_state=42,
)
CLF = dict(
    max_depth=4, max_iter=300, learning_rate=0.06, min_samples_leaf=80,
    l2_regularization=0.2, early_stopping=True, validation_fraction=0.15,
    n_iter_no_change=20, random_state=42,
)


def load_pbp(years) -> pd.DataFrame:
    frames = [pd.read_parquet(DATA / f"play_by_play_{y}.parquet", columns=PBP_COLS) for y in years]
    return pd.concat(frames, ignore_index=True)


def _base_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["season_type"] == "REG"].copy()
    df = df[df["two_point_attempt"].fillna(0).astype(float) == 0]
    return df


def rushes(df: pd.DataFrame) -> pd.DataFrame:
    out = _base_filter(df)
    out = out[
        (out["play_type"] == "run")
        & (out["qb_kneel"].fillna(0).astype(float) == 0)
        & out["rusher_player_id"].notna()
        & out["yardline_100"].notna()
    ].copy()
    out["yardline_100"] = out["yardline_100"].astype(float).clip(1, 99)
    out["ydstogo"] = out["ydstogo"].fillna(10).astype(float)
    out["down"] = out["down"].fillna(0).astype(int).clip(0, 4).astype(str)
    out["goal_to_go"] = out["goal_to_go"].fillna(0).astype(float)
    out["shotgun"] = out["shotgun"].fillna(0).astype(float)
    out["qb_scramble"] = out["qb_scramble"].fillna(0).astype(float)
    out["run_location"] = out["run_location"].fillna("unknown").astype(str)
    out["run_gap"] = out["run_gap"].fillna("unknown").astype(str)
    out["rush_yards"] = out["rushing_yards"].fillna(out["yards_gained"] if "yards_gained" in out.columns else 0).fillna(0).astype(float)
    # rushing_yards already present
    out["rush_yards"] = pd.to_numeric(out["rushing_yards"], errors="coerce").fillna(0.0)
    out["rush_td"] = out["rush_touchdown"].fillna(0).astype(float)
    return out


def throws(df: pd.DataFrame) -> pd.DataFrame:
    out = _base_filter(df)
    out = out[
        (out["play_type"] == "pass")
        & (out["qb_spike"].fillna(0).astype(float) == 0)
        & (out["sack"].fillna(0).astype(float) == 0)
        & out["passer_player_id"].notna()
        & out["yardline_100"].notna()
    ].copy()
    out["yardline_100"] = out["yardline_100"].astype(float).clip(1, 99)
    out["ydstogo"] = out["ydstogo"].fillna(10).astype(float)
    out["down"] = out["down"].fillna(0).astype(int).clip(0, 4).astype(str)
    out["goal_to_go"] = out["goal_to_go"].fillna(0).astype(float)
    out["shotgun"] = out["shotgun"].fillna(0).astype(float)
    out["air_yards"] = pd.to_numeric(out["air_yards"], errors="coerce")
    # rare missing air yards: treat as 0 (throwaway-like)
    out["air_yards"] = out["air_yards"].fillna(0.0)
    out["ez_target"] = (out["air_yards"] >= out["yardline_100"]).astype(float)
    out["pass_location"] = out["pass_location"].fillna("unknown").astype(str)
    out["pass_length"] = out["pass_length"].fillna("unknown").astype(str)
    out["complete"] = out["complete_pass"].fillna(0).astype(float)
    out["rec_yards"] = pd.to_numeric(out["receiving_yards"], errors="coerce").fillna(0.0)
    out["pass_yards"] = pd.to_numeric(out["passing_yards"], errors="coerce").fillna(out["rec_yards"])
    out["pass_td"] = out["pass_touchdown"].fillna(0).astype(float)
    out["interception"] = out["interception"].fillna(0).astype(float)
    out["has_receiver"] = out["receiver_player_id"].notna().astype(int)
    return out


def maps_from_train(series_dict) -> dict:
    maps = {}
    for col, s in series_dict.items():
        cats = sorted(s.astype(str).unique().tolist())
        maps[col] = {c: i for i, c in enumerate(cats)}
    return maps


def encode(df: pd.DataFrame, num_cols, cat_cols, cat_maps) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in num_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in cat_cols:
        mapped = df[c].astype(str).map(cat_maps[c])
        X[c] = mapped.astype(float)  # unseen -> NaN
    return X


def fit_regressor(X, y, cat_idx):
    m = HistGradientBoostingRegressor(**REG_RUSH, categorical_features=cat_idx)
    m.fit(X, y)
    return m


def fit_classifier(X, y, cat_idx):
    m = HistGradientBoostingClassifier(**CLF, categorical_features=cat_idx)
    m.fit(X, y)
    return m


def proba(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


class PlayModels:
    def __init__(self, rush_maps, pass_maps, rush_yards, rush_td, rec_yards, catch, pass_td, intercept):
        self.rush_maps = rush_maps
        self.pass_maps = pass_maps
        self.rush_yards = rush_yards
        self.rush_td = rush_td
        self.rec_yards = rec_yards
        self.catch = catch
        self.pass_td = pass_td
        self.intercept = intercept
        self.rush_cat_idx = [len(RUSH_NUM) + i for i in range(len(RUSH_CAT))]
        self.pass_cat_idx = [len(PASS_NUM) + i for i in range(len(PASS_CAT))]

    def predict_rushes(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        X = encode(out, RUSH_NUM, RUSH_CAT, self.rush_maps)
        yhat = self.rush_yards.predict(X)
        # cannot gain more than the remaining field; allow small losses
        cap = out["yardline_100"].to_numpy()
        out["x_rush_yards"] = np.clip(yhat, -12, cap)
        out["x_rush_td"] = np.clip(proba(self.rush_td, X), 0.0001, 0.999)
        return out

    def predict_throws(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        X = encode(out, PASS_NUM, PASS_CAT, self.pass_maps)
        cap = out["yardline_100"].to_numpy()
        out["x_rec_yards"] = np.clip(self.rec_yards.predict(X), -12, cap)
        out["x_pass_yards"] = out["x_rec_yards"]
        out["x_catch"] = np.clip(proba(self.catch, X), 0.0001, 0.999)
        out["x_pass_td"] = np.clip(proba(self.pass_td, X), 0.0001, 0.999)
        out["x_int"] = np.clip(proba(self.intercept, X), 0.0001, 0.999)
        return out


def train_models(rush_df: pd.DataFrame, throw_df: pd.DataFrame) -> PlayModels:
    rush_maps = maps_from_train({c: rush_df[c] for c in RUSH_CAT})
    pass_maps = maps_from_train({c: throw_df[c] for c in PASS_CAT})
    Xr = encode(rush_df, RUSH_NUM, RUSH_CAT, rush_maps)
    r_cat = [len(RUSH_NUM) + i for i in range(len(RUSH_CAT))]
    rush_yards = fit_regressor(Xr, rush_df["rush_yards"].to_numpy(), r_cat)
    rush_td = fit_classifier(Xr, rush_df["rush_td"].to_numpy().astype(int), r_cat)

    Xt = encode(throw_df, PASS_NUM, PASS_CAT, pass_maps)
    p_cat = [len(PASS_NUM) + i for i in range(len(PASS_CAT))]
    rec_yards = fit_regressor(Xt, throw_df["rec_yards"].to_numpy(), p_cat)
    catch = fit_classifier(Xt, throw_df["complete"].to_numpy().astype(int), p_cat)
    pass_td = fit_classifier(Xt, throw_df["pass_td"].to_numpy().astype(int), p_cat)
    intercept = fit_classifier(Xt, throw_df["interception"].to_numpy().astype(int), p_cat)
    return PlayModels(rush_maps, pass_maps, rush_yards, rush_td, rec_yards, catch, pass_td, intercept)


def cls_metrics(y, p, name):
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "pred_mean": float(p.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
    }
    # decile calibration (absolute gap)
    try:
        q = pd.qcut(p, 10, duplicates="drop")
        cal = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True).mean()
        out["decile_abs_gap_mean"] = float((cal["y"] - cal["p"]).abs().mean())
    except Exception:
        out["decile_abs_gap_mean"] = None
    return out


def reg_metrics(y, yhat, name):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return {
        "n": int(len(y)),
        "actual_mean": float(y.mean()),
        "pred_mean": float(yhat.mean()),
        "mae": float(mean_absolute_error(y, yhat)),
        "rmse": float(math.sqrt(mean_squared_error(y, yhat))),
        "r2": float(r2_score(y, yhat)),
    }


def play_level_report(rushes_p, throws_p) -> dict:
    return {
        "rush_yards": reg_metrics(rushes_p["rush_yards"], rushes_p["x_rush_yards"], "rush_yards"),
        "rush_td": cls_metrics(rushes_p["rush_td"], rushes_p["x_rush_td"], "rush_td"),
        "rec_yards": reg_metrics(throws_p["rec_yards"], throws_p["x_rec_yards"], "rec_yards"),
        "catch": cls_metrics(throws_p["complete"], throws_p["x_catch"], "catch"),
        "pass_td": cls_metrics(throws_p["pass_td"], throws_p["x_pass_td"], "pass_td"),
        "interception": cls_metrics(throws_p["interception"], throws_p["x_int"], "int"),
    }


def roster_lookup(path: Path) -> pd.DataFrame:
    ros = pd.read_parquet(path, columns=["week", "gsis_id", "full_name", "position"])
    ros = ros[ros["gsis_id"].notna()].copy()
    pos = (
        ros.dropna(subset=["position"])
        .groupby(["gsis_id", "position"])
        .size()
        .reset_index(name="n")
        .sort_values(["gsis_id", "n"], ascending=[True, False])
        .groupby("gsis_id")
        .head(1)[["gsis_id", "position"]]
    )
    name = ros.sort_values("week").groupby("gsis_id", as_index=False).tail(1)[["gsis_id", "full_name"]]
    out = name.merge(pos, on="gsis_id", how="left").rename(columns={"gsis_id": "player_id"})
    out["position"] = out["position"].replace({"FB": "RB"})
    return out


def team_label(rows: pd.DataFrame, id_col: str) -> pd.Series:
    """Most-frequent posteam; if two+ teams, 'A/B' in chronological order."""
    tmp = (
        rows.groupby([id_col, "posteam"])
        .agg(n=("week", "size"), first_week=("week", "min"), last_week=("week", "max"))
        .reset_index()
        .sort_values([id_col, "first_week"])
    )
    def _one(g):
        teams = g.sort_values("first_week")["posteam"].tolist()
        # unique preserve order
        seen = []
        for t in teams:
            if t not in seen:
                seen.append(t)
        return "/".join(seen)
    return tmp.groupby(id_col).apply(_one, include_groups=False).rename("team")


def build_credits(rushes_p: pd.DataFrame, throws_p: pd.DataFrame) -> pd.DataFrame:
    r = rushes_p.copy()
    r_cred = pd.DataFrame({
        "season": r["season"], "week": r["week"], "game_id": r["game_id"],
        "posteam": r["posteam"], "defteam": r["defteam"],
        "player_id": r["rusher_player_id"], "player_name_pbp": r["rusher_player_name"],
        "rush_att": 1, "rush_yards": r["rush_yards"], "x_rush_yards": r["x_rush_yards"],
        "rush_td": r["rush_td"], "x_rush_td": r["x_rush_td"],
        "targets": 0, "receptions": 0, "x_rec": 0.0,
        "rec_yards": 0.0, "x_rec_yards": 0.0, "rec_td": 0.0, "x_rec_td": 0.0,
        "pass_att": 0, "pass_yards": 0.0, "x_pass_yards": 0.0,
        "pass_td": 0.0, "x_pass_td": 0.0, "ints": 0.0, "x_int": 0.0,
    })

    t = throws_p.copy()
    qb = pd.DataFrame({
        "season": t["season"], "week": t["week"], "game_id": t["game_id"],
        "posteam": t["posteam"], "defteam": t["defteam"],
        "player_id": t["passer_player_id"], "player_name_pbp": t["passer_player_name"],
        "rush_att": 0, "rush_yards": 0.0, "x_rush_yards": 0.0,
        "rush_td": 0.0, "x_rush_td": 0.0,
        "targets": 0, "receptions": 0, "x_rec": 0.0,
        "rec_yards": 0.0, "x_rec_yards": 0.0, "rec_td": 0.0, "x_rec_td": 0.0,
        "pass_att": 1, "pass_yards": t["pass_yards"], "x_pass_yards": t["x_pass_yards"],
        "pass_td": t["pass_td"], "x_pass_td": t["x_pass_td"],
        "ints": t["interception"], "x_int": t["x_int"],
    })

    rec_mask = t["has_receiver"] == 1
    tr = t[rec_mask]
    rec = pd.DataFrame({
        "season": tr["season"], "week": tr["week"], "game_id": tr["game_id"],
        "posteam": tr["posteam"], "defteam": tr["defteam"],
        "player_id": tr["receiver_player_id"], "player_name_pbp": tr["receiver_player_name"],
        "rush_att": 0, "rush_yards": 0.0, "x_rush_yards": 0.0,
        "rush_td": 0.0, "x_rush_td": 0.0,
        "targets": 1, "receptions": tr["complete"], "x_rec": tr["x_catch"],
        "rec_yards": tr["rec_yards"], "x_rec_yards": tr["x_rec_yards"],
        "rec_td": tr["pass_td"], "x_rec_td": tr["x_pass_td"],
        "pass_att": 0, "pass_yards": 0.0, "x_pass_yards": 0.0,
        "pass_td": 0.0, "x_pass_td": 0.0, "ints": 0.0, "x_int": 0.0,
    })
    return pd.concat([r_cred, qb, rec], ignore_index=True)


def score_rows(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["fp_rush"] = 0.1 * g["rush_yards"] + 6.0 * g["rush_td"]
    g["xfp_rush"] = 0.1 * g["x_rush_yards"] + 6.0 * g["x_rush_td"]
    g["fp_rec"] = 0.1 * g["rec_yards"] + 6.0 * g["rec_td"] + 1.0 * g["receptions"]
    g["xfp_rec"] = 0.1 * g["x_rec_yards"] + 6.0 * g["x_rec_td"] + 1.0 * g["x_rec"]
    g["fp_pass"] = 0.04 * g["pass_yards"] + 4.0 * g["pass_td"] - 2.0 * g["ints"]
    g["xfp_pass"] = 0.04 * g["x_pass_yards"] + 4.0 * g["x_pass_td"] - 2.0 * g["x_int"]
    g["fp"] = g["fp_rush"] + g["fp_rec"] + g["fp_pass"]
    g["xfp"] = g["xfp_rush"] + g["xfp_rec"] + g["xfp_pass"]
    g["td"] = g["rush_td"] + g["rec_td"] + g["pass_td"]
    g["xtd"] = g["x_rush_td"] + g["x_rec_td"] + g["x_pass_td"]
    # luck components (sum to fp - xfp)
    g["luck_td"] = 6.0 * (g["rush_td"] - g["x_rush_td"]) + 6.0 * (g["rec_td"] - g["x_rec_td"]) + 4.0 * (g["pass_td"] - g["x_pass_td"])
    g["luck_yd"] = 0.1 * (g["rush_yards"] - g["x_rush_yards"]) + 0.1 * (g["rec_yards"] - g["x_rec_yards"]) + 0.04 * (g["pass_yards"] - g["x_pass_yards"])
    g["luck_rec"] = 1.0 * (g["receptions"] - g["x_rec"])
    g["luck_int"] = -2.0 * (g["ints"] - g["x_int"])
    g["luck_fp"] = g["fp"] - g["xfp"]
    g["luck_td_count"] = g["td"] - g["xtd"]
    return g


AGG = {
    "rush_att": "sum", "rush_yards": "sum", "x_rush_yards": "sum",
    "rush_td": "sum", "x_rush_td": "sum",
    "targets": "sum", "receptions": "sum", "x_rec": "sum",
    "rec_yards": "sum", "x_rec_yards": "sum", "rec_td": "sum", "x_rec_td": "sum",
    "pass_att": "sum", "pass_yards": "sum", "x_pass_yards": "sum",
    "pass_td": "sum", "x_pass_td": "sum", "ints": "sum", "x_int": "sum",
    "player_name_pbp": "first", "posteam": "last", "defteam": "last",
}


def aggregate(credits: pd.DataFrame, roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = credits.groupby(["player_id", "season", "week", "game_id"], as_index=False).agg(AGG)
    weekly = score_rows(weekly)
    # season
    sag = {k: "sum" for k in AGG if k not in ("player_name_pbp", "posteam", "defteam")}
    sag["player_name_pbp"] = "first"
    season = weekly.groupby(["player_id", "season"], as_index=False).agg(
        **{k: (k, fn) for k, fn in sag.items()},
        games=("game_id", "nunique"),
        weeks=("week", "nunique"),
    )
    season = score_rows(season)
    teams = team_label(credits, "player_id")
    season = season.merge(teams, left_on="player_id", right_index=True, how="left")
    weekly = weekly.merge(teams, left_on="player_id", right_index=True, how="left")
    season = season.merge(roster, on="player_id", how="left")
    weekly = weekly.merge(roster, on="player_id", how="left")
    season["player_name"] = season["full_name"].fillna(season["player_name_pbp"])
    weekly["player_name"] = weekly["full_name"].fillna(weekly["player_name_pbp"])
    season["position"] = season["position"].fillna("UNK")
    weekly["position"] = weekly["position"].fillna("UNK")
    season["looks"] = season["rush_att"] + season["targets"]
    weekly["looks"] = weekly["rush_att"] + weekly["targets"]
    # opportunities used in the explorer
    season["opps"] = np.where(
        season["position"] == "QB",
        season["pass_att"],
        season["rush_att"] + season["targets"],
    )
    weekly["opps"] = np.where(
        weekly["position"] == "QB",
        weekly["pass_att"],
        weekly["rush_att"] + weekly["targets"],
    )
    season["fp_pg"] = season["fp"] / season["games"].clip(lower=1)
    season["xfp_pg"] = season["xfp"] / season["games"].clip(lower=1)
    return season, weekly


def corr(a, b) -> float | None:
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 5 or s["a"].std() == 0 or s["b"].std() == 0:
        return None
    return float(s["a"].corr(s["b"]))


def main():
    print("Loading PBP 2023-2026...")
    pbp_all = load_pbp([2023, 2024, 2025, 2026])
    print(f"  total rows {len(pbp_all):,}")

    # ----- holdout: train 2023-24, test 2025 -----
    print("\n=== HOLDOUT: train 2023-24, evaluate 2025 ===")
    train_r = rushes(pbp_all[pbp_all.season.isin([2023, 2024])])
    train_t = throws(pbp_all[pbp_all.season.isin([2023, 2024])])
    test_r = rushes(pbp_all[pbp_all.season == 2025])
    test_t = throws(pbp_all[pbp_all.season == 2025])
    print(f"  train rushes {len(train_r):,} throws {len(train_t):,}")
    print(f"  test  rushes {len(test_r):,} throws {len(test_t):,}")
    holdout_models = train_models(train_r, train_t)
    test_r_p = holdout_models.predict_rushes(test_r)
    test_t_p = holdout_models.predict_throws(test_t)
    holdout_play = play_level_report(test_r_p, test_t_p)
    print(json.dumps(holdout_play, indent=2))

    ros25 = roster_lookup(DATA / "roster_2025.parquet")
    cred25 = build_credits(test_r_p, test_t_p)
    season25, _ = aggregate(cred25, ros25)
    skill25 = season25[season25.position.isin(["QB", "RB", "WR", "TE"])].copy()
    holdout_player = {
        "n_players": int(len(skill25)),
        "corr_fp_xfp_all": corr(skill25.fp, skill25.xfp),
        "mae_fp_all": float((skill25.fp - skill25.xfp).abs().mean()),
        "by_position": {},
    }
    for pos, min_opp in [("QB", 150), ("RB", 40), ("WR", 25), ("TE", 25)]:
        sub = skill25[(skill25.position == pos) & (skill25.opps >= min_opp)]
        holdout_player["by_position"][pos] = {
            "n": int(len(sub)),
            "min_opps": min_opp,
            "corr_fp_xfp": corr(sub.fp, sub.xfp),
            "mae_fp": float((sub.fp - sub.xfp).abs().mean()) if len(sub) else None,
            "fp_mean": float(sub.fp.mean()) if len(sub) else None,
            "xfp_mean": float(sub.xfp.mean()) if len(sub) else None,
        }
    print("player-season 2025:", json.dumps(holdout_player, indent=2))

    # ----- production: train 2023-25, apply 2026 -----
    print("\n=== PRODUCTION: train 2023-25, apply 2026 ===")
    prod_r = rushes(pbp_all[pbp_all.season.isin([2023, 2024, 2025])])
    prod_t = throws(pbp_all[pbp_all.season.isin([2023, 2024, 2025])])
    print(f"  train rushes {len(prod_r):,} throws {len(prod_t):,}")
    models = train_models(prod_r, prod_t)
    r26 = models.predict_rushes(rushes(pbp_all[pbp_all.season == 2026]))
    t26 = models.predict_throws(throws(pbp_all[pbp_all.season == 2026]))
    play26 = play_level_report(r26, t26)
    print("2026 play-level calibration (in-season application, not a holdout):")
    print(json.dumps(play26, indent=2))

    ros26 = roster_lookup(DATA / "roster_2026.parquet")
    cred26 = build_credits(r26, t26)
    season26, weekly26 = aggregate(cred26, ros26)
    skill26 = season26[season26.position.isin(["QB", "RB", "WR", "TE"])].copy()
    weekly_skill = weekly26[weekly26.position.isin(["QB", "RB", "WR", "TE"])].copy()

    # league totals from play level (no double-count)
    league = {
        "games": int(pbp_all.loc[pbp_all.season == 2026, "game_id"].nunique()),
        "date_min": str(pbp_all.loc[pbp_all.season == 2026, "game_date"].min()),
        "date_max": str(pbp_all.loc[pbp_all.season == 2026, "game_date"].max()),
        "weeks": sorted(int(w) for w in pbp_all.loc[pbp_all.season == 2026, "week"].unique()),
        "rush_yards_actual": float(r26.rush_yards.sum()),
        "rush_yards_exp": float(r26.x_rush_yards.sum()),
        "rush_td_actual": float(r26.rush_td.sum()),
        "rush_td_exp": float(r26.x_rush_td.sum()),
        "rec_yards_actual": float(t26.rec_yards.sum()),
        "rec_yards_exp": float(t26.x_rec_yards.sum()),
        "receptions_actual": float(t26.complete.sum()),
        "receptions_exp": float(t26.x_catch.sum()),
        "pass_td_actual": float(t26.pass_td.sum()),
        "pass_td_exp": float(t26.x_pass_td.sum()),
        "int_actual": float(t26.interception.sum()),
        "int_exp": float(t26.x_int.sum()),
    }
    league["rush_yards_diff"] = league["rush_yards_actual"] - league["rush_yards_exp"]
    league["rush_td_diff"] = league["rush_td_actual"] - league["rush_td_exp"]
    league["rec_yards_diff"] = league["rec_yards_actual"] - league["rec_yards_exp"]
    league["receptions_diff"] = league["receptions_actual"] - league["receptions_exp"]
    league["pass_td_diff"] = league["pass_td_actual"] - league["pass_td_exp"]
    league["int_diff"] = league["int_actual"] - league["int_exp"]
    print("2026 league totals:", json.dumps(league, indent=2))

    # ffopportunity comparison
    ep = pd.read_parquet(DATA / "ep_weekly_2026.parquet")
    ep_s = (
        ep[ep.position.isin(["QB", "RB", "WR", "TE"])]
        .groupby(["player_id", "full_name", "position"], as_index=False)
        .agg(ff_fp=("total_fantasy_points", "sum"), ff_xfp=("total_fantasy_points_exp", "sum"))
    )
    merged = skill26.merge(ep_s[["player_id", "ff_fp", "ff_xfp"]], on="player_id", how="inner")
    ff_cmp = {
        "n_merged": int(len(merged)),
        "corr_our_xfp_ff_xfp": corr(merged.xfp, merged.ff_xfp),
        "corr_our_fp_ff_fp": corr(merged.fp, merged.ff_fp),
        "corr_our_luck_ff_luck": corr(merged.fp - merged.xfp, merged.ff_fp - merged.ff_xfp),
        "corr_our_xfp_actual_fp": corr(merged.xfp, merged.fp),
        "mae_our_xfp_vs_ff_xfp": float((merged.xfp - merged.ff_xfp).abs().mean()),
        "by_position": {},
    }
    for pos, min_opp in [("QB", 150), ("RB", 40), ("WR", 25), ("TE", 25)]:
        sub = merged[(merged.position == pos) & (merged.opps >= min_opp)]
        ff_cmp["by_position"][pos] = {
            "n": int(len(sub)),
            "corr_xfp": corr(sub.xfp, sub.ff_xfp),
            "corr_luck": corr(sub.fp - sub.xfp, sub.ff_fp - sub.ff_xfp),
            "corr_xfp_actual": corr(sub.xfp, sub.fp),
        }
    print("ffopportunity comparison:", json.dumps(ff_cmp, indent=2))

    # qualified leaderboards for the report
    def top_bottom(df, pos, min_opp, n=5):
        sub = df[(df.position == pos) & (df.opps >= min_opp)].sort_values("luck_fp", ascending=False)
        cols = [
            "player_name", "team", "games", "opps", "fp", "xfp", "luck_fp",
            "td", "xtd", "luck_td_count", "luck_td", "luck_yd", "luck_rec", "luck_int",
            "fp_pg", "xfp_pg",
        ]
        return {
            "luckiest": sub.head(n)[cols].to_dict(orient="records"),
            "unluckiest": sub.sort_values("luck_fp").head(n)[cols].to_dict(orient="records"),
        }

    boards = {
        "QB": top_bottom(skill26, "QB", 150),
        "RB": top_bottom(skill26, "RB", 40),
        "WR": top_bottom(skill26, "WR", 25),
        "TE": top_bottom(skill26, "TE", 25),
    }

    pbp_ts = json.loads((DATA / "pbp_timestamp.json").read_text())
    validation = {
        "pbp_timestamp": pbp_ts,
        "holdout_2025_play_level": holdout_play,
        "holdout_2025_player_season": holdout_player,
        "production_train_rushes": int(len(prod_r)),
        "production_train_throws": int(len(prod_t)),
        "applied_2026_play_level": play26,
        "league_2026": league,
        "ffopportunity": ff_cmp,
        "features": {
            "rush": {"numeric": RUSH_NUM, "categorical": RUSH_CAT},
            "pass": {"numeric": PASS_NUM, "categorical": PASS_CAT},
        },
        "scoring": {
            "rush_rec_ppr": "0.1/yard, 6/TD, 1/reception",
            "pass": "0.04/pass yard, 4/pass TD, -2/INT",
            "xINT": "included",
            "omitted": "fumbles, two-point conversions",
        },
        "coverage": "2026 REG weeks 1-10, 149 games, through Sunday Nov 15, 2026. LAC @ BAL MNF Nov 16 not included. Week 10 byes: CHI, DEN, PHI, TB.",
        "leaderboards": boards,
    }
    # make json-safe
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if (isinstance(o, float) and (math.isnan(o) or math.isinf(o))) else float(o)
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list):
            return [conv(v) for v in o]
        if pd.isna(o) if not isinstance(o, (list, dict, str)) else False:
            return None
        return o

    validation = conv(validation)
    (OUT / "xfp_validation.json").write_text(json.dumps(validation, indent=2))
    skill26.to_csv(OUT / "xfp_player_season_2026.csv", index=False)
    weekly_skill.to_csv(OUT / "xfp_player_week_2026.csv", index=False)
    season25.to_csv(OUT / "xfp_player_season_2025_holdout.csv", index=False)

    # compact JSON for the explorer
    players = []
    keep = skill26.sort_values("fp", ascending=False)
    for _, r in keep.iterrows():
        players.append({
            "id": r.player_id,
            "name": r.player_name,
            "team": r.team if pd.notna(r.team) else "",
            "pos": r.position,
            "g": int(r.games),
            "opps": int(r.opps),
            "rushes": int(r.rush_att),
            "targets": int(r.targets),
            "att": int(r.pass_att),
            "fp": round(float(r.fp), 2),
            "xfp": round(float(r.xfp), 2),
            "dfp": round(float(r.luck_fp), 2),
            "fpg": round(float(r.fp_pg), 2),
            "xfpg": round(float(r.xfp_pg), 2),
            "td": round(float(r.td), 2),
            "xtd": round(float(r.xtd), 2),
            "dtd": round(float(r.luck_td_count), 2),
            "ltd": round(float(r.luck_td), 2),
            "lyd": round(float(r.luck_yd), 2),
            "lrec": round(float(r.luck_rec), 2),
            "lint": round(float(r.luck_int), 2),
            "ry": round(float(r.rush_yards), 1),
            "xry": round(float(r.x_rush_yards), 1),
            "rtd": round(float(r.rush_td), 2),
            "xrtd": round(float(r.x_rush_td), 2),
            "recy": round(float(r.rec_yards), 1),
            "xrecy": round(float(r.x_rec_yards), 1),
            "rec": round(float(r.receptions), 2),
            "xrec": round(float(r.x_rec), 2),
            "rec_td": round(float(r.rec_td), 2),
            "xrec_td": round(float(r.x_rec_td), 2),
            "py": round(float(r.pass_yards), 1),
            "xpy": round(float(r.x_pass_yards), 1),
            "ptd": round(float(r.pass_td), 2),
            "xptd": round(float(r.x_pass_td), 2),
            "ints": round(float(r.ints), 2),
            "xints": round(float(r.x_int), 2),
        })

    weeks = []
    for _, r in weekly_skill.iterrows():
        weeks.append({
            "id": r.player_id,
            "w": int(r.week),
            "opp": r.defteam if pd.notna(r.defteam) else "",
            "fp": round(float(r.fp), 2),
            "xfp": round(float(r.xfp), 2),
            "td": round(float(r.td), 2),
            "xtd": round(float(r.xtd), 2),
        })

    explorer = {
        "meta": {
            "title": "2026 Fantasy Luck Explorer",
            "dataThrough": "Sunday, November 15, 2026",
            "dataThroughNote": "2026 regular season, Weeks 1–10, 149 games. Week 10 Monday Night Football (Chargers at Ravens, Nov 16) is not included. Chicago, Denver, Philadelphia, and Tampa Bay were on bye in Week 10.",
            "pbpUpdated": "November 16, 2026, 1:32 PM America/Chicago",
            "scoring": "PPR for rush/rec (0.1/yard, 6/TD, 1/reception). Passing: 0.04/yard, 4/TD, −2/INT. Expected interceptions are included. Fumbles and two-point conversions are omitted.",
            "method": "Expected points come from histogram gradient-boosting models trained on 2023–2025 regular-season nflverse play-by-play. Each rush gets expected rushing yards and a rush-TD probability from field position, down, distance, goal-to-go, shotgun, scramble flag, and run location/gap. Each targeted throw gets an expected catch, expected receiving yards, a TD probability, and an interception probability from air yards, field position, an end-zone-target flag, down, distance, shotgun, and pass location/length. A league-average play in the same situation is the baseline — this is not a player-talent model. Production models were trained on 2023–25; a 2023–24 train / 2025 holdout was used as a sanity check.",
            "validation": {
                "holdout2025_corr_fp_xfp": holdout_player["corr_fp_xfp_all"],
                "holdout2025_by_pos": {k: v.get("corr_fp_xfp") for k, v in holdout_player["by_position"].items()},
                "league2026": {
                    "rush_yd": [league["rush_yards_actual"], league["rush_yards_exp"]],
                    "rec_yd": [league["rec_yards_actual"], league["rec_yards_exp"]],
                    "rec": [league["receptions_actual"], league["receptions_exp"]],
                    "rush_td": [league["rush_td_actual"], league["rush_td_exp"]],
                    "pass_td": [league["pass_td_actual"], league["pass_td_exp"]],
                    "int": [league["int_actual"], league["int_exp"]],
                },
                "corr_ffopportunity_xfp": ff_cmp["corr_our_xfp_ff_xfp"],
                "corr_ffopportunity_luck": ff_cmp["corr_our_luck_ff_luck"],
                "corr_our_xfp_actual": ff_cmp["corr_our_xfp_actual_fp"],
            },
        },
        "players": players,
        "weeks": weeks,
    }
    explorer = conv(explorer)
    (OUT / "explorer_data.json").write_text(json.dumps(explorer, separators=(",", ":")))
    print(f"\nWrote season/week CSVs, validation, and explorer_data.json ({len(players)} players, {len(weeks)} week rows)")
    print("Done training/aggregation.")


if __name__ == "__main__":
    main()
