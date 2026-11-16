#!/usr/bin/env python3
"""
Preliminary expected-TD (xTD) model for the 2026 NFL regular season.

Method
------
Train historical touchdown rates on 2023-2025 regular-season plays, then apply
those rates to 2026 regular-season plays.

A league-average play from the same situation is the baseline. This is not a
player-talent or quarterback-quality model.

Rushes
    P(rush TD) by yardline_100. Exact yard line for the 1-20 (ample historical
    volume). Beyond that, yard lines are pooled into 21-25, 26-30, 31-40,
    41-50, 51-60, 61-80, 81-99 so sparse long-distance rushes do not get
    noisy rates.

Targets / targeted throws (sacks excluded)
    Split into end-zone targets (air_yards >= yardline_100) and all other
    targeted throws.
    Non-EZ: same yard-line scheme as rushes.
    EZ targets: coarser bins (1, 2, 3, 4, 5, 6-7, 8-10, 11-15, 16-20,
    21-25, 26-30, 31-40, 41+) because deep end-zone shots are rare.
    If a bin has fewer than 30 training plays, its rate is partially pooled
    toward the parent (next-coarser) bin.

Exclusions: two-point tries, kneels, spikes, sacks, play_type not in
{run, pass}, missing yard line or player id.

Attribution
    Rush xTD / TDs -> rusher; receiving xTD / TDs -> targeted receiver;
    passing xTD / TDs -> passer.

Data: nflverse/nflfastR play-by-play, CC BY 4.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

PBP_COLS = [
    "season", "season_type", "week", "game_id", "game_date", "play_id",
    "play_type", "posteam", "yardline_100", "air_yards", "down", "ydstogo",
    "goal_to_go", "qb_kneel", "qb_spike", "qb_scramble", "two_point_attempt",
    "sack", "rush_attempt", "pass_attempt", "complete_pass", "rush_touchdown",
    "pass_touchdown", "touchdown", "rusher_player_id", "rusher_player_name",
    "receiver_player_id", "receiver_player_name", "passer_player_id",
    "passer_player_name",
]

# Bins used when we are not on an exact 1-20 yard line.
RUSH_AND_NONEZ_BINS = [
    (21, 25, "21-25"),
    (26, 30, "26-30"),
    (31, 40, "31-40"),
    (41, 50, "41-50"),
    (51, 60, "51-60"),
    (61, 80, "61-80"),
    (81, 99, "81-99"),
]

EZ_BINS = [
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 3, "3"),
    (4, 4, "4"),
    (5, 5, "5"),
    (6, 7, "6-7"),
    (8, 10, "8-10"),
    (11, 15, "11-15"),
    (16, 20, "16-20"),
    (21, 25, "21-25"),
    (26, 30, "26-30"),
    (31, 40, "31-40"),
    (41, 99, "41+"),
]

MIN_N = 30


def load_pbp(years) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(DATA / f"play_by_play_{y}.parquet", columns=PBP_COLS) for y in years],
        ignore_index=True,
    )


def clean_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["season_type"] == "REG"]
    df = df[df["play_type"].isin(["run", "pass"])]
    df = df[df["two_point_attempt"].fillna(0).astype(float) == 0]
    df = df[df["qb_kneel"].fillna(0).astype(float) == 0]
    df = df[df["qb_spike"].fillna(0).astype(float) == 0]
    df = df[df["yardline_100"].notna()]
    df["yardline_100"] = df["yardline_100"].astype(int).clip(1, 99)
    df["rush_touchdown"] = df["rush_touchdown"].fillna(0).astype(float)
    df["pass_touchdown"] = df["pass_touchdown"].fillna(0).astype(float)
    df["air_yards"] = pd.to_numeric(df["air_yards"], errors="coerce")
    df["ez_target"] = np.where(
        df["air_yards"].notna(),
        (df["air_yards"] >= df["yardline_100"]).astype(int),
        0,
    )
    return df


def bin_label(yl: int, scheme: str) -> str:
    if scheme == "rush_nonez":
        if yl <= 20:
            return str(yl)
        for lo, hi, lab in RUSH_AND_NONEZ_BINS:
            if lo <= yl <= hi:
                return lab
    else:
        for lo, hi, lab in EZ_BINS:
            if lo <= yl <= hi:
                return lab
    raise ValueError(f"No bin for yl={yl} scheme={scheme}")


def parent_label(label: str, scheme: str) -> str | None:
    """Next-coarser bin for shrinkage."""
    if scheme == "rush_nonez":
        if label.isdigit() and int(label) <= 20:
            yl = int(label)
            if yl <= 5:
                return "1-5"
            if yl <= 10:
                return "6-10"
            if yl <= 20:
                return "11-20"
            return None
        mapping = {
            "21-25": "21-30",
            "26-30": "21-30",
            "31-40": "31-50",
            "41-50": "31-50",
            "51-60": "51-99",
            "61-80": "51-99",
            "81-99": "51-99",
        }
        return mapping.get(label)
    # EZ
    mapping = {
        "1": "1-5", "2": "1-5", "3": "1-5", "4": "1-5", "5": "1-5",
        "6-7": "6-10", "8-10": "6-10",
        "11-15": "11-20", "16-20": "11-20",
        "21-25": "21-30", "26-30": "21-30",
        "31-40": "31+", "41+": "31+",
    }
    return mapping.get(label)


def parent_membership(yl: int, parent: str) -> bool:
    if parent.endswith("+"):
        return yl >= int(parent[:-1])
    if "-" in parent:
        lo, hi = parent.split("-")
        return int(lo) <= yl <= int(hi)
    return yl == int(parent)


def rates_for(plays: pd.DataFrame, td_col: str, scheme: str) -> pd.DataFrame:
    plays = plays.copy()
    plays["bin"] = plays["yardline_100"].map(lambda y: bin_label(y, scheme))
    grp = (
        plays.groupby("bin", as_index=False)
        .agg(n=(td_col, "size"), tds=(td_col, "sum"))
    )
    grp["empirical"] = grp["tds"] / grp["n"]

    # parent rates
    parent_rates = {}
    for lab in grp["bin"]:
        par = parent_label(lab, scheme)
        if par is None:
            continue
        mask = plays["yardline_100"].map(lambda y, p=par: parent_membership(y, p))
        n = int(mask.sum())
        tds = float(plays.loc[mask, td_col].sum())
        parent_rates[par] = tds / n if n else np.nan

    overall = float(plays[td_col].mean()) if len(plays) else 0.0

    def shrink(row):
        emp, n, lab = row["empirical"], row["n"], row["bin"]
        if n >= MIN_N:
            return float(emp)
        par = parent_label(lab, scheme)
        base = parent_rates.get(par, overall) if par else overall
        if pd.isna(base):
            base = overall
        w = n / MIN_N
        return float(w * emp + (1 - w) * base)

    grp["p_td"] = grp.apply(shrink, axis=1)
    grp["scheme"] = scheme

    # expand to every yard line
    rows = []
    for yl in range(1, 100):
        lab = bin_label(yl, scheme)
        r = grp[grp["bin"] == lab].iloc[0]
        rows.append(
            {
                "yardline_100": yl,
                "bin": lab,
                "n_bin": int(r["n"]),
                "tds_bin": float(r["tds"]),
                "empirical": float(r["empirical"]),
                "p_td": float(r["p_td"]),
                "scheme": scheme,
            }
        )
    return pd.DataFrame(rows), grp


def roster_lookup() -> pd.DataFrame:
    ros = pd.read_parquet(
        DATA / "roster_2026.parquet",
        columns=["week", "gsis_id", "full_name", "position"],
    )
    ros = ros[ros["gsis_id"].notna()].copy()
    pos_mode = (
        ros.dropna(subset=["position"])
        .groupby(["gsis_id", "position"])
        .size()
        .reset_index(name="n")
        .sort_values(["gsis_id", "n"], ascending=[True, False])
        .groupby("gsis_id")
        .head(1)[["gsis_id", "position"]]
    )
    name = (
        ros.sort_values("week")
        .groupby("gsis_id", as_index=False)
        .tail(1)[["gsis_id", "full_name"]]
    )
    out = name.merge(pos_mode, on="gsis_id", how="left").rename(columns={"gsis_id": "player_id"})
    return out


def team_from_plays(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    # most frequent posteam; tie -> last by week
    cnt = (
        df.groupby([id_col, "posteam"])
        .agg(n=("week", "size"), last_week=("week", "max"))
        .reset_index()
        .sort_values(["n", "last_week"], ascending=[False, False])
    )
    return (
        cnt.groupby(id_col, as_index=False)
        .head(1)[[id_col, "posteam"]]
        .rename(columns={id_col: "player_id", "posteam": "team"})
    )


def ident(df, id_col, name_col, roster):
    out = df.merge(roster, left_on=id_col, right_on="player_id", how="left")
    out["player_name"] = out["full_name"].fillna(out[name_col])
    out["player_id"] = out[id_col]
    teams = team_from_plays(df, id_col)
    out = out.drop(columns=["team"], errors="ignore").merge(teams, on="player_id", how="left")
    return out


def summarize(df, actual_col, extras=None):
    aggs = {
        "opps": ("xtd", "size"),
        "xtd": ("xtd", "sum"),
        "td": (actual_col, "sum"),
        "team": ("team", "first"),
        "position": ("position", lambda s: s.dropna().iloc[-1] if s.dropna().shape[0] else np.nan),
        "player_name": ("player_name", lambda s: s.dropna().iloc[-1] if s.dropna().shape[0] else np.nan),
    }
    if extras:
        aggs.update(extras)
    g = df.groupby("player_id", as_index=False).agg(**aggs)
    g["td_minus_xtd"] = g["td"] - g["xtd"]
    return g


def main():
    print("Loading 2023-2025 training PBP...")
    train = clean_base(load_pbp([2023, 2024, 2025]))
    print(f"  training REG run/pass plays: {len(train):,}")

    rushes_tr = train[
        (train.play_type == "run")
        & (train.rush_attempt.fillna(0).astype(float) == 1)
        & train.rusher_player_id.notna()
    ]
    throws_tr = train[
        (train.play_type == "pass")
        & (train.pass_attempt.fillna(0).astype(float) == 1)
        & (train.sack.fillna(0).astype(float) == 0)
        & train.receiver_player_id.notna()
        & train.passer_player_id.notna()
    ]

    rush_yl, rush_bins = rates_for(rushes_tr, "rush_touchdown", "rush_nonez")
    nonez_yl, nonez_bins = rates_for(throws_tr[throws_tr.ez_target == 0], "pass_touchdown", "rush_nonez")
    ez_yl, ez_bins = rates_for(throws_tr[throws_tr.ez_target == 1], "pass_touchdown", "ez")

    rush_yl.to_csv(OUT / "rush_xtd_lookup_2023_2025.csv", index=False)
    nonez_yl.assign(ez_bin=0).to_csv(OUT / "pass_nonez_xtd_lookup_2023_2025.csv", index=False)
    ez_yl.assign(ez_bin=1).to_csv(OUT / "pass_ez_xtd_lookup_2023_2025.csv", index=False)
    rush_bins.to_csv(OUT / "rush_xtd_bins_2023_2025.csv", index=False)
    nonez_bins.assign(kind="non_ez").to_csv(OUT / "pass_nonez_xtd_bins_2023_2025.csv", index=False)
    ez_bins.assign(kind="ez").to_csv(OUT / "pass_ez_xtd_bins_2023_2025.csv", index=False)

    print("\nRush TD rates (2023-2025 REG):")
    print(f"  n={len(rushes_tr):,}  TDs={int(rushes_tr.rush_touchdown.sum()):,}  "
          f"rate={rushes_tr.rush_touchdown.mean():.3%}")
    for yl in [1, 2, 3, 4, 5, 10, 15, 20]:
        r = rush_yl[rush_yl.yardline_100 == yl].iloc[0]
        print(f"  yardline {yl:2d}: p={r.p_td:.3f}  n_bin={r.n_bin}")
    print("  pooled longer rushes:")
    for lab in ["21-25", "26-30", "31-40", "41-50", "51-60"]:
        r = rush_yl[rush_yl.bin == lab].iloc[0]
        print(f"  {lab}: p={r.p_td:.3f}  n_bin={r.n_bin}  tds={r.tds_bin:.0f}")

    print("\nNon-EZ targeted throw TD rates:")
    print(f"  n={int((throws_tr.ez_target==0).sum()):,}  TDs={int(throws_tr.loc[throws_tr.ez_target==0,'pass_touchdown'].sum()):,}")
    for yl in [1, 2, 3, 5, 10, 20]:
        r = nonez_yl[nonez_yl.yardline_100 == yl].iloc[0]
        print(f"  yardline {yl:2d}: p={r.p_td:.3f}  n_bin={r.n_bin}")

    print("\nEnd-zone target TD rates:")
    print(f"  n={int((throws_tr.ez_target==1).sum()):,}  TDs={int(throws_tr.loc[throws_tr.ez_target==1,'pass_touchdown'].sum()):,}")
    shown = set()
    for _, r in ez_yl.iterrows():
        if r.bin in shown:
            continue
        shown.add(r.bin)
        print(f"  EZ bin {r.bin:>6}: p={r.p_td:.3f}  n={r.n_bin}  tds={r.tds_bin:.0f}")

    print("\nLoading 2026 PBP...")
    raw26 = pd.read_parquet(DATA / "play_by_play_2026.parquet", columns=PBP_COLS)
    pbp26 = clean_base(raw26)

    rushes = pbp26[
        (pbp26.play_type == "run")
        & (pbp26.rush_attempt.fillna(0).astype(float) == 1)
        & pbp26.rusher_player_id.notna()
    ].merge(rush_yl[["yardline_100", "p_td", "bin"]].rename(columns={"p_td": "xtd", "bin": "xtd_bin"}),
            on="yardline_100", how="left")
    assert rushes.xtd.notna().all()

    throws = pbp26[
        (pbp26.play_type == "pass")
        & (pbp26.pass_attempt.fillna(0).astype(float) == 1)
        & (pbp26.sack.fillna(0).astype(float) == 0)
        & pbp26.receiver_player_id.notna()
        & pbp26.passer_player_id.notna()
    ].copy()
    nonez = throws[throws.ez_target == 0].merge(
        nonez_yl[["yardline_100", "p_td", "bin"]].rename(columns={"p_td": "xtd", "bin": "xtd_bin"}),
        on="yardline_100", how="left",
    )
    ez = throws[throws.ez_target == 1].merge(
        ez_yl[["yardline_100", "p_td", "bin"]].rename(columns={"p_td": "xtd", "bin": "xtd_bin"}),
        on="yardline_100", how="left",
    )
    throws = pd.concat([nonez, ez], ignore_index=True)
    assert throws.xtd.notna().all()

    games = raw26[raw26.season_type == "REG"]
    n_games = int(games.game_id.nunique())
    date_min, date_max = str(games.game_date.min()), str(games.game_date.max())
    weeks = sorted(int(w) for w in games.week.unique())
    w10 = int(games[games.week == 10].game_id.nunique())
    print(f"  2026 REG: {n_games} games, weeks {weeks[0]}-{weeks[-1]}, {date_min} to {date_max}")
    print(f"  Week 10 games in file: {w10} (LAC @ BAL MNF Mon Nov 16 not yet included)")
    print(f"  Rushes: actual TDs {int(rushes.rush_touchdown.sum())}  xTD {rushes.xtd.sum():.1f}  "
          f"diff {rushes.rush_touchdown.sum()-rushes.xtd.sum():+.1f}")
    print(f"  Targeted throws: actual TDs {int(throws.pass_touchdown.sum())}  xTD {throws.xtd.sum():.1f}  "
          f"diff {throws.pass_touchdown.sum()-throws.xtd.sum():+.1f}")
    print(f"  2026 EZ targets: {int(throws.ez_target.sum())}  TDs {int(throws.loc[throws.ez_target==1,'pass_touchdown'].sum())}  "
          f"xTD {throws.loc[throws.ez_target==1,'xtd'].sum():.1f}")

    roster = roster_lookup()
    rush_p = ident(rushes, "rusher_player_id", "rusher_player_name", roster)
    rush_p["inside5"] = (rush_p.yardline_100 <= 5).astype(int)
    rec_p = ident(throws, "receiver_player_id", "receiver_player_name", roster)
    qb_p = ident(throws, "passer_player_id", "passer_player_name", roster)

    rush_sum = summarize(rush_p, "rush_touchdown", {"inside5": ("inside5", "sum"), "rushes": ("xtd", "size")})
    rec_sum = summarize(rec_p, "pass_touchdown", {"ez_targets": ("ez_target", "sum"), "targets": ("xtd", "size")}).rename(
        columns={"td": "rec_td", "xtd": "rec_xtd"}
    )
    qb_sum = summarize(qb_p, "pass_touchdown").rename(columns={"opps": "attempts", "td": "pass_td", "xtd": "pass_xtd"})
    qb_sum["td_minus_xtd"] = qb_sum["pass_td"] - qb_sum["pass_xtd"]

    combined = pd.merge(
        rush_sum[["player_id", "player_name", "position", "team", "td", "xtd", "rushes", "inside5"]].rename(
            columns={"td": "rush_td", "xtd": "rush_xtd"}
        ),
        rec_sum[["player_id", "player_name", "position", "team", "rec_td", "rec_xtd", "targets", "ez_targets"]],
        on="player_id", how="outer", suffixes=("_r", "_c"),
    )
    for col in ["player_name", "position", "team"]:
        combined[col] = combined[f"{col}_c"].combine_first(combined[f"{col}_r"])
        combined = combined.drop(columns=[f"{col}_r", f"{col}_c"])
    for c in ["rush_td", "rush_xtd", "rushes", "inside5", "rec_td", "rec_xtd", "targets", "ez_targets"]:
        combined[c] = combined[c].fillna(0)
    combined["td"] = combined["rush_td"] + combined["rec_td"]
    combined["xtd"] = combined["rush_xtd"] + combined["rec_xtd"]
    combined["td_minus_xtd"] = combined["td"] - combined["xtd"]
    combined["looks"] = combined["rushes"] + combined["targets"]

    rb = combined[combined.position == "RB"].copy()
    wrte = combined[combined.position.isin(["WR", "TE"])].copy()
    qb = qb_sum[qb_sum.position == "QB"].copy()

    MIN_RB, MIN_WR, MIN_QB = 40, 25, 150
    rb_q = rb[rb.looks >= MIN_RB].sort_values("td_minus_xtd", ascending=False)
    wr_q = wrte[wrte.targets >= MIN_WR].sort_values("td_minus_xtd", ascending=False)
    qb_q = qb[qb.attempts >= MIN_QB].sort_values("td_minus_xtd", ascending=False)

    rb.sort_values("td_minus_xtd", ascending=False).to_csv(OUT / "rb_xtd_2026.csv", index=False)
    wrte.sort_values("td_minus_xtd", ascending=False).to_csv(OUT / "wrte_xtd_2026.csv", index=False)
    qb.sort_values("td_minus_xtd", ascending=False).to_csv(OUT / "qb_pass_xtd_2026.csv", index=False)
    combined.sort_values("td_minus_xtd", ascending=False).to_csv(OUT / "all_skill_xtd_2026.csv", index=False)

    # ffopportunity
    ep = pd.read_parquet(DATA / "ep_weekly_2026.parquet")
    rec = ep[ep.receptions.fillna(0) > 0]
    implied_ppr = rec.rec_yards_gained.fillna(0) * 0.1 + rec.rec_touchdown.fillna(0) * 6 + rec.receptions.fillna(0)
    scoring = "PPR" if (rec.rec_fantasy_points - implied_ppr).abs().median() < 0.05 else "unknown"
    ep_skill = ep[ep.position.isin(["QB", "RB", "WR", "TE"])].copy()
    season = ep_skill.groupby(["player_id", "full_name", "position"], as_index=False).agg(
        team=("posteam", "last"),
        games=("week", "nunique"),
        fp=("total_fantasy_points", "sum"),
        xfp=("total_fantasy_points_exp", "sum"),
        rush_td=("rush_touchdown", "sum"),
        rush_xtd=("rush_touchdown_exp", "sum"),
        rec_td=("rec_touchdown", "sum"),
        rec_xtd=("rec_touchdown_exp", "sum"),
        pass_td=("pass_touchdown", "sum"),
        pass_xtd=("pass_touchdown_exp", "sum"),
        rush_att=("rush_attempt", "sum"),
        targets=("rec_attempt", "sum"),
        pass_att=("pass_attempt", "sum"),
    )
    season["fp_minus_xfp"] = season.fp - season.xfp
    season["td"] = season.rush_td + season.rec_td
    season["xtd_off"] = season.rush_xtd + season.rec_xtd
    season["td_minus_xtd"] = season.td - season.xtd_off
    season["pass_td_minus_xtd"] = season.pass_td - season.pass_xtd
    season.to_csv(OUT / "ffopportunity_season_2026.csv", index=False)

    def corr_msg(a, b, name):
        m = a.merge(b, on="player_id", how="inner")
        if len(m) > 5:
            print(f"  corr {name}: {m.d1.corr(m.d2):.3f} (n={len(m)})")

    print("\n================ PRELIMINARY LEADERBOARDS (our yard-line / EZ-bin model) ================")
    print(f"Filters: RB looks>={MIN_RB}; WR/TE targets>={MIN_WR}; QB attempts>={MIN_QB}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    def show(title, df, cols, n=6):
        print(f"\n{title}")
        print(df[cols].head(n).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    cols_rb = ["player_name", "team", "looks", "rushes", "targets", "td", "xtd", "td_minus_xtd", "inside5", "ez_targets", "rush_td", "rush_xtd", "rec_td", "rec_xtd"]
    show("RB lucky (TD - xTD)", rb_q, cols_rb)
    show("RB unlucky (TD - xTD)", rb_q.sort_values("td_minus_xtd"), cols_rb)
    cols_wr = ["player_name", "team", "position", "targets", "td", "xtd", "td_minus_xtd", "ez_targets", "rec_td", "rec_xtd", "rush_td", "rush_xtd"]
    show("WR/TE lucky (TD - xTD)", wr_q, cols_wr)
    show("WR/TE unlucky (TD - xTD)", wr_q.sort_values("td_minus_xtd"), cols_wr)
    cols_qb = ["player_name", "team", "attempts", "pass_td", "pass_xtd", "td_minus_xtd"]
    show("QB passing lucky", qb_q, cols_qb)
    show("QB passing unlucky", qb_q.sort_values("td_minus_xtd"), cols_qb)

    ep_rb = season[(season.position == "RB") & (season.rush_att + season.targets >= MIN_RB)]
    ep_wr = season[(season.position.isin(["WR", "TE"])) & (season.targets >= MIN_WR)]
    ep_qb = season[(season.position == "QB") & (season.pass_att >= MIN_QB)]

    print(f"\n================ FFOPPORTUNITY ({scoring}) expected fantasy points + their xTD ================")
    cols_ep = ["full_name", "team", "position", "games", "fp", "xfp", "fp_minus_xfp", "td", "xtd_off", "td_minus_xtd"]
    show("RB over xFP", ep_rb.sort_values("fp_minus_xfp", ascending=False), cols_ep)
    show("RB under xFP", ep_rb.sort_values("fp_minus_xfp"), cols_ep)
    show("RB over xTD (ffopp rush+rec)", ep_rb.sort_values("td_minus_xtd", ascending=False), cols_ep)
    show("RB under xTD (ffopp rush+rec)", ep_rb.sort_values("td_minus_xtd"), cols_ep)
    show("WR/TE over xFP", ep_wr.sort_values("fp_minus_xfp", ascending=False), cols_ep)
    show("WR/TE under xFP", ep_wr.sort_values("fp_minus_xfp"), cols_ep)
    show("WR/TE over xTD (ffopp)", ep_wr.sort_values("td_minus_xtd", ascending=False), cols_ep)
    show("WR/TE under xTD (ffopp)", ep_wr.sort_values("td_minus_xtd"), cols_ep)
    cols_eqb = ["full_name", "team", "games", "fp", "xfp", "fp_minus_xfp", "pass_td", "pass_xtd", "pass_td_minus_xtd"]
    show("QB over xFP", ep_qb.sort_values("fp_minus_xfp", ascending=False), cols_eqb)
    show("QB under xFP", ep_qb.sort_values("fp_minus_xfp"), cols_eqb)
    show("QB over pass xTD (ffopp)", ep_qb.sort_values("pass_td_minus_xtd", ascending=False), cols_eqb)
    show("QB under pass xTD (ffopp)", ep_qb.sort_values("pass_td_minus_xtd"), cols_eqb)

    corr_msg(
        rb_q[["player_id", "td_minus_xtd"]].rename(columns={"td_minus_xtd": "d1"}),
        ep_rb[["player_id", "td_minus_xtd"]].rename(columns={"td_minus_xtd": "d2"}),
        "RB TD-xTD, our model vs ffopportunity",
    )
    corr_msg(
        wr_q[["player_id", "td_minus_xtd"]].rename(columns={"td_minus_xtd": "d1"}),
        ep_wr[["player_id", "td_minus_xtd"]].rename(columns={"td_minus_xtd": "d2"}),
        "WR/TE TD-xTD, our model vs ffopportunity",
    )
    corr_msg(
        qb_q[["player_id", "td_minus_xtd"]].rename(columns={"td_minus_xtd": "d1"}),
        ep_qb[["player_id", "pass_td_minus_xtd"]].rename(columns={"pass_td_minus_xtd": "d2"}),
        "QB pass TD-xTD, our model vs ffopportunity",
    )

    # Michael Wilson detail for the story
    mw = rec_p[rec_p.player_name == "Michael Wilson"]
    if len(mw):
        print(f"\nMichael Wilson detail: targets={len(mw)} EZ={int(mw.ez_target.sum())} "
              f"inside-10 EZ={int(((mw.ez_target==1)&(mw.yardline_100<=10)).sum())} "
              f"inside-5 looks={int((mw.yardline_100<=5).sum())} "
              f"TD={int(mw.pass_touchdown.sum())} xTD={mw.xtd.sum():.2f}")

    meta = {
        "pbp_timestamp": json.loads((DATA / "pbp_timestamp.json").read_text()),
        "ffopportunity_timestamp_raw": (DATA / "ffopp_timestamp.txt").read_text().strip(),
        "games": n_games,
        "date_min": date_min,
        "date_max": date_max,
        "weeks": weeks,
        "week10_games": w10,
        "missing_note": "Week 10 MNF LAC @ BAL Mon Nov 16, 2026 not in file. Week 10 bye teams appear to be CHI, DEN, PHI, TB.",
        "league_2026_rush_td": float(rushes.rush_touchdown.sum()),
        "league_2026_rush_xtd": float(rushes.xtd.sum()),
        "league_2026_pass_td": float(throws.pass_touchdown.sum()),
        "league_2026_pass_xtd": float(throws.xtd.sum()),
        "ffopportunity_scoring": scoring,
        "filters": {"rb_looks": MIN_RB, "wrte_targets": MIN_WR, "qb_attempts": MIN_QB},
        "n_qualified": {"rb": int(len(rb_q)), "wrte": int(len(wr_q)), "qb": int(len(qb_q))},
        "train_rushes": int(len(rushes_tr)),
        "train_rush_tds": int(rushes_tr.rush_touchdown.sum()),
        "train_throws": int(len(throws_tr)),
        "train_pass_tds": int(throws_tr.pass_touchdown.sum()),
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("\nWrote output to", OUT)


if __name__ == "__main__":
    main()
