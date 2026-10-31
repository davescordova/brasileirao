"""Brasileirão Bayesian prediction model.

Fetches the season schedule from the Globo Esporte JSON API, fits a Bayesian
Poisson model (PyMC + NumPyro on CPU), simulates the rest of the season with
Monte Carlo and writes an interactive HTML page plus CSV files.

Outputs:
    site/index.html                 -> published on GitHub Pages
    site/probabilities.csv          -> latest table (downloadable from the page)
    data/history/<dated>.csv        -> one snapshot per run, committed to the repo
"""

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymc as pm
import pytensor.tensor as pt
import requests

SEASON = 2026
N_ROUNDS = 38  # Brasileirão Série A: 20 clubs, 38 rounds, 380 matches
REQUEST_DELAY_SECONDS = 1.0
COMPETITION_UUID = "d1a37fa4-e948-43a6-ba53-ab24ab3a45b1"
N_SIMS = 10_000
DRAWS = 2000
TUNE = 1000
CHAINS = 4
SEED = 42

SITE_DIR = "site"
HISTORY_DIR = os.path.join("data", "history")
LOCAL_TZ = ZoneInfo("America/Sao_Paulo")

SCHEDULE_COLUMNS = ["season", "round", "date", "home_team", "away_team",
                    "home_score", "away_score"]

# (column, legend label, color, text color inside the bar)
CATEGORIES = [
    ("Champion (%)", "Campeão", "#2ecc71", "white"),
    ("Libertadores 2nd-6th (%)", "Libertadores", "#f1c40f", "black"),
    ("Sul-Americana 7th-12th (%)", "Sul-Americana", "#3498db", "white"),
    ("None 13th-16th (%)", "Nenhum", "#95a5a6", "black"),
    ("Relegation (%)", "Rebaixamento", "#e74c3c", "white"),
]


# --------------------------------------------------------------------------
# Data collection
# --------------------------------------------------------------------------
def fetch_round(round_number, season=SEASON):
    """Fetch a single round. Rounds not yet published return 404 -> empty frame."""
    url = (
        "https://api.globoesporte.globo.com/tabela/"
        f"{COMPETITION_UUID}/fase/fase-unica-campeonato-brasileiro-{season}/"
        f"rodada/{str(round_number).zfill(2)}/jogos/"
    )
    time.sleep(REQUEST_DELAY_SECONDS)
    response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})

    if response.status_code == 404:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    response.raise_for_status()

    rows = []
    for match in response.json():
        teams = match.get("equipes", {})
        rows.append({
            "season": season,
            "round": round_number,
            "date": match.get("data_realizacao", ""),
            "home_team": teams.get("mandante", {}).get("nome_popular", "N/A"),
            "away_team": teams.get("visitante", {}).get("nome_popular", "N/A"),
            "home_score": match.get("placar_oficial_mandante"),
            "away_score": match.get("placar_oficial_visitante"),
        })
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)


def fetch_full_schedule(season=SEASON, n_rounds=N_ROUNDS):
    """Fetch every round of the season and return one combined DataFrame."""
    rounds = [fetch_round(r, season) for r in range(1, n_rounds + 1)]
    rounds = [r for r in rounds if not r.empty]
    assert rounds, "No matches were retrieved from the API."

    schedule = pd.concat(rounds, ignore_index=True)
    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")
    schedule["home_score"] = pd.to_numeric(schedule["home_score"]).astype("Int64")
    schedule["away_score"] = pd.to_numeric(schedule["away_score"]).astype("Int64")
    return schedule.drop_duplicates(
        subset=["round", "home_team", "away_team"], keep="first"
    ).reset_index(drop=True)


# --------------------------------------------------------------------------
# Standings
# --------------------------------------------------------------------------
def build_standings(matches, teams):
    """Current table from played matches (vectorized)."""
    home = pd.DataFrame({
        "team": matches["home_team"],
        "GF": matches["home_score"],
        "GA": matches["away_score"],
    })
    away = pd.DataFrame({
        "team": matches["away_team"],
        "GF": matches["away_score"],
        "GA": matches["home_score"],
    })
    rows = pd.concat([home, away], ignore_index=True)
    rows["W"] = (rows["GF"] > rows["GA"]).astype(int)
    rows["D"] = (rows["GF"] == rows["GA"]).astype(int)
    rows["L"] = (rows["GF"] < rows["GA"]).astype(int)
    rows["P"] = 1

    table = (
        rows.groupby("team")[["P", "W", "D", "L", "GF", "GA"]].sum()
        .reindex(teams, fill_value=0)
    )
    table["GD"] = table["GF"] - table["GA"]
    table["Pts"] = table["W"] * 3 + table["D"]
    table = table.sort_values(["Pts", "W", "GD", "GF"], ascending=False)
    return table.rename_axis("team").reset_index()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def fit_model(played, team_to_idx, n_teams):
    """Bayesian Poisson model: attack/defense per team + home advantage."""
    home_idx = played["home_team"].map(team_to_idx).to_numpy()
    away_idx = played["away_team"].map(team_to_idx).to_numpy()

    with pm.Model():
        atts = pm.ZeroSumNormal("atts", sigma=2, shape=n_teams)
        defs = pm.ZeroSumNormal("defs", sigma=2, shape=n_teams)
        home_adv = pm.Normal("home_adv", mu=0, sigma=1)
        intercept = pm.Normal("intercept", mu=0, sigma=2)

        home_theta = pt.exp(intercept + home_adv + atts[home_idx] - defs[away_idx])
        away_theta = pt.exp(intercept + atts[away_idx] - defs[home_idx])

        pm.Poisson("home_goals", mu=home_theta, observed=played["home_score"].to_numpy())
        pm.Poisson("away_goals", mu=away_theta, observed=played["away_score"].to_numpy())

        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            nuts_sampler="numpyro",
            random_seed=SEED,
            progressbar=False,
        )

    post = idata.posterior
    return {
        "intercept": float(post["intercept"].mean()),
        "home_adv": float(post["home_adv"].mean()),
        "atts": post["atts"].mean(dim=["chain", "draw"]).values,
        "defs": post["defs"].mean(dim=["chain", "draw"]).values,
    }


# --------------------------------------------------------------------------
# Monte Carlo simulation
# --------------------------------------------------------------------------
def simulate_season(schedule, params, team_to_idx, n_teams, n_sims=N_SIMS):
    """Simulate unplayed matches, keep real results, return position probabilities."""
    rng = np.random.default_rng(SEED)
    home_idx = schedule["home_team"].map(team_to_idx).to_numpy()
    away_idx = schedule["away_team"].map(team_to_idx).to_numpy()
    n_total = len(schedule)

    actual_home = schedule["home_score"].fillna(-1).to_numpy(dtype="int64")
    actual_away = schedule["away_score"].fillna(-1).to_numpy(dtype="int64")

    lam_home = np.exp(params["intercept"] + params["home_adv"]
                      + params["atts"][home_idx] - params["defs"][away_idx])
    lam_away = np.exp(params["intercept"]
                      + params["atts"][away_idx] - params["defs"][home_idx])

    goals_home = rng.poisson(lam_home, size=(n_sims, n_total))
    goals_away = rng.poisson(lam_away, size=(n_sims, n_total))
    goals_home = np.where(actual_home != -1, actual_home, goals_home)
    goals_away = np.where(actual_away != -1, actual_away, goals_away)

    home_win = goals_home > goals_away
    away_win = goals_away > goals_home
    draw = goals_home == goals_away
    points_home = 3 * home_win + draw
    points_away = 3 * away_win + draw
    gd_home = goals_home - goals_away

    m_home = np.zeros((n_teams, n_total))
    m_home[home_idx, np.arange(n_total)] = 1
    m_away = np.zeros((n_teams, n_total))
    m_away[away_idx, np.arange(n_total)] = 1

    total_points = points_home @ m_home.T + points_away @ m_away.T
    total_wins = home_win @ m_home.T + away_win @ m_away.T
    total_gd = gd_home @ m_home.T - gd_home @ m_away.T

    score = total_points * 1_000_000 + total_wins * 1000 + total_gd
    ranks = np.argsort(np.argsort(-score, axis=1), axis=1)  # 0 = champion

    return {
        "champion": (ranks == 0).mean(axis=0),
        "libertadores": ((ranks >= 1) & (ranks <= 5)).mean(axis=0),
        "sulamericana": ((ranks >= 6) & (ranks <= 11)).mean(axis=0),
        "none": ((ranks >= 12) & (ranks < n_teams - 4)).mean(axis=0),
        "relegation": (ranks >= n_teams - 4).mean(axis=0),
    }


def build_results_table(teams, probs):
    return pd.DataFrame({
        "Team": teams,
        "Champion (%)": probs["champion"] * 100,
        "Libertadores 2nd-6th (%)": probs["libertadores"] * 100,
        "Sul-Americana 7th-12th (%)": probs["sulamericana"] * 100,
        "None 13th-16th (%)": probs["none"] * 100,
        "Relegation (%)": probs["relegation"] * 100,
    }).round(2).sort_values("Champion (%)", ascending=False)


# --------------------------------------------------------------------------
# Interactive chart / page
# --------------------------------------------------------------------------
def build_figure(results_df, standings, n_played, n_total, updated_at):
    # Current standings order, first place on top
    plot_df = results_df.set_index("Team").loc[standings["team"].tolist()[::-1]]
    plot_df = plot_df.join(standings.set_index("team")[["Pts", "P"]])
    positions = {team: i + 1 for i, team in enumerate(standings["team"])}
    y_labels = [f"{positions[t]}º {t}" for t in plot_df.index]

    fig = go.Figure()
    for col, label, color, text_color in CATEGORIES:
        values = plot_df[col].to_numpy()
        fig.add_trace(go.Bar(
            name=label,
            y=y_labels,
            x=values,
            orientation="h",
            marker_color=color,
            text=[f"{v:.0f}%" if v >= 4 else "" for v in values],
            textposition="inside",
            insidetextanchor="middle",
            textfont={"color": text_color, "size": 11},
            customdata=np.stack([plot_df.index, plot_df["Pts"], plot_df["P"]], axis=1),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]} pts em %{customdata[2]} jogos)"
                f"<br>{label}: <b>%{{x:.2f}}%</b><extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="stack",
        title={
            "text": (
                f"<b>Brasileirão {SEASON} – Probabilidade de classificação final por time</b><br>"
                f"<sup>Com base nos {n_played} de {n_total} jogos realizados "
                f"(atualizado em {updated_at})</sup>"
            ),
            "x": 0.5,
        },
        xaxis={"range": [0, 100], "title": "Probabilidade (%)", "ticksuffix": "%"},
        yaxis={"automargin": True},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.08,
                "traceorder": "normal"},
        height=max(600, 32 * len(plot_df) + 200),
        margin={"l": 20, "r": 20, "t": 90, "b": 60},
        plot_bgcolor="white",
        hoverlabel={"bgcolor": "white", "font_size": 13},
        uniformtext={"minsize": 8, "mode": "hide"},
    )
    return fig


def write_page(fig, path):
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                             config={"displaylogo": False, "responsive": True})
    page = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Brasileirão {SEASON} – Probabilidades</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 16px; color: #222; }}
  footer {{ font-size: 13px; color: #666; margin-top: 12px; }}
  a {{ color: #2471a3; }}
</style>
</head>
<body>
{chart_html}
<footer>
  Modelo bayesiano de Poisson (ataque/defesa por time + mando de campo),
  {N_SIMS:,} simulações de Monte Carlo do restante da temporada.
  <a href="probabilities.csv" download>Baixar tabela (CSV)</a>
</footer>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(schedule):
    now = datetime.now(LOCAL_TZ)

    played = schedule.dropna(subset=["home_score", "away_score"]).copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    assert len(played) > 0, "No played matches found yet, cannot fit the model."
    print(f"Played matches: {len(played)} / {len(schedule)}")

    teams = sorted(pd.unique(schedule[["home_team", "away_team"]].values.ravel()))
    team_to_idx = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    standings = build_standings(played, teams)
    print(standings.to_string(index=False))

    params = fit_model(played, team_to_idx, n_teams)
    probs = simulate_season(schedule, params, team_to_idx, n_teams)
    results_df = build_results_table(teams, probs)
    print(results_df.to_string(index=False))

    os.makedirs(SITE_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(SITE_DIR, "probabilities.csv"), index=False)
    results_df.to_csv(
        os.path.join(HISTORY_DIR, f"brasileirao_{SEASON}_probabilities_{now:%Y_%m_%d}.csv"),
        index=False,
    )

    fig = build_figure(results_df, standings, len(played), len(schedule),
                       now.strftime("%d/%m/%Y %H:%M"))
    write_page(fig, os.path.join(SITE_DIR, "index.html"))
    print(f"Page written to {os.path.join(SITE_DIR, 'index.html')}")


if __name__ == "__main__":
    print(f"Fetching Brasileirão {SEASON} schedule ({N_ROUNDS} rounds)...")
    run(fetch_full_schedule())
