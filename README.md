# ⚽ Brasileirão 2026 — Bayesian Prediction Model

[![Update predictions](https://github.com/davescordova/brasileirao/actions/workflows/update.yml/badge.svg)](https://github.com/davescordova/brasileirao/actions/workflows/update.yml)
[![Live page](https://img.shields.io/badge/live-GitHub%20Pages-2ea44f?logo=github)](https://davescordova.github.io/brasileirao/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyMC](https://img.shields.io/badge/PyMC-Bayesian-1F77B4)](https://www.pymc.io/)
[![JAX](https://img.shields.io/badge/JAX%20%2B%20NumPyro-NUTS-A8B1FF)](https://num.pyro.ai/)
[![Plotly](https://img.shields.io/badge/Plotly-interactive-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Daily-updated probabilities of every club's final position in the **Campeonato Brasileiro Série A 2026**: title, Libertadores, Sul-Americana, mid-table and relegation.

👉 **[See the interactive chart](https://davescordova.github.io/brasileirao/)**. Hover over the bars to see the exact percentages.

---

## 📊 What it shows

| Category | Positions |
|---|---|
| 🏆 Champion | 1st |
| 🟡 Libertadores | 2nd – 6th |
| 🔵 Sul-Americana | 7th – 12th |
| ⚪ None | 13th – 16th |
| 🔻 Relegation | 17th – 20th |

## 🧠 How it works

1. **Data**: fetches every round of the season from the Globo Esporte JSON API.
2. **Model**: Bayesian Poisson model with an attack and a defense strength per team plus a global home-advantage term, fitted with NUTS (PyMC + NumPyro) on all matches played so far.
3. **Simulation**: 10,000 Monte Carlo simulations of the remaining fixtures. Real results are kept, and the tie-breakers are points, wins, then goal difference.
4. **Output**: an interactive Plotly page and a CSV with the probabilities.

## 🤖 Automation

A [GitHub Actions workflow](.github/workflows/update.yml) runs every day at **02:39 (Brasília)**:

- runs `brasileirao.py`
- commits a dated snapshot to [`data/history/`](data/history)
- publishes `site/` to GitHub Pages


## 💻 Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python brasileirao.py
```

The page is written to `site/index.html` and the table to `site/probabilities.csv`.

## 📁 Structure

```
├── brasileirao.py              # data → model → simulation → page
├── requirements.txt
├── data/history/               # daily CSV snapshots
└── .github/workflows/update.yml
```

## ⚠️ Disclaimer

The data comes from an unofficial endpoint, and the probabilities are model estimates, not certainties.

## 📄 License

[MIT](LICENSE) © 2026 Daves de Azevedo Cordova