 # HockeyAnalysis

*A research project focusing on the quantitative identification, evaluation, and theoretical modeling of playing styles in the Swedish Hockey League (SHL).*

---

## 1. Project Overview

This project investigates **team-level tactical styles** in professional ice hockey using **event-level data** from the Swedish Hockey League (SHL). It integrates **unsupervised clustering, statistical testing, spatial analytics, and game-theoretic modeling** to provide a multi-faceted understanding of tactical behavior.  

Key contributions include:
- **Playing Style Identification**: Extracted 13 aggregated performance features and applied **K-Means clustering** to identify three distinct tactical styles:
  - **Defensive Counterattack**
  - **High-Pressure Offense**
  - **Puck Control Play**
- **Performance Evaluation**: Constructed a **pairwise win-rate matrix** for style matchups, corrected using **Bayesian Averaging** to mitigate sample imbalance.
- **Statistical Validation**: Applied **Welch’s t-tests**, **Pearson’s chi-squared test**, and **Bootstrap resampling** to validate clustering distinctiveness and matchup significance.
- **Spatial Analytics**: Analyzed **possession loss distributions** via **Kernel Density Estimation (KDE)**, revealing spatial vulnerabilities for different styles.
- **Game-Theoretic Modeling**: Developed a **2-player strategic game** to capture style interactions, identifying the **Nash Equilibrium** within the tactical meta.

---

## 2. Data

- **Source**: SHL event-level game data provided by **Sportlogiq Inc.**
- **Structure**: Detailed logs for every pass, shot, carry, block, entry, and clearance, including:
  - `gameid`, `teamid`, `teaminpossession`, `xadjcoord`, `yadjcoord`, `compiledgametime`, `ishomegame`, `eventname`, `type`
- **Processing**:
  - Sorted events by game and time.
  - Engineered features such as pass success rates, controlled zone entries, carry distances, and defensive actions.
  - Aggregated features by **team-game level** for clustering.

---

## 3. Methodology

### 3.1 Feature Engineering & Aggregation
- Derived **count-based** (e.g., number of passes, entries) and **average-based** (e.g., carry distance, xG) features.
- Built **efficiency indicators** (e.g., pass success rate) by combining raw counts.

### 3.2 Clustering for Style Identification
- Selected **13 standardized features** covering offense, defense, puck control, and efficiency.
- Determined **k = 3** as optimal using **Silhouette Score** and **Elbow Method**.
- Interpreted styles based on multi-dimensional radar charts.

### 3.3 Evaluating Inter-Style Effectiveness
- Constructed a **pairwise win-rate matrix** between styles.
- Corrected raw win rates using **Bayesian Averaging** to address sample imbalance.
- Validated matchup dynamics through cross-validation (70/30 split).

### 3.4 Spatial Analysis of Possession Loss
- Extracted **possession loss events** and unified coordinates to the **attacking-right perspective**.
- Computed **KDE-based heatmaps** for each style, visualizing key turnover locations and their tactical implications.

### 3.5 Game-Theoretic Framework
- Modeled style matchups as a **2-player strategic game** using the corrected win-rate matrix as payoffs.
- Identified **Defensive Counterattack** as the **strictly dominant strategy**, forming the **pure-strategy Nash Equilibrium**.
- Discussed implications for league-wide tactical evolution and mixed-strategy adaptations.

---

## 4. Repository Structure

| File | Description |
|---|---|
| `style.ipynb` | Main notebook for style clustering, matchup analysis, and visualization *(recommended rename: `style_possession_analysis.ipynb`)* |
| `possession_loss_heatmap.py` | Script for extracting possession loss events, coordinate normalization, and Z‑score heatmap generation |
| `game_style.py` | Aggregation of team-game features and style assignment |
| `Hockey_Validation.ipynb` | Validation notebook for statistical tests and robustness checks |

---

## 5. Key Findings

- **Three distinctive styles** (Defensive Counterattack, High-Pressure Offense, Puck Control Play) emerge from clustering.
- **Defensive Counterattack** dominates the tactical meta:
  - Achieves **82% win rate vs. Puck Control Play** (95% CI: [69.6%, 92.8%]).
  - Outperforms High-Pressure Offense in direct matchups.
- **Game-theoretic analysis** identifies Defensive Counterattack as the **unique Nash Equilibrium strategy**.
- **Spatial turnover analysis** shows:
  - Puck Control Play: higher defensive/neutral zone turnovers → vulnerability to counterattacks.
  - High-Pressure Offense: offensive zone turnovers with lower defensive risk.
  - Defensive Counterattack: turnovers concentrated near the center, supporting structured defense.

---

## 6. How to Run

### Setup
```bash
pip install numpy pandas matplotlib seaborn scipy
