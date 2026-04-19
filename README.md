# UPI-FIFA-Paradox 🏟️⚽

**Solving the "FIFA Paradox": The Unified Performance Index (UPI)**

A sports data science research project that builds a comprehensive player rating framework by synthesizing multiple state-value models — rewarding *every deliberate action* on the pitch instead of just goals and assists.

## Dataset

[StatsBomb Open Data](https://github.com/statsbomb/open-data) — accessed via the `statsbombpy` Python API.

- **Event Data:** On-ball actions (passes, dribbles, shots, interceptions)
- **StatsBomb 360 Data:** Freeze-frame tracking with $(x,y)$ positions of all visible players

## Project Structure

```
UPI/
├── data/
│   ├── raw/               # Downloaded JSON from StatsBomb
│   └── processed/         # Cleaned & merged CSVs / Parquet files
├── notebooks/             # Jupyter notebooks for EDA & visualization
├── src/
│   ├── data/              # Data extraction scripts (statsbombpy)
│   ├── features/          # Feature engineering (spatial, contextual, 360)
│   ├── models/            # Markov Chains, xG, xT, VAEP, discrete EPV
│   └── visualization/     # Pitch maps, Voronoi diagrams, Radar plots
├── roadmap.md             # Detailed research roadmap
├── requirements.txt       # Python dependencies
└── .gitignore
```

## Models

| Model | Purpose |
|-------|---------|
| **Markov Chain** | State-value model — credit = P(Goal)_after − P(Goal)_before |
| **xG** | Logistic Regression anchoring terminal shot quality |
| **xT** | Grid-based model rewarding ball progression into high-threat zones |
| **VAEP** | Values *all* on-ball actions via P(Score) increase & P(Concede) decrease |
| **Discrete EPV** | 360 freeze-frame Voronoi + pitch control for off-ball credit |

## Setup

```bash
# Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## License

Data provided under the [StatsBomb Open Data License](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf). Please credit StatsBomb when publishing any research based on this data.
