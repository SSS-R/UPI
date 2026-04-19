# Solving the "FIFA Paradox": The Unified Performance Index (UPI)

## 1. The Core Problem (The FIFA Paradox)
Traditional soccer rating systems suffer from "outcome bias." They heavily overvalue terminal actions (goals, assists) and undervalue the "invisible" tactical contributions of deep-lying playmakers and defensive midfielders. A player might execute a perfect, line-breaking pass under heavy pressure, but if their teammate misses the subsequent shot, the playmaker receives zero credit. This project aims to mathematically solve this paradox.

## 2. Project Objective
To build the **Unified Performance Index (UPI)**: a single, comprehensive player rating framework that synthesizes multiple state-value models. Instead of rewarding binary outcomes, the UPI will assign a fractional goal value (credit) to *every deliberate action* on the pitch based on how it changes the team's probability of scoring or conceding.

## 3. The Dataset
We are using the **StatsBomb Open Data** repository (e.g., Lionel Messi Data Biography, 2018 FIFA World Cup) accessed via the `statsbombpy` API. 
We will merge two specific data streams:
*   **Event Data:** To capture on-ball actions (passes, dribbles, shots, interceptions).
*   **StatsBomb 360 Data:** "Freeze-frame" tracking data that provides the exact $(x,y)$ locations of all visible teammates and opponents at the exact millisecond an event occurs. 

**Target Features to Extract:**
*   `location (x,y)` and `pass_end_location`: For spatial state transitions.
*   `under_pressure` (Boolean): To assess decision-making difficulty.
*   `shot_statsbomb_xg`: Baseline for terminal shot quality.
*   `360_frames`: To calculate spatial dominance and pitch control.

## 4. The Methodological Framework (The Models)
The agent will help code a multi-dimensional model synthesizing the following frameworks:

*   **Sarah Rudd's Markovian State-Value Model:** 
    We will divide the pitch into transient states (zones) and absorbing states (Goal, Ball Lost). We will calculate the transition matrix to determine the Goal Probability $P(Goal)$ of every state. Player credit is calculated as: $Credit = P_{after} - P_{before}$.
*   **Expected Goals (xG) & Expected Threat (xT):**
    xG (Logistic Regression) will be used to anchor the terminal shot states. xT will be used to reward ball progression that moves the ball into higher-threat zones.
*   **VAEP (Valuing Actions by Estimating Probabilities):**
    Unlike xT, VAEP will value *all* on-ball actions (including defensive tackles and interceptions) by calculating both the increase in $P(Score)$ and the decrease in $P(Concede)$ using the possession history.
*   **Event-Level Expected Possession Value (Discrete EPV):**
    Because StatsBomb 360 does not provide continuous tracking data, we will build a "discrete" EPV model. Using the 360 freeze frames, we will calculate Voronoi diagrams and Pitch Control at the exact moment of an event to reward players for off-ball space creation and bypassing defensive lines.

## 5. Normalization Strategy
To ensure the UPI can fairly compare a defensive midfielder to a striker, we will implement:
1.  **Statistical Normalization:** Z-score standardization of model features (e.g., distance, pressure) so scales are uniform.
2.  **Positional Normalization (Adjusted Alpha):** We will calculate the average performance score for specific positions and subtract this positional average from the individual's score. The final UPI will reflect how much a player *overperformed* compared to the average player in their exact role.