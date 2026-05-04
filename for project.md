Dataset Selection and Feature Extraction for the Unified Performance Index
(UPI)
This project leverages StatsBomb Open Data, an industry gold standard that makes professional,
elite-level football datasets publicly available via their GitHub repository. Specifically, we are extracting
data from matches that include 360-degree tracking data, such as the 2018 FIFA World Cup and the
Lionel Messi Data Biography. The dataset contains a large number of records, with over 200,000 events
in total.
Feature Extraction: Strategy Raw football event datasets are highly sparse, often containing over 100
columns where many are irrelevant to non-terminal actions.
To build our Markovian State-Value framework and Event-Level Expected Possession Value (EPV)
model, we must aggressively filter out the noise. The core variables we extract are:
● location (x,y) and pass_end_location: These features represent the ball’s location before and
after each action, allowing it to be mapped to specific states in a Markov model, track spatial
progression across pitch zones, and ultimately compute Expected Threat (xT) by assessing how
positional changes affect the probability of scoring.
● under_pressure: A boolean indicating defensive pressing, which differentiates between safe
states and dangerous, high-pressure states. It identifies whether an action occurs while the player
is being actively challenged by opponents. It adds contextual information about the surrounding
defensive situation, enabling the model to reflect the level of difficulty associated with each
action.
● shot_statsbomb_xg: This is a pre-calculated probability that a shot results in a goal. It serves as
an anchor for terminal states in the Markov model, ensuring that scoring actions are grounded in
realistic probabilities.
● 360_frames: The 360° freeze-frame data records the positions of all visible players at each event.
It represents the full game context, including team shape and player positioning, and supports
advanced spatial analysis such as pitch control.
Visualizations and Purpose: To accurately value a player's decision-making (and solve the outcome
bias known as the "FIFA Paradox"), spatial coordinates alone are insufficient.
Visualization 1: Raw Event Data (Spatial Locations): As seen in this traditional pitch map, extracting just
the location and pass_end_location shows us what happened. However, it completely ignores the
defensive context, failing to capture the difficulty of the passes


The purpose of using this specific tracking data is to calculate "Pitch Control" (Event-Level EPV).
Visualizing the defense's structural shape proves why these features are strictly necessary: they allow the
model to mathematically reward defensive midfielders and playmakers for executing actions that
successfully bypass opponent pressure, quantifying the "invisible" tactical contributions that traditional
stats ignore.


for visualization see at @notebooks/src/visualization