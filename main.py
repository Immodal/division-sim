#%%
import numpy as np
import random
import pandas as pd
import plotly.express as px


class Elo:
    def __init__(self, elo_k):
        self.elo_k = elo_k

    # winIndicator:
    # 1.0 = rating1 won
    # 0.0 = rating2 won
    # 0.5 = draw
    def calculate_elo_delta(self, rating1, rating2, winIndicator):
        return self.elo_k * (winIndicator - self.calculate_elo_expected_win_rate(rating1, rating2))

    def calculate_elo_expected_win_rate(self, rating1, rating2):
        return 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))


class Participant:
    def __init__(self, id, true_elo, initial_elo, division):
        self.id = id
        self.true_elo = true_elo
        self.current_elo = initial_elo
        self.division = division


ELO = Elo(8)
INITIAL_ELO = 1600
N_PARTICIPANTS = 50
random.seed(24)
TRUE_ELOS = [round(np.random.normal(INITIAL_ELO, 100)) for i in range(N_PARTICIPANTS)]
TRUE_ELOS.sort(reverse=True)

#%%
random.seed(42)
N_DIVISIONS = 5
DIVISION_SIZE = 10
ROUNDS_PER_REVISION = 1
# Create participants
participants = []
for i in range(len(TRUE_ELOS)):
    participants.append(Participant(i, TRUE_ELOS[i], INITIAL_ELO, N_DIVISIONS-1))

# Dict for tracking matchup matrix
matches = { p1.id: {p2.id:0 for p2 in participants} for p1 in participants }
division_changes = []
# Shuffle so that participants start at random places rather than in order of true elo
random.shuffle(participants)
# Rounds
for r in range(100):
    exclude = set()
    for p1 in participants:
        # Dont match against self
        exclude.add(p1)
        for p2 in participants:
            # Dont repeat matches or match against another division
            if p2 in exclude or p2.division != p1.division: continue
            p1_wr = ELO.calculate_elo_expected_win_rate(p1.true_elo, p2.true_elo)
            p1_win = int(random.random() <= p1_wr)
            elo_delta = round(ELO.calculate_elo_delta(p1.current_elo, p2.current_elo, p1_win))
            p1.current_elo += elo_delta
            p2.current_elo -= elo_delta

            matches[p1.id][p2.id] += 1
            matches[p2.id][p1.id] += 1

    # Revise divisions
    if r%ROUNDS_PER_REVISION==0:
        participants.sort(key=lambda x: x.current_elo, reverse=True)
        for i in range(N_DIVISIONS):
            div_size = DIVISION_SIZE if i!=N_DIVISIONS-1 else len(participants)-i*DIVISION_SIZE
            for j in range(div_size):
                p = participants[i*DIVISION_SIZE+j]
                if p.division != i:
                    division_changes.append({"round": r, "participant": p.id, "old": p.division, "new": i})
                    p.division = i

# Results
matches_df = pd.DataFrame.from_dict(matches)
matches_df.to_csv(f"matrix{ROUNDS_PER_REVISION}.csv")
matches_df = pd.DataFrame.from_dict(matches)

division_changes_df = pd.DataFrame.from_dict(division_changes)
round_div_change_df = division_changes_df[['round','participant']].groupby("round").agg('count')
fig = px.bar(round_div_change_df, x=round_div_change_df.index, y='participant')
fig.show()
#%%