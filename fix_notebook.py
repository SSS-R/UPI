import json

with open('UPI_Final_Report.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 3: Fix Passing Network player count (Cell index 3 is cell_type code)
c3 = nb['cells'][3]
source3 = c3['source']
new_source3 = []
for line in source3:
    if 'team_passes = passes[passes["team"] == top_team].copy()' in line:
        new_source3.append('    # Filter until first substitution to only show the starting XI\n')
        new_source3.append('    subs = events_df[(events_df["type"] == "Substitution") & (events_df["team"] == top_team)]\n')
        new_source3.append('    first_sub = subs["minute"].min() if not subs.empty else 90\n')
        new_source3.append('    team_passes = passes[(passes["team"] == top_team) & (passes["minute"] < first_sub)].copy()\n')
    else:
        new_source3.append(line)
c3['source'] = new_source3

# Cell 5: Fix mplsoccer heatmap error (Cell index 5 is cell_type code)
c5 = nb['cells'][5]
source5 = c5['source']
new_source5 = []
for line in source5:
    if 'heatmap = pitch.heatmap(xT_surface.T, ax=ax, cmap=cmap, alpha=0.8)' in line:
        new_source5.append('heatmap = ax.imshow(xT_surface.T, extent=(0, 120, 80, 0), cmap=cmap, alpha=0.8, aspect="auto")\n')
    else:
        new_source5.append(line)
c5['source'] = new_source5

with open('fixed_notebook.json', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
