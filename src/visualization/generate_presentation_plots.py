import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from statsbombpy import sb
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def fetch_and_extract_data(match_id):
    """
    1. Data Acquisition & Feature Extraction
    """
    print(f"Fetching event data for match_id: {match_id} (e.g., 2022 FIFA World Cup Final)...")
    events_df = sb.events(match_id=match_id)
    
    # Extract only the specific features we need for our models
    features_to_keep = ['id', 'type', 'location', 'pass_end_location', 'under_pressure', 'shot_statsbomb_xg']
    # statsbombpy might not perfectly align these columns in every row, so we extract safely
    existing_features = [f for f in features_to_keep if f in events_df.columns]
    
    extracted_events = events_df[existing_features].copy()
    
    # Ensure under_pressure is boolean
    if 'under_pressure' in extracted_events.columns:
        extracted_events['under_pressure'] = extracted_events['under_pressure'].fillna(False).astype(bool)
        
    print(f"Fetching 360 frame data for match_id: {match_id}...")
    try:
        frames_dict = sb.frames(match_id=match_id, fmt='dict')
        frames_df = pd.DataFrame(frames_dict)
    except AttributeError:
        # Fallback if the user has an older version of statsbombpy
        print("Note: Update statsbombpy if `frames` method issues arise.")
        frames_df = pd.DataFrame() 
        
    return extracted_events, frames_df

def plot_spatial_event_map(events):
    """
    2. Visualization 1: Spatial Event Map (The 'What')
    """
    print("Generating Visualization 1: Spatial Event Map...")
    
    # Filter for some passes to show
    passes = events[events['type'] == 'Pass'].dropna(subset=['location', 'pass_end_location'])
    # Take a sample of the first 25 passes to avoid clutter
    sample_passes = passes.head(25)
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(10, 7))
    fig.patch.set_facecolor('#22312b')
    
    for idx, row in sample_passes.iterrows():
        try:
            start_x, start_y = row['location']
            end_x, end_y = row['pass_end_location']
            # Plot the pass as an arrow
            pitch.arrows(start_x, start_y, end_x, end_y, width=2,
                         headwidth=10, headlength=10, color='cyan', ax=ax, alpha=0.7)
            pitch.scatter(start_x, start_y, color='white', s=30, ax=ax, zorder=2)
        except (ValueError, TypeError):
            continue
            
    title_text = "Raw Event Data (Spatial Locations)\nNotice how we know *what* happened, but not the defensive context."
    ax.set_title(title_text, fontsize=16, color='white', pad=20, fontfamily='sans-serif')
    
    plt.tight_layout()
    output_path = '1_spatial_event_map.png'
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"-> Saved {output_path}")

def plot_freeze_frame_map(events, frames_df):
    """
    3. Visualization 2: StatsBomb 360 Freeze Frame (The 'Context')
    """
    print("Generating Visualization 2: 360 Freeze Frame Map...")
    
    if frames_df.empty:
        print("Could not find 360 frames. Skipping visualization 2.")
        return

    # Find passes under pressure
    passes = events[(events['type'] == 'Pass') & (events['under_pressure'] == True)].dropna(subset=['location'])
    
    # Merge with frames
    merged = pd.merge(passes, frames_df, left_on='id', right_on='event_uuid', how='inner')
    
    if merged.empty:
        print("No perfectly matching 360 frames found for passes under pressure.")
        return

    # Take the first event under pressure
    target_event = merged.iloc[0]
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(10, 7))
    fig.patch.set_facecolor('#22312b')
    
    # Plot the visible area polygon
    if 'visible_area' in target_event and isinstance(target_event['visible_area'], list):
        # StatsBomb 360 visible area is a list of points e.g. [x1, y1, x2, y2, ...]
        coords = target_event['visible_area']
        try:
            visible_area = np.array(coords).reshape(-1, 2)
            pitch.polygon([visible_area], ax=ax, fill=True, color='white', alpha=0.1)
        except Exception as e:
            print(f"Warning: Could not plot visible area polygon. {e}")
            
    # Plot players
    if 'freeze_frame' in target_event and isinstance(target_event['freeze_frame'], list):
        for player in target_event['freeze_frame']:
            loc = player['location']
            teammate = player['teammate']
            actor = player.get('actor', False)
            
            # Colors: Blue for teammate, Red for opponent
            color = '#1f77b4' if teammate else '#d62728'
            edgecolor = 'white' if actor else 'none'
            linewidth = 3 if actor else 0
            size = 250 if actor else 120
            
            pitch.scatter(loc[0], loc[1], s=size, color=color, 
                          edgecolors=edgecolor, linewidths=linewidth, ax=ax, zorder=3, alpha=0.9)
                          
    # Plot the ball flight (pass location -> end location)
    if 'location' in target_event and 'pass_end_location' in target_event:
        start_x, start_y = target_event['location']
        end_x, end_y = target_event['pass_end_location']
        pitch.arrows(start_x, start_y, end_x, end_y, width=2,
                     headwidth=10, headlength=10, color='yellow', ax=ax, alpha=0.8, zorder=4)

    title_text = "StatsBomb 360 Data: Defensive Pressure & Pitch Control\nProving why freeze frames are necessary for Event-Level EPV modeling."
    ax.set_title(title_text, fontsize=16, color='white', pad=20, fontfamily='sans-serif')
    
    plt.tight_layout()
    output_path = '2_360_freeze_frame.png'
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"-> Saved {output_path}")

def main():
    # 2022 World Cup Final: Argentina vs France (Has robust 360 Data)
    WC_FINAL_MATCH_ID = 3869151
    
    events_df, frames_df = fetch_and_extract_data(WC_FINAL_MATCH_ID)
    
    if not events_df.empty:
        plot_spatial_event_map(events_df)
        plot_freeze_frame_map(events_df, frames_df)

if __name__ == "__main__":
    main()
