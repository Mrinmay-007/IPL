import  pandas as pd
import plotly.express as px



def year(df):
    yr = df['season'].unique().tolist()
    yr.insert(0, 'Overall')  # Insert 'Overall' at the start
    return yr

def Team(df):
    T = df['team1'].drop_duplicates().tolist()
    T.insert(0,'select team')
    return T

def opp(df,T1,from_yr,to_yr):
    T = df[ ((df['team1'] == T1) | (df['team2'] == T1)) &
            ((df['season'] >= from_yr) & (df['season'] <= to_yr))
        ]
    opponents = pd.concat([T['team1'], T['team2']])
    opponents = opponents[opponents != T1].drop_duplicates().tolist()
    opponents.sort()
    return opponents

def wickets(df, from_yr, to_yr):
    wick = (
        df[
            (df['season'] >= from_yr) &
            (df['season'] <= to_yr) &
            (df['is_wicket'] == 1) &  # Wicket must be taken
            (df['super_over'] == 'N') &  # Exclude super overs
            (
                    (df['dismissal_kind'] != 'run out') |  # Either it's not a run out
                    ((df['dismissal_kind'] == 'run out') & (df['fielder'] == df['bowler']))
            )
            ]
        .groupby(['bowler', 'season'])['is_wicket']  # Include 'season' in grouping
        .sum()
        .reset_index()
    )
    return wick


def overall_stat(df, from_yr, to_yr):

    wick = wickets(df, from_yr, to_yr)

    # Group by bowler and season to get total runs conceded
    runs = df[
        (df['extras_type'] != 'legbyes') &
        (df['extras_type'] != 'byes')
        ].groupby(['bowler', 'season'])['total_runs'].sum().reset_index()

    # Merge the wickets data with the runs data on 'bowler' and 'season'
    merged_df = runs.merge(wick, on=['bowler', 'season'], how='left')

    # Calculate the bowling average as total_runs / is_wicket
    merged_df['bowl_avg'] = merged_df['total_runs'] / merged_df['is_wicket']
    merged_df.fillna(0, inplace=True)
    # Round bowling average to 2 decimal places
    merged_df['bowl_avg'] = merged_df['bowl_avg'].round(2)

    # Ensure wickets are displayed as integers
    merged_df['is_wicket'] = merged_df['is_wicket'].astype(int)
    # Filter based on season, not bowler
    filtered_bowler = merged_df[(merged_df['season'] >= from_yr) & (merged_df['season'] <= to_yr)]

    # Sort by season and fill missing values
    filtered_bowler.fillna(0, inplace=True)
    filtered_bowler.rename(columns={'is_wicket': 'Total_wicket'}, inplace=True)


    filtered_bowler = filtered_bowler.sort_values(by='Total_wicket', ascending=False).head(50)

    return filtered_bowler

# def wickets_plot(df, from_yr, to_yr,bowler):
    # fig = px.line(wick, x="season", y="is_wicket")
    # st.title(bowler + " Medal Tally over the years")
    # st.plotly_chart(fig)

def batsman_analysis(df, from_yr, to_yr):
    # Step 1: Filter the DataFrame for the specified range of seasons and calculate matches played
    match = df[
        (df['season'] >= from_yr) & (df['season'] <= to_yr)
    ].groupby('batter')['match_id'].nunique().reset_index()

    # Rename columns for clarity
    match.columns = ['batter', 'Match']

    # Step 2: Calculate total runs
    runs = df[
        (df['season'] >= from_yr) & (df['season'] <= to_yr)
    ].groupby('batter')['batsman_runs'].sum().reset_index()

    # Rename columns
    runs.columns = ['batter', 'Runs']

    # Step 3: Count valid balls faced (exclude wides and no-balls)
    balls_faced = df[
        (df['season'] >= from_yr) & (df['season'] <= to_yr) &
        (~df['extras_type'].isin(['wides', 'noballs']))
    ].groupby('batter')['ball'].count().reset_index()

    # Rename columns
    balls_faced.columns = ['batter', 'Balls_faced']

    # Step 4: Calculate strike rate
    balls_faced['Strike_Rate'] = ((runs['Runs'] / balls_faced['Balls_faced']) * 100).round(3)

    # Step 5: Calculate sixes
    six = df[
        (df['season'] >= from_yr) & (df['season'] <= to_yr) & (df['batsman_runs'] == 6)
    ].groupby('batter')['batsman_runs'].count().reset_index()

    # Rename columns
    six.columns = ['batter', '6s']

    # Step 6: Calculate fours
    four = df[
        (df['season'] >= from_yr) & (df['season'] <= to_yr) & (df['batsman_runs'] == 4)
    ].groupby('batter')['batsman_runs'].count().reset_index()

    # Rename columns
    four.columns = ['batter', '4s']

    # Step 7: Count dismissals
    dismiss = df[
        (df['season'] >= from_yr) & (df['season'] <= to_yr) & (df['is_wicket'] == 1) & (df['player_dismissed'].notna())
    ].groupby('player_dismissed')['is_wicket'].count().reset_index()

    # Rename columns
    dismiss.columns = ['batter', 'Dismissals']

    # Step 8: Calculate not outs (total matches played - dismissals)
    not_out = match.merge(dismiss, on='batter', how='left')
    not_out['Dismissals'] = not_out['Dismissals'].fillna(0)  # Fill NaNs for players with no dismissals
    not_out['Not_out'] = (not_out['Match'] - not_out['Dismissals']).astype(int)

    # Step 9: Calculate average (Runs / Dismissals, where dismissals > 0)
    not_out['Avg'] = (runs['Runs'] / not_out['Dismissals']).replace([float('inf'), 0], 0).round(3)

    # Step 10: Calculate centuries (100+ runs in a match) and half-centuries (50-99 runs in a match)
    match_runs = df[
        (df['season'] >= from_yr) & (df['season'] <= to_yr)
    ].groupby(['batter', 'match_id'])['batsman_runs'].sum().reset_index()

    # Filter for centuries (100+ runs)
    centuries = match_runs[match_runs['batsman_runs'] >= 100]
    century_count = centuries.groupby('batter')['batter'].count().reset_index(name='100s')

    # Filter for half-centuries (50 to 99 runs)
    half_centuries = match_runs[(match_runs['batsman_runs'] >= 50) & (match_runs['batsman_runs'] < 100)]
    half_cen_count = half_centuries.groupby('batter')['batter'].count().reset_index(name='50s')

    # Step 11: Merge all data together
    final_data = match.merge(runs, on='batter') \
                      .merge(balls_faced, on='batter') \
                      .merge(four, on='batter', how='left') \
                      .merge(six, on='batter', how='left') \
                      .merge(not_out[['batter', 'Not_out', 'Avg']], on='batter') \
                      .merge(half_cen_count, on='batter', how='left') \
                      .merge(century_count, on='batter', how='left')

    # Fill NaN values for players who may not have hit fours, sixes, half-centuries, or centuries
    final_data['4s'] = final_data['4s'].fillna(0).astype(int)
    final_data['6s'] = final_data['6s'].fillna(0).astype(int)
    final_data['50s'] = final_data['50s'].fillna(0).astype(int)
    final_data['100s'] = final_data['100s'].fillna(0).astype(int)

    # Sort the final data by Runs in descending order
    final_data = final_data.sort_values(by='Runs', ascending=False)

    # Return the final data with top 30 players
    return final_data[['batter', 'Match', 'Not_out', 'Runs', 'Balls_faced', 'Avg', 'Strike_Rate', '4s', '6s', '50s', '100s']]

#
# def batsman_career(df,player,from_yr,to_yr):
#     # player =df['batter'].nunique().tolist()
#
#     runs = df[
#     (df['season'] >= from_yr) & (df['season'] <= to_yr)
#     ].groupby(['batter', 'season'])['batsman_runs'].sum().reset_index()
#
#     runs = runs[runs['batter'] == player]
#
#     runs['season'] = runs['season'].astype(str)
#
# # Plot the line chart with markers
#     fig = px.line(
#         runs,
#         x="season",
#         y="batsman_runs",
#         markers=True,
#         title=f"{player}'s Runs per Season",
#         labels={"season": "Year", "batsman_runs": "Runs"}
#     )
# # Show the figure
#     fig.show()
#     return fig
