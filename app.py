
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import title

import preprocess, func
from func import overall_stat, year, wickets, batsman_analysis
from preprocess import process

# Load the datasets
delv = pd.read_csv('deliveries.csv')
mat = pd.read_csv('matches.csv')

# Process the data
df = process(delv, mat)

# Debug: check if 'season' exists
# st.write(df.columns)  # Check if 'season' exists after merging

# Sidebar menu
st.sidebar.title('IPL Analysis')
menu = st.sidebar.radio(
    'Select an Option',
    ('Winning-Stat', 'Batsman-Stat', 'Bowler-Stat')
)
# ----------------------\
#   Winning Stat         |
# ----------------------/
if menu == 'Winning-Stat':
    colors = ['#127538', '#660a26','#175206', '#6e0308']
    colors2 = ['#063152','#0b61a3','#131a1f','#5a6670']
    colors3 = ['#127538', '#660a26','#175206', '#6e0308']
    st.markdown("<h1 style='text-align: center; color: green;'> Team Analysis </h1>",
                unsafe_allow_html=True)
    years = df['season'].unique()
    # Sidebar filters for year range
    from_year = st.sidebar.selectbox('From Year', ['Overall'] + list(years))
    to_year = st.sidebar.selectbox('To Year', ['Overall'] + list(years))

    # Set from_year and to_year based on selection
    if from_year == 'Overall':
        from_year = df['season'].min()  # Earliest year
    if to_year == 'Overall':
        to_year = df['season'].max()  # Latest year

    # Select Team 1
    t1 = func.Team(df)
    T1 = st.selectbox('Team 1', t1)

    # Get opponents based on selected Team 1
    if T1 != 'select team':
        t2 = func.opp(df, T1,from_year,to_year)
        t2.insert(0, 'select team')
    else:
        t2 = ['select team']

    # Select Team 2
    T2 = st.selectbox('Team 2', t2)

    # Filter matches based on selected years and teams
    play = df[((df['season'] >= from_year) & (df['season'] <= to_year)) &
              (((df['team1'] == T1) & (df['team2'] == T2)) | ((df['team1'] == T2) & (df['team2'] == T1)))
              ]

    # Drop duplicate matches based on 'match_id'
    play = play.drop_duplicates(subset='match_id')

    # Number of matches and wins
    M = play.shape[0]
    win1 = play[play['winner'] == T1].shape[0]
    win2 = M - win1

    # Prepare data for the pie chart
    data = {'Team': [T1, T2], 'Wins': [win1, win2]}

    # Create the pie chart
    fig = px.pie(data, names='Team', values='Wins', title='Team Wins Distribution', hole=0.2,color_discrete_sequence=colors2)

    # Update pie chart trace for better visualization
    fig.update_traces(
        textinfo='percent',  # Show percentages on slices
        pull=[0.05, 0.05],  # Slightly "explode" the slices
        rotation=120,  # Set the starting rotation of the pie chart
        hoverinfo='label+percent+value',  # Display label, percentage, and value on hover
        marker=dict(line=dict(color='black', width=1)),  # Black border around slices
        textfont_size=20  # Increase text font size for better readability
    )
    # Display the pie chart
    if M==0 :
        st.markdown(f"<h3 style='text-align: center; color: red;'>No data found for the year between {from_year} to {to_year}</h3>",
                    unsafe_allow_html=True)
    else:
        st.plotly_chart(fig)

        x= play[play['toss_winner']==play['winner']].shape[0]
        st.markdown("<h3 style='text-align: center; color: red;'> Game Win acording to Toss Win </h3>",unsafe_allow_html=True)
        y= M-x
        dat = {'Outcome': ['Toss Winner Wins', 'Toss Winner Loses'], 'Count': [x, y]}
        fig = px.pie(dat, names='Outcome', values='Count', title='Win Distribution Based on Toss', hole=0.2,color_discrete_sequence=colors3)
        # Display the pie chart in Streamlit
        st.plotly_chart(fig)
        st.table(play[['toss_winner', 'toss_decision', 'venue', 'winner']])
# =============================================================================================
    st.markdown("<h1 style='text-align: center; color: green;'> Ground Analysis </h1>",
                unsafe_allow_html=True)
    df['normalized_venue'] = df['venue'].str.lower().str.split(',').str[0].str.strip()
    ven = df.drop_duplicates(subset='normalized_venue')['venue'].tolist()
    ven.sort()
    ven.insert(0,'--')
    v=st.selectbox('Select the venue',ven)
    if v != '--':

        x = df[df['venue'] == v].drop_duplicates(subset='match_id')
        m =x.shape[0]
        w= x[(x['toss_winner'] == x['winner'])].shape[0]
        f = x[(x['toss_decision'] == 'field')].shape[0]
        b = x[(x['toss_decision'] == 'bat')].shape[0]
        b_w = x[(x['toss_decision'] == 'bat') & (x['toss_winner'] == x['winner'])].shape[0]
        f_w = x[(x['toss_decision'] == 'field') & (x['toss_winner'] == x['winner'])].shape[0]
        data = {
            'Team': ['Batting Wins', 'Batting Losses'],
            'Wins': [b_w, b - b_w]
        }
        fig = px.pie(data, names='Team', values='Wins', title='Wins Distribution Based on Toss Decision = Bat', hole=0.2, color_discrete_sequence=colors)
        fig.update_traces(
            textinfo='percent',  # Show percentages on slices
            pull=[0.05, 0.05],  # Slightly "explode" the slices
            rotation=120,  # Set the starting rotation of the pie chart
            hoverinfo='label+percent+value',  # Display label, percentage, and value on hover
            marker=dict(line=dict(color='black', width=1)),  # Black border around slices
            textfont_size=20  # Increase text font size for better readability
        )
        st.plotly_chart(fig)

        data = {
            'Team': [ 'Fielding Wins', 'Fielding Losses'],
            'Wins': [ f_w, f - f_w]
        }
        fig = px.pie(data, names='Team', values='Wins', title='Wins Distribution Based on Toss Decision = Field', hole=0.2, color_discrete_sequence=colors2)
        fig.update_traces(
            textinfo='percent',  # Show percentages on slices
            pull=[0.05, 0.05],  # Slightly "explode" the slices
            rotation=120,  # Set the starting rotation of the pie chart
            hoverinfo='label+percent+value',  # Display label, percentage, and value on hover
            marker=dict(line=dict(color='black', width=1)),  # Black border around slices
            textfont_size=20  # Increase text font size for better readability
        )
        st.plotly_chart(fig)

        data = {
            'Team': [ 'Wins', 'Lose'],
            'Wins': [ w, m-w]
        }
        fig = px.pie(data, names='Team', values='Wins', title='Overall Wins Distribution Based on Toss Decision ', hole=0.2, color_discrete_sequence=colors3)
        fig.update_traces(
            textinfo='percent',  # Show percentages on slices
            pull=[0.05, 0.05],  # Slightly "explode" the slices
            rotation=120,  # Set the starting rotation of the pie chart
            hoverinfo='label+percent+value',  # Display label, percentage, and value on hover
            marker=dict(line=dict(color='black', width=1)),  # Black border around slices
            textfont_size=20  # Increase text font size for better readability
        )
        st.plotly_chart(fig)
# ----------------------\
#   Batsman Stat         |
# ----------------------/
if menu == 'Batsman-Stat':
    years = year(df)
    from_year = st.sidebar.selectbox('From Year', years)
    to_year = st.sidebar.selectbox('To Year', years)
    if from_year == 'Overall':
        from_year = df['season'].min()  # Get earliest year
    if to_year == 'Overall':
        to_year = df['season'].max()  # Get latest year


    YR = st.selectbox('Select Year', years)

    # Main title for the app
    st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>Top 3 Player of the Match Awards in {YR}</h1>",
                unsafe_allow_html=True)

    # Group by season and player of the match to count the awards
    awards = df[df['player_of_match'] == df['batter']]
    awards = awards.drop_duplicates(subset='match_id')
    awards = awards[awards['player_of_match']==awards['batter']
    ].groupby(['season', 'player_of_match']).size().reset_index(name='award_count')

    # Filter for the selected year
    yrs = awards[awards['season'] == YR]

    # Sort players by award count and select the top 3
    top3_mvp = yrs.nlargest(3, 'award_count')

    # Check if there is any data for the selected year
    if not top3_mvp.empty:
        # Create 3 columns to display the details for each top player
        for index, row in top3_mvp.iterrows():
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"<h2 style='text-align: center;'>Season</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; color: #FF5733;'>{row['season']}</h3>",
                            unsafe_allow_html=True)

            with col2:
                st.markdown(f"<h2 style='text-align: center;'>Player</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; color: #33C3F0;'>{row['player_of_match']}</h3>",
                            unsafe_allow_html=True)

            with col3:
                st.markdown(f"<h2 style='text-align: center;'>Awards Won</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; color: #FFC300;'>{row['award_count']}</h3>",
                            unsafe_allow_html=True)

            # Add spacing between players
            st.markdown("<br><hr><br>", unsafe_allow_html=True)

    else:
        # Display a warning if no data is found for the selected year
        st.markdown(f"<h3 style='text-align: center; color: red;'>No data found for the year {YR}</h3>",
                    unsafe_allow_html=True)

    st.title('Batsman Analysis')
    batsman_stat = batsman_analysis(df, from_year, to_year)
    st.write(batsman_stat)


    player = df['batter'].unique().tolist()
    player.insert(0, 'select batsman')

    # Selectbox for selecting the player
    selected_player = st.selectbox('Select a Batsman', player)

    # Ensure a valid player is selected before calling the function
    if selected_player != 'select batsman':
        # Filter the data based on the selected player and year range
        runs = df[
            (df['season'] >= from_year) & (df['season'] <= to_year)
            ].groupby(['batter', 'season'])['batsman_runs'].sum().reset_index()

        # Filter the runs DataFrame for the selected player
        runs = runs[runs['batter'] == selected_player]

        # Convert 'season' to string to treat it as a categorical axis
        runs['season'] = runs['season'].astype(str)

        # Plot the line chart with markers
        fig = px.line(
            runs,
            x="season",
            y="batsman_runs",
            markers=True,
            title=f"{selected_player}'s Runs per Season",
            labels={"season": "Year", "batsman_runs": "Runs"}
        )

        # Show the figure
        st.title(f"{selected_player}'s Total Runs Over the Years")
        st.plotly_chart(fig)
        # fig.show()
    else:
        st.write("Please select a valid batsman.")

    x = df[
        (df['batter'] == selected_player) &
        (df['is_wicket'] == 1) &
        (df['season'] >= from_year) & (df['season'] <= to_year)
        ].groupby('bowler')['is_wicket'].count().sort_values(ascending=False).reset_index()

    # Rename column for clarity
    x = x.rename(columns={'is_wicket': 'Dismissed'})

    # Get runs conceded by bowlers to V Kohli in 2024
    y = df[
        (df['batter'] == selected_player) &
        (df['season'] >= from_year) & (df['season'] <= to_year)
        ].groupby('bowler')['batsman_runs'].sum().sort_values(ascending=False).reset_index()

    # Get the number of balls bowled to V Kohli, excluding wides and no-balls
    z = df[
        (df['batter'] == selected_player) &
        (~df['extras_type'].isin(['wides', 'noballs'])) &
        (df['season'] >= from_year) & (df['season'] <= to_year)
        ].groupby('bowler')['ball'].count().sort_values(ascending=False).reset_index()
    z['Strike_Rate'] = ((y['batsman_runs'] / z['ball']) * 100).round(2)
    # Rename column for clarity
    z = z.rename(columns={'ball': 'Balls'})
    y = y.rename(columns={'batsman_runs': 'Runs'})
    # Merging all DataFrames on the 'bowler' column
    res = x.merge(y, on='bowler').merge(z, on='bowler')

    # Show the final result
    res = res.sort_values(by=['Dismissed', 'Strike_Rate'], ascending=[False, True])
    weak =    res.sort_values(by=['Dismissed', 'Strike_Rate'], ascending=[True, False])
    strong=res.head(3)
    weak=weak.head(3)
    st.header(f'Strong Bowler against {selected_player}')
    st.table(strong)
    st.header(f'Weak Bowler against {selected_player}')
    st.table(weak)

#----------------------\
#   Bowler Stat         |
#----------------------/

if menu == 'Bowler-Stat':
    years = year(df)
    from_year = st.sidebar.selectbox('From Year', years)
    to_year = st.sidebar.selectbox('To Year', years)

    if from_year == 'Overall':
        from_year = df['season'].min()  # Get earliest year
    if to_year == 'Overall':
        to_year = df['season'].max()  # Get latest year

    YR = st.selectbox('Select Year', years)

    # Main title for the app
    st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>Top 3 MVP Bowler in {YR}</h1>",
                unsafe_allow_html=True)

    # Group by season and player of the match to count the awards
    awards = df[df['player_of_match'] == df['bowler']]
    awards = awards.drop_duplicates(subset='match_id')
    awards = awards[awards['player_of_match']==awards['bowler']
    ].groupby(['season', 'player_of_match']).size().reset_index(name='award_count')

    # Filter for the selected year
    yrs = awards[awards['season'] == YR]

    # Sort players by award count and select the top 3
    top3_mvp = yrs.nlargest(3, 'award_count')

    # Check if there is any data for the selected year
    if not top3_mvp.empty:
        # Create 3 columns to display the details for each top player
        for index, row in top3_mvp.iterrows():
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"<h2 style='text-align: center;'>Season</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; color: #FF5733;'>{row['season']}</h3>",
                            unsafe_allow_html=True)

            with col2:
                st.markdown(f"<h2 style='text-align: center;'>Player</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; color: #33C3F0;'>{row['player_of_match']}</h3>",
                            unsafe_allow_html=True)

            with col3:
                st.markdown(f"<h2 style='text-align: center;'>Awards Won</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center; color: #FFC300;'>{row['award_count']}</h3>",
                            unsafe_allow_html=True)

            # Add spacing between players
            st.markdown("<br><hr><br>", unsafe_allow_html=True)

    else:
        # Display a warning if no data is found for the selected year
        st.markdown(f"<h3 style='text-align: center; color: red;'>No data found for the year {YR}</h3>",
                    unsafe_allow_html=True)

    # Display the wickets data
    wickets_data = overall_stat(df, int(from_year), int(to_year))
    st.title('Top 50 wicket taker')
    st.table(wickets_data)

    # plot
    all_bowler=df['bowler'].dropna().unique().tolist()
    all_bowler.sort()
    selected_bowler = st.selectbox('Select a Bowler', all_bowler)

    # progress tally
    # ----------------------
    wick = wickets(df,from_year,to_year)
    # Convert 'season' to integers (if not already)
    wick['season'] = pd.to_numeric(wick['season'], errors='coerce').fillna(0).astype(int)

    # Convert 'is_wicket' to numeric values (if not already)
    wick['is_wicket'] = pd.to_numeric(wick['is_wicket'], errors='coerce').fillna(0).astype(int)

    # Ensure the 'bowler' column is a string
    wick['bowler'] = wick['bowler'].astype(str)

    # Now filter the selected bowler and plot
    bowler_data = wick[wick['bowler'] == selected_bowler]

    # Plot the bar chart
    # fig = px.line(bowler_data, x="season", y="is_wicket", title=f"{selected_bowler} Total Wickets Over the Years")
    fig = px.line(
        bowler_data,
        x="season",
        y="is_wicket",
        title=f"{selected_bowler} Total Wickets Over the Years",
        markers=True  # This should be part of px.line
    )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Total Wickets Over the Years",
        # title_x=0.,  # Center the title
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        hovermode="x unified"
    )

    # Display title and chart in Streamlit
    st.title(f"{selected_bowler}'s Total Wickets Over the Years")
    st.plotly_chart(fig)




