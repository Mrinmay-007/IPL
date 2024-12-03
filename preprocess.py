
import pandas as pd

def process(delv, mat):
    # Ensure the 'season' is processed correctly; extract and convert to integer
    mat['season'] = mat['season'].apply(lambda x: int(str(x)[:4]))  # Convert to integer after extracting year

    # Merge the match and delivery dataframes based on match_id and id
    df = delv.merge(mat, left_on='match_id', right_on='id', how='left')

    return df

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



