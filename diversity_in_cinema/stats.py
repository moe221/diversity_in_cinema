import pandas as pd
import numpy as np
from functools import reduce


def movie_stats(df_movie):
    '''
    input: pandas dataframe containing 4 columns with the following information of a movie: frame_number, face_id, gender, race
    output: 2 pandas dataframe with some statistics (data frame 1: gender statistics, data frame 2: race statistics)

     '''
    ###########################################################################
    #--------------------------------------------------------------------------
    # STATISTICS: number frames
    df = df_movie
    df2 = df.groupby(["frame_number"]).nunique()["gender"].reset_index()
    df2['gender_value'] = df['gender']
    df3 = df2[df2.gender == 1]

    #--------------------------------------------------------------------------
    # STATISTICS: average screentime
    # gender

    df_gender = df.groupby(['gender']).count().reset_index().drop(columns=['face_id', 'race'])
    df_gender['gender_screentime'] = (df_gender['frame_number'] / df_gender['frame_number'].sum()) * 100

    df_only = df3.groupby(['gender_value']).count().reset_index().drop(columns=['gender'])
    df_only = df_only.rename(columns={'frame_number': 'only_1_gender', 'gender_value': 'gender'})

    dfg = [df_gender, df_only]
    df_gender_final = reduce( lambda left, right: pd.merge(left, right, on='gender'), dfg)
    df_gender_final['only_1_screentime'] = (df_gender_final['only_1_gender'] / df_gender_final['only_1_gender'].sum()) * 100

    # race
    df_race = df.groupby(['race']).count().reset_index()

    df_race['race_screentime'] = (df_race['frame_number'] / df_race['frame_number'].sum()) * 100
    df_race = df_race.drop(columns=['face_id', 'gender'])

    # women
    woman_df = df_movie[df_movie['gender'] == 'Woman']

    women_by_race = woman_df.groupby(['race']).count().reset_index()
    women_by_race = women_by_race.drop(columns=['face_id', 'gender'])
    women_by_race = women_by_race.rename( columns={'frame_number': 'woman_frames'})

    # men
    man_df = df_movie[df_movie['gender'] == 'Man']
    men_by_race = man_df.groupby(['race']).count().reset_index()
    men_by_race = men_by_race.drop(columns=['face_id', 'gender'])
    men_by_race = men_by_race.rename(columns={'frame_number': 'man_frames'})

    dfs = [df_race, women_by_race, men_by_race]

    df_race_final = reduce( lambda left, right: pd.merge(left, right, on='race'), dfs)
    df_race_final['woman_r_screentime'] = df_race_final[ 'woman_frames'] / df_race_final['frame_number'] * 100
    df_race_final['man_r_screentime'] = df_race_final[ 'man_frames'] / df_race_final['frame_number'] * 100

    ############################################################################

    return df_gender_final, df_race_final
