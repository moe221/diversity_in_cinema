import pandas as pd
import numpy as np
from functools import reduce


def movie_stats(df_movie_location):
    '''
    input: .csv file containing 5 columns with the following information of a movie: unnamed:0, frame_number, face_id, gender and race
    output: 2 pandas dataframe with some statistics (data frame 1: gender statistics, data frame 2: race statistics)
     '''
    # Import the movie´s data (csv-> pandas dataframe)
    df_movie = pd.read_csv(df_movie_location)

    # clean up of unnecessary columns
    df_movie = df_movie.drop(columns=['Unnamed: 0'])

    # frames with faces
    face_frames = len(pd.unique(df_movie['frame_number']))

    # transformation frames to seconds (probably not needed)
    face_frames_sec = face_frames / 2
    ###########################################################################
    # STATISTICS: total number
    # women
    woman_df = df_movie[df_movie['gender'] == 'Woman']
    number_women = len(woman_df)

    # men
    man_df = df_movie[df_movie['gender'] == 'Man']
    number_men = len(man_df)
    #--------------------------------------------------------------------------
    # STATISTICS: races
    # white
    women_white = woman_df.groupby('race')['race'].count()['white']
    men_white = man_df.groupby('race')['race'].count()['white']

    # latino
    women_latino = woman_df.groupby('race')['race'].count()['latino hispanic']
    men_latino = man_df.groupby('race')['race'].count()['latino hispanic']

    # black
    women_black = woman_df.groupby('race')['race'].count()['black']
    men_black = man_df.groupby('race')['race'].count()['black']

    # asian
    women_asian = woman_df.groupby('race')['race'].count()['asian']
    men_asian = man_df.groupby('race')['race'].count()['asian']

    # middle-eastern
    women_middleeastern = woman_df.groupby(
        'race')['race'].count()['middle eastern']
    men_middleeastern = man_df.groupby(
        'race')['race'].count()['middle eastern']

    # indian
    women_indian = woman_df.groupby('race')['race'].count()['indian']
    men_indian = man_df.groupby('race')['race'].count()['indian']

    # STATISTICS: white versus non-white
    women_nonwhite = number_women - women_white
    men_nonwhite = number_men - men_white
    #--------------------------------------------------------------------------
    # STATISTICS: number frames
    df = df_movie
    df2 = df.groupby(["frame_number"]).nunique()["gender"].reset_index()
    df2['gender_value'] = df['gender']
    df3 = df2[df2.gender == 1]
    df3.groupby('gender_value')['gender_value'].count()

    # frames: gender (only women versus only men)
    only_women = len(df3[df3['gender_value'] == 'Woman'])
    only_men = len(df3[df3['gender_value'] == 'Man'])

    #--------------------------------------------------------------------------
    # STATISTICS: average screentime
    # gender
    only_women_perc = (only_women / face_frames) * 100
    only_men_perc = (only_men / face_frames) * 100

    df_gender = df.groupby(
        ['gender']).count().reset_index().drop(columns=['face_id', 'race'])
    df_gender['gender_screentime'] = (df_gender['frame_number'] /
                                      df_gender['frame_number'].sum()) * 100

    df_only = df3.groupby(['gender_value'
                           ]).count().reset_index().drop(columns=['gender'])
    df_only = df_only.rename(columns={
        'frame_number': 'only_1_gender',
        'gender_value': 'gender'
    })

    dfg = [df_gender, df_only]
    df_gender_final = reduce(
        lambda left, right: pd.merge(left, right, on='gender'), dfg)
    df_gender_final['only_1_screentime'] = (
        df_gender_final['only_1_gender'] /
        df_gender_final['only_1_gender'].sum()) * 100

    # race
    df_race = df.groupby(['race']).count().reset_index()
    df_race['frame_number'].sum()

    df_race['race_screentime'] = (df_race['frame_number'] /
                                  df_race['frame_number'].sum()) * 100
    df_race = df_race.drop(columns=['face_id', 'gender'])

    women_by_race = woman_df.groupby(['race']).count().reset_index()
    women_by_race = women_by_race.drop(columns=['face_id', 'gender'])
    women_by_race = women_by_race.rename(
        columns={'frame_number': 'woman_frames'})

    men_by_race = man_df.groupby(['race']).count().reset_index()
    men_by_race = men_by_race.drop(columns=['face_id', 'gender'])
    men_by_race = men_by_race.rename(columns={'frame_number': 'man_frames'})

    dfs = [df_race, women_by_race, men_by_race]

    df_race_final = reduce(
        lambda left, right: pd.merge(left, right, on='race'), dfs)
    df_race_final['woman_r_screentime'] = df_race_final[
        'woman_frames'] / df_race_final['frame_number'] * 100
    df_race_final['man_r_screentime'] = df_race_final[
        'man_frames'] / df_race_final['frame_number'] * 100

    ###########################################################################

    return df_race_final, df_gender_final
