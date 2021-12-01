from facenet_pytorch import MTCNN, extract_face
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def extract_face_mtcnn(image):

    """
    Function which given an image resturns a list of faces all detected
    faces
    """

    mtcnn = MTCNN(keep_all=True,
                      min_face_size=35,
                      post_process=False)
    face_list = []

    print("Extracting faces...")

    img = Image.fromarray(image)

    boxes, probs= mtcnn.detect(img)

    if boxes is not None:
        for box, prob in zip(boxes, probs):

            # return only faces detcted with 99% or higher
            # certanty
            if prob >= 0.99:

                face = extract_face(img, box)

                out = np.array(face.permute(1, 2, 0))
                out = out.astype('uint8')

                face_list.append(out)

        print(f"{len(face_list)} faces detected.")

    return face_list

def output_preproc(df):
    '''One Hot Encodes gender and race data from output dataframe'''
    ohe_g = OneHotEncoder(sparse=False)

    ohe_g.fit(df[['gender']])
    gender_encoded = ohe_g.transform(df[['gender']])
    results_g = gender_encoded.T

    for i, cat in enumerate(ohe_g.categories_[0]):
        df[cat] = results_g[i]

    ohe_r = OneHotEncoder(sparse=False)

    ohe_r.fit(df[['race']])
    race_encoded = ohe_r.transform(df[['race']])
    results_r = race_encoded.T

    for i, cat in enumerate(ohe_r.categories_[0]):
        df[cat] = results_r[i]

    return df


def woman_of_color(x):
    if 'Woman' in x and 'white' not in x:
        return 1
    return 0

def baseline_stats(df):
    '''Creates a dataframe of engineered/composite features from preprocessed output'''
    df_new = output_preproc(df)

    df_new['women_of_color'] = df_new['gender'] + ' ' + df_new['race']
    df_new['women_of_color'] = df_new['women_of_color'].apply(woman_of_color)

    df_new = df_new.groupby('frame_number').sum()

    df_new['face_count'] = df_new['Man'] + df_new['Woman']

    only_men = len(df_new[df_new['Woman'] == 0])
    only_women = len(df_new[df_new['Man'] == 0])

    dict_stats = {
        'total_frames': [len(df_new)],
        'total_seconds': [len(df_new) / 2],
        'total_faces': [df_new['face_count'].sum()],
        'total_men': [df_new['Man'].sum()],
        'total_women': [df_new['Woman'].sum()],
        'total_asian': [df_new['asian'].sum()],
        'total_black': [df_new['black'].sum()],
        'total_indian': [df_new['indian'].sum()],
        'total_latino_hispanic': [df_new['latino hispanic'].sum()],
        'total_middle_eastern': [df_new['middle eastern'].sum()],
        'total_white': [df_new['white'].sum()],
        'total_women_of_color': [df_new['women_of_color'].sum()],
        'only_men': only_men,
        'only_women': only_women
    }

    df_stats = pd.DataFrame.from_dict(dict_stats)

    return df_stats


def final_stats(df):
    '''Creates final statistical dataframe for use in dashboard'''
    df_new = baseline_stats(df)

    dict_stats = {
        'man_screentime':
        df_new['total_men'] / df_new['total_faces'] * 100,
        'woman_screentime':
        df_new['total_women'] / df_new['total_faces'] * 100,
        'only_men':
        df_new['only_men'] / df_new['total_frames'] * 100,
        'only_women':
        df_new['only_women'] / df_new['total_frames'] * 100,
        'asian_screentime':
        df_new['total_asian'] / df_new['total_faces'] * 100,
        'black_screentime':
        df_new['total_black'] / df_new['total_faces'] * 100,
        'indian_screentime':
        df_new['total_indian'] / df_new['total_faces'] * 100,
        'latino_hispanic_screentime':
        df_new['total_latino_hispanic'] / df_new['total_faces'] * 100,
        'middle_eastern_screentime':
        df_new['total_middle_eastern'] / df_new['total_faces'] * 100,
        'white_screentime':
        df_new['total_white'] / df_new['total_faces'] * 100,
        'women_of_color':
        df_new['total_women_of_color'] / df_new['total_frames'] * 100
    }

    final_df = pd.DataFrame.from_dict(dict_stats)

    return final_df
