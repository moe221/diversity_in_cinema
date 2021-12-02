from facenet_pytorch import MTCNN, extract_face
from PIL import Image

from tensorflow import keras
from tensorflow.python.lib.io import file_io
from diversity_in_cinema.params import *


import os
from io import BytesIO
import io

from google.cloud import storage

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def extract_face_mtcnn(image):

    """
    Function which given an image resturns a list of faces all detected
    faces

    """

    mtcnn = MTCNN(keep_all=True,
                      min_face_size=30,
                      post_process=False,
                      image_size=224)
    face_list = []

    img = Image.fromarray(image)
    boxes, probs= mtcnn.detect(img)

    if boxes is not None:
        for box, prob in zip(boxes, probs):

            # return only faces detcted with 97% or higher
            # certainty
            if prob >= 0.97:

                face = extract_face(img, box)
                out = np.array(face.permute(1, 2, 0))
                out = out.astype('uint8')
                face_list.append(out)

    return face_list

def gcp_file_names(bucket_name, subfolders):

    """
    Function ro grab file names from a GCP bucket directory

    Parameters:

    bucket_name: Name of GCP bucket
    subfolders: complete subfolder path as a string where file names should
                be retrieved from in the format folder_1/folder_2/.../folder_n

    """

    client = storage.Client()
    file_names = [str(x).split(f"{subfolders}/")[1].\
        split(".csv")[0].\
            strip() + ".csv" for x in \
                client.list_blobs(bucket_name, prefix=subfolders)]

    return file_names


def upload_file_to_gcp(file, bucket_name, file_name, file_type="text/csv"):

    """
    Function to upload a file to a GCP bucket

    """

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    bucket.blob(f'{file_name}').upload_from_string(file.to_csv(index=False),
                                                   file_type)
    print(
        f"Uploaded {file_name} data to: {bucket_name}/{file_name}")


def upload_image_to_gcp(image, bucket_name, image_name):

    """
    Function for uploading an image to a GCP Bucket

    """
    image = Image.fromarray(image)

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(f"{image_name}")

    b = io.BytesIO()
    image.save(b, "jpeg")
    image.close()

    blob.upload_from_string(b.getvalue(), content_type="image/jpeg")



def remove_duplicate_4k_titles(title_list):

    """
    Function for removing duplicate 4K movie titles

    """

    for title in title_list:
        if "[4K]" in title:
            if title.replace("[4K]","").strip() in title_list:
                title_list.remove(title)
    return title_list


class ModelCheckpointInGcs(keras.callbacks.ModelCheckpoint):

    """
    Modification of the ModelCheckpoint class that allows models to be
    uploaded directly to GCP

    """

    def __init__(
        self,
        filepath,
        gcs_dir: str,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        options=None,
        **kwargs,
    ):
        super().__init__(
            filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq,
            options=options,
            **kwargs,
        )
        self._gcs_dir = gcs_dir

    def _save_model(self, epoch, logs):
        super()._save_model(epoch, logs)
        filepath = self._get_file_path(epoch, logs)
        if os.path.isfile(filepath):
            with file_io.FileIO(filepath, mode="rb") as inp:
                with file_io.FileIO(
                    os.path.join(self._gcs_dir, filepath), mode="wb+"
                ) as out:
                    out.write(inp.read())


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
        'total_women_of_color': [df_new['women_of_color'].sum()],
        'only_men_count': only_men,
        'only_women_count': only_women
    }


    for cat in ["Man",
                "Woman",
                "asian",
                "black",
                "indian",
                "latino hispanic",
                "middle eastern",
                "white"] :

        new_key = "total" + "_" + cat.strip().replace(" ", "_")

        if cat.strip() not in df_new.columns:
            dict_stats[new_key] = [0]

        else:
            dict_stats[new_key] = [df_new[cat].sum()]


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
        df_new['only_men_count'] / df_new['total_frames'] * 100,
        'only_women':
        df_new['only_women_count'] / df_new['total_frames'] * 100,
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

    final_df = pd.concat([pd.DataFrame.from_dict(dict_stats), df_new], axis=1)

    return final_df


def getImagePixels(file, bucket_name):

    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(f"{file}")
    data = blob.download_as_string()

    img = Image.open(BytesIO(data))
    img = np.array(img)

    return img



if __name__ == "__main__":
    client = storage.Client()
    my_list = [str(x).split(f"output/")[1].\
        split(".csv")[0].\
            strip() for x in
    client.list_blobs(BUCKET_NAME, prefix="output")]

    print(my_list)
