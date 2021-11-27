import pandas as pd
from tqdm import tqdm

import numpy as np
from PIL import Image
import io
import sys
sys.path.insert(0,'..')

from google.cloud import storage

from utils import extract_face_mtcnn
from scraper import get_movies, download_all_frames
from cnn_model import predict_face
from params import *

def upload_file_to_gcp(file, movie_name):

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    bucket.blob(f'output/{movie_name}').upload_from_string(file.to_csv(),
                                                           'text/csv')

    print(f"Uploaded {movie_name} data to: {BUCKET_NAME}/output/{movie_name}")

def upload_image_to_gcp(image, movie_name, image_name):

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(f"face_images/{movie_name}/{image_name}")
    b = io.BytesIO()
    image.save(b, "jpeg")
    image.close()
    blob.upload_from_string(b.getvalue(), content_type="image/jpeg")


def classify_faces(dataframe):

    """
    A function which takes in a dataframe of frame number and the
    extracted face images in each frame as an array

    """

    df_list = []
    i = 0
    for frame, faces in zip(dataframe["frame"], tqdm(dataframe["faces"])):

        frame_list = []
        gender_list = []
        race_list = []
        face_id_list = []

        results = predict_face(faces)

        for faces in results.values():
            gender = faces["gender"]
            race = faces["dominant_race"]

            frame_list.append(frame)
            gender_list.append(gender)
            race_list.append(race)

            face_id_list.append(i)

            i += 1

        df = pd.DataFrame(data={"frame_number":frame_list,
                                "face_id":face_id_list,
                                "gender":gender_list,
                                "race":race_list})
        df_list.append(df)

    return pd.concat(df_list)


def main():

    """
    Function that conducts the following steps:
    1. Get a list of all movie title hosted on movie-screencaps.com

    """

    # generate list of all movies
    movie_list = list(get_movies().keys())[:10]

    for movie in tqdm(movie_list):

        print(f"extracting frames from {movie}...")
        df_movie_frames= download_all_frames(movie, frame_interval=500)

        face_dict = {}

        print("extracting faces...")
        for frame, frame_id in zip(df_movie_frames["Image"], tqdm(df_movie_frames["Frame_Id"])):
            faces_list = extract_face_mtcnn(frame)
            face_dict[frame_id] = faces_list

            print("uploading face images to GCP")
            for i, face in enumerate(faces_list):
                image = Image.fromarray(face)

                image_name = f"{frame_id}_face{i}.jpg"
                upload_image_to_gcp(image, movie, image_name)


        # save as dataframe
        faces_df = pd.DataFrame(data={"frame": list(face_dict.keys()),
                                      "faces": list(face_dict.values())})

        print("classifying faces")
        df_analyzed = classify_faces(faces_df)

        print("uploading results to GCP")
        upload_file_to_gcp(df_analyzed, movie)



if __name__ == "__main__":
    main()
