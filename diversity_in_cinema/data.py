import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as datetime


import numpy as np
from PIL import Image
import io
import sys
sys.path.insert(0,'..')

from google.cloud import storage

from utils import extract_face_mtcnn
from diversity_in_cinema.scraper import get_movies, download_one_frame_selenium, frame_urls, download_one_frame_bs
from cnn_model import predict_face
from params import *

def upload_file_to_gcp(file, movie_name):

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    bucket.blob(f'output/{movie_name}.csv').upload_from_string(file.to_csv(),
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
    if dataframe.empty:
        return None

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


def download_all_frames(title, frame_interval):

    urls = frame_urls(title, frame_interval)

    with ThreadPoolExecutor(max_workers=20) as executor:

        face_dict = {}
        frame_count = 0
        for frame_id, frame in executor.map(download_one_frame_bs,
                                        urls,
                                        timeout=None):

            frame_count += 1
            if frame_id is None:
                continue

            else:
                print(f"Grabbing frame number {frame_id}\n")
                faces_list = extract_face_mtcnn(frame)
                face_dict[frame_id] = faces_list

                if len(faces_list) > 0:
                    print(f"{len(faces_list)} faces detected.")
                    print("uploading face images to GCP\n")
                    for i, face in tqdm(enumerate(faces_list)):
                        image = Image.fromarray(face)

                        image_name = f"{frame_id}_face{i}.jpg"
                        upload_image_to_gcp(image, title, image_name)
                else:
                    print(f"No faces detected in frame number {frame_id}\n")

            print(f"{frame_count} / {len(urls)} frames scraped!\n")

        faces_df = pd.DataFrame(data={"frame": list(face_dict.keys()),
                                "faces": list(face_dict.values())})

        return faces_df


def main():

    """
    Function that conducts the following steps:
    1. Get a list of all movie title hosted on movie-screencaps.com

    """

    # generate list of all movies
    movie_list = list(get_movies().keys())

    for movie in tqdm(movie_list):


        print(f"[{datetime.datetime.now()}] - Scarping {movie}...\n")

        faces_df = download_all_frames(movie, frame_interval=3)

        print("classifying faces")
        df_analyzed = classify_faces(faces_df)

        if df_analyzed is None:
            continue

        # luis function faces_df
        # two dataframe -> google bucket

        print(f"[{datetime.datetime.now()}] - uploading results to GCP")
        upload_file_to_gcp(df_analyzed, f"{movie}")



if __name__ == "__main__":
    main()
