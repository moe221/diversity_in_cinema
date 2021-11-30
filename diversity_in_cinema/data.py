import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import datetime as datetime

from PIL import Image
import io

from google.cloud import storage

from diversity_in_cinema.utils import extract_face_mtcnn
from diversity_in_cinema.scraper import get_movies, download_one_frame_selenium, frame_urls, download_one_frame_bs
from diversity_in_cinema.cnn_model import predict_face
from diversity_in_cinema.params import *

def upload_file_to_gcp(file, sub_directory,file_name):

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    bucket.blob(f'{sub_directory}/{file_name}').upload_from_string(file.to_csv(index=False),
                                                           'text/csv')

    print(f"Uploaded {file_name} data to: {BUCKET_NAME}/{sub_directory}/{file_name}")

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


def download_all_frames(title, frame_interval, workers):

    urls = frame_urls(title, frame_interval)

    # grab less frames from 4k titles because of memory issues
    if "[4K]" in title:
        while len(urls) > 6500:

            frame_interval += 1
            urls = frame_urls(title, frame_interval)

    with ThreadPoolExecutor(max_workers=workers) as executor:

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

        return faces_df, len(urls)



def add_to_overview(extracted_frames, total_frames, face_count, movie_name):

    """
    Function to create, update and upload a summary dataframe of all analyzed
    movies
    """

    data = {"title": [movie_name],
            "number_of_frames": [total_frames],
            "extracted_frames":[extracted_frames],
            "detected_faces": [face_count]}

    df = pd.DataFrame(data)

    try:
        # load image labels
        df_summary = pd.read_csv(f"gs://{BUCKET_NAME}/output/summary.csv", index_col=None)
        df_updated = pd.concat([df_summary, df], axis=0)
        upload_file_to_gcp(df_updated, "output", "summary.csv")

    except FileNotFoundError:
        upload_file_to_gcp(df, "output", "summary.csv")


def main(movie_list, frame_interval, workers):

    """
    Function that conducts the following steps:
    1. Get a list of all movie title hosted on movie-screencaps.com

    """

    for movie in tqdm(movie_list):

        # check if movie was already processed:

        client = storage.Client()
        processed_movies = [str(x).split(",")[1].replace(".csv", "").replace("output/", "").strip() for x in client.list_blobs(BUCKET_NAME, prefix='output')]
        if movie.strip() in processed_movies:
            continue


        print(f"[{datetime.datetime.now()}] - Scarping {movie}...\n")
        try:
            faces_df, total_frames = download_all_frames(movie, frame_interval, workers)
        except:
            continue

        print("classifying faces")
        df_analyzed = classify_faces(faces_df)

        if df_analyzed is None:
            continue

        # luis function faces_df
        # two dataframe -> google bucket

        print(f"[{datetime.datetime.now()}] - uploading results to GCP")
        upload_file_to_gcp(df_analyzed, "output", f"{movie}.csv")

        # update overview file
        face_count = len(df_analyzed)
        extracted_frames = len(faces_df)

        add_to_overview(extracted_frames, total_frames, face_count, movie)



if __name__ == "__main__":

    movie_list_df = pd.read_csv(
        f"gs://{BUCKET_NAME}/data/shuffled_movie_list.csv", index_col=None)

    movie_list = movie_list_df["movies"].values
    main(movie_list[: 184], frame_interval=3, workers=50)


    # next [184: 366]
    # next [366: 550]
    # next [550: 733]
