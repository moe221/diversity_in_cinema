import pandas as pd
from tqdm import tqdm
import datetime as datetime

from google.cloud import storage

from diversity_in_cinema.params import *
from diversity_in_cinema.scraper import get_movies
from diversity_in_cinema.statistics import create_stats_csv

from diversity_in_cinema.scraper import face_scraper
from diversity_in_cinema.cnn_model import predict_face
from diversity_in_cinema.utils import upload_file_to_gcp
from diversity_in_cinema.utils import remove_duplicate_4k_titles



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
        df_summary = pd.read_csv(f"gs://{BUCKET_NAME}/output/summary.csv",
                                 index_col=None)

        df_updated = pd.concat([df_summary, df], axis=0)

        upload_file_to_gcp(df_updated, BUCKET_NAME, "output/summary.csv")

    except FileNotFoundError:
        upload_file_to_gcp(df, BUCKET_NAME, "output/summary.csv")


def main(movie_list, frame_interval, workers):

    """
    Function that conducts the following steps:
    1. Get a list of all movie title hosted on movie-screencaps.com

    """

    for movie in tqdm(movie_list):

        # check if movie was already processed:
        client = storage.Client()
        processed_movies = [str(x).split(",")[1].\
            replace(".csv", "").\
                replace("output/", "").\
                    strip() for x in client.\
                        list_blobs(BUCKET_NAME, prefix='output')]

        if movie.strip() in processed_movies:
            continue


        print(f"[{datetime.datetime.now()}] - Scarping {movie}...\n")
        try:
            faces_df, total_frames = face_scraper(movie,
                                                  frame_interval,
                                                  workers)
        except:
            continue

        print("classifying faces")
        df_analyzed = classify_faces(faces_df)

        if df_analyzed is None:
            continue


        print(f"[{datetime.datetime.now()}] - uploading results to GCP")
        upload_file_to_gcp(df_analyzed, BUCKET_NAME, f"output/{movie}.csv")

        # update overview file
        face_count = len(df_analyzed)
        extracted_frames = len(faces_df)

        add_to_overview(extracted_frames, total_frames, face_count, movie)



if __name__ == "__main__":

    # scrape all movies on movie-screencaps.com
    movie_list = get_movies()
    movie_list = remove_duplicate_4k_titles()
    main(movie_list, frame_interval=3, workers=100)


    # testing
    # movie_list_df = pd.read_csv(
    #     f"gs://{BUCKET_NAME}/data/shuffled_movie_list.csv", index_col=None)

    # movie_list = movie_list_df["movies"].values
    # main(movie_list[550: 733], frame_interval=3, workers=50)

    # calculating statistics
    # create_stats_csv()

    # next [184: 366]
    # next [366: 550]
    # next [550: 733]
