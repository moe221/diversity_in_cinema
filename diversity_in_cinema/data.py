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



def main():
    # generate list of all movies
    movie_list = list(get_movies().keys())[:10]

    for movie in tqdm(movie_list):

        print(f"extracting frames from {movie}...")
        df_movie_frames= download_all_frames(movie, frame_interval=1000)

        face_dict = {}

        print("extracting faces...")
        for frame, frame_id in zip(df_movie_frames["Image"], tqdm(df_movie_frames["Frame_Id"])):
            faces_list = extract_face_mtcnn(frame)
            face_dict[frame_id] = faces_list

            print("uploading face images to GCP")
            for i, face in enumerate(faces_list):
                out = np.array(face.permute(1, 2, 0))
                out = out.astype('uint8')
                image = Image.fromarray(out)

                image_name = f"{frame_id}_face{i}.jpg"
                upload_image_to_gcp(image, movie, image_name)


        # save as dataframe
        # faces_df = pd.DataFrame(data={"frame": list(face_dict.keys()),
        #                             "faces": list(face_dict.values())})





    #TODO classify faces
    #TODO ulpoad classification to gcp

if __name__ == "__main__":
    main()
