from tqdm.std import tqdm

from face_averaging.face_averaging import average_image
from diversity_in_cinema.params import BUCKET_NAME
from diversity_in_cinema.params import  BUCKET_NAME_STREAMLIT
from diversity_in_cinema.utils_cloud import bulk_get_image_pixels
from diversity_in_cinema.utils_cloud import upload_image_to_gcp
from diversity_in_cinema.utils_cloud import gcp_file_names
from google.cloud import storage


def main():
    bucket_location_prefix = 'face_images'

    # Get names of folders
    client = storage.Client()

    lst = client.list_blobs(BUCKET_NAME, prefix='face_images')
    processed_movies = set([str(x).split("/")[1] for x in lst])

    faces_created = [str(x).split("faces/")[1].split("_avg_face.jpg")[0] for x in client.
                                        list_blobs(BUCKET_NAME_STREAMLIT, prefix='faces')]

    # check if movie face was already created
    for movie_name in tqdm(processed_movies):
        if "Panther" in movie_name:
        # if movie_name in faces_created:
        #     continue

            print(f'Now Averaging: {movie_name}')
            movie_path = bucket_location_prefix + '/' + movie_name
            print(movie_path)

            # Get names of paths
            file_names = gcp_file_names(BUCKET_NAME, movie_path)
            print(file_names)

            # Lots of folders are empty!
            if len(file_names) == 0:
                print("There are no faces in this folder!")
                continue

            file_paths = [movie_path + '/' + name for name in file_names]

            try:
                images = bulk_get_image_pixels(file_paths[:1000], BUCKET_NAME)

            except Exception as e:
                print("An error occurred loading the images. Skipping movie. The error:")
                print(e)

            try:
                av = average_image(images,
                                output_dim = 900,
                                face_predictor_path = 'diversity_in_cinema/face_averaging/shape_predictor_68_face_landmarks.dat')

            except Exception as e:
                print('An error occurred averaging the faces. Skipping movie. The error:')
                print(e)
                continue

            try:
                image_filename = 'faces/' + movie_name + '_avg_face_test.jpg'
                upload_image_to_gcp(av, BUCKET_NAME_STREAMLIT, image_filename)

            except Exception as e:
                print("failed to upload to GCP")

                print('An error occured while saving the image. The error:')
                print(e)

if __name__ == "__main__":
    main()
