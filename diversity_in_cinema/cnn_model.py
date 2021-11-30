from deepface import DeepFace
import numpy as np
from diversity_in_cinema.params import *
from google.cloud import storage
from tensorflow.keras.models import load_model


def predict_face(face_list):
    #facial analysis
    return DeepFace.analyze(img_path = face_list,
                     detector_backend = 'opencv',
                     actions = ['gender', 'race'],
                     enforce_detection=False)


def predict_face_custome(face_list):

    # pre process and transform to shape (n, 224, 224, 3)
    # X = np.stack(face_list, axis=0)

    # load models from gcp
    client = storage.Client().bucket(BUCKET_NAME)

    storage_location = 'models/gender_model.h5'
    blob = client.blob(storage_location)
    blob.download_to_filename('gender_model.h5')

    print("=> gender model downloaded from storage")

    model_gender = load_model('gender_model.h5')

    storage_location = 'models/race_model.h5'
    blob = client.blob(storage_location)
    blob.download_to_filename('race_model.h5')

    print("=> race model downloaded from storage")

    model_race= load_model('race_model.h5')

    # predict gender:
    # result_gender = np.argmax(model_gender.predict(X), axis=-1) returns an array of numbers

    # predict race
    # result_race = np.argmax(model_race.predict(X), axis=-1)

    # gender_dict = {1: "Man", 0: "Woman"}
    # race_dict = {0:'Asian', 1: 'Black', 2: 'Indian', 3: 'Latino_Hispanic', 4: 'Middle Eastern', 5: 'White'}

    # results_list = []

    # for i, j in zip(result_race, result_gender):
    #     results_dict = {"dominant_race": race_dict[i]  , "gender": gender_dict[j]}
    #     results_list.append(results_dict)

    # return a list of dictionaries: {"dominant_race": , "gender": }
    pass
