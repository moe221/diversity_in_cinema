from google.cloud import storage
import numpy as np
import pandas as pd

from PIL import Image
from io import BytesIO

from diversity_in_cinema.params import *
from diversity_in_cinema.utils import ModelCheckpointInGcs

from tqdm import tqdm

from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder



# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'gender'


# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

# trained model storage location on GCP
STORAGE_LOCATION = 'models/gender_model.joblib'



def load_model_from_gcp():
    client = storage.Client().bucket(BUCKET_NAME)

    storage_location = 'models/vgg16.h5'

    blob = client.blob(storage_location)

    blob.download_to_filename('vgg16.h5')

    print("=> pipeline downloaded from storage")

    model = load_model('vgg16.h5')

    return model


def upload_file_to_gcp(file, file_name):

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    bucket.blob(f'models/{file_name}').upload_from_string(file.to_csv(),
                                                           'text/csv')

def load_vgg16_model():

    model = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224,3),
    pooling=None)

    return model

def freeze_layers(model, num_layers):

  # Freeze the layers
  for layer in model.layers[:num_layers]:
    layer.trainable = False
  # Check the trainable status of the individual layers
  for layer in model.layers:
    print(layer, layer.trainable)

  return model


def build_model_gender(model):

    # Create the model
    model_new = models.Sequential()
    # Add the VGG16 convolutional base model
    model_new.add(model)

    model_new.add(layers.Flatten())
    model_new.add(layers.Dense(500 , activation='relu'))
    model_new.add(layers.Dense(2, activation='sigmoid'))

    return model_new

def compile_model(model):

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    return model

def encode_labels(y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    keys = le.classes_
    values = le.transform(le.classes_)
    dictionary = dict(zip(keys, values))

    df = pd.DataFrame(dictionary.items(), columns=["label", "value"])
    upload_file_to_gcp(df, f"{MODEL_NAME}-Labels.csv")

    return y


def getImagePixels(file, bucket):

    blob = bucket.get_blob(f"{BUCKET_TRAIN_DATA_PATH}/{file}")
    data = blob.download_as_string()

    img = Image.open(BytesIO(data))
    img = np.array(img)

    return img


def balance_sample(df, nrows, balance_on):

    df_list = []
    targets = df[balance_on].unique()

    rows_per_class = int(nrows/len(targets))

    for target in targets:

        df_target = df[df[balance_on] == target]

        df_sample = df_target.sample(rows_per_class)

        df_list.append(df_sample)


    return pd.concat(df_list, axis=0)

def load_training_data(nrows=1000):

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    # load image labels
    train_df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}/fairface_label_train.csv")

    idx = train_df[(train_df['race'] == 'East Asian') | (train_df['race'] == 'Southeast Asian')].index
    train_df.loc[idx, 'race'] = 'Asian'

    total_sample_size = nrows

    # balance sampled data on target class and shuffel
    train_df  = balance_sample(train_df, total_sample_size, "gender")
    train_df = train_df.sample(frac=1).reset_index(drop=True)


    train_df['pixels'] = [getImagePixels(file, bucket) for file in tqdm(train_df["file"].values)]

    X_train = train_df["pixels"]

    y_train_gender = train_df[["gender"]]


    # encode target
    y_train_gender_encoded = encode_labels(y_train_gender.values)
    y_train_gender_cat = to_categorical(y_train_gender_encoded)

    # reshape input data
    X_train_reshaped = np.stack(X_train.values, axis=0)

    return X_train_reshaped, y_train_gender_cat


def create_model():
    # build_model for gender
    model = load_model_from_gcp()
    model = freeze_layers(model, num_layers=-2)
    model = build_model_gender(model)
    model = compile_model(model)

    return model

def train_model(model, X_test, y_test):

    # checkpoint
    filepath="gender_model_weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    gcs_dir = f"gs://{BUCKET_NAME}/models"

    checkpoint = ModelCheckpointInGcs(filepath,
                                      gcs_dir,
                                      monitor='val_accuracy',
                                      verbose=1,
                                      save_best_only=True,
                                      mode='max')

    es = EarlyStopping(patience=20, restore_best_weights=True)

    model.fit(
            x = X_test,
            y = y_test,
            validation_split=0.3,
            epochs=500,
            batch_size=16,
            callbacks=[es, checkpoint],
            verbose=1
        )

    return model


def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('gender_model.h5')



def save_model(model):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    model.save('gender_model.h5')
    print("saved gender_model.h5 locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded gender_model.h5' to gcp cloud storage under \n => {STORAGE_LOCATION}")



if __name__ == '__main__':
    # get training data from GCP bucket
    print("loading data")
    X_train, y_train= load_training_data(nrows=30_000)


    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)

    print("creating model")
    model = create_model()

    print("traning model")
    trained_model = train_model(model, X_train, y_train)

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(trained_model)
