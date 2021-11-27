from google.cloud import storage
import pandas as pd
import joblib
from trainer import *

from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# model folder name (will contain the folders for all trained model versions)
MODEL_NAME_GENDER = 'gender'
MODEL_NAME_RACE = 'race'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

# trained model storage location on GCP
STORAGE_LOCATION = 'models/gender_model.joblib'


def load_labels():

    # load image labels
    train_df = pd.read_csv("raw_data/fairface/fairface_label_train.csv")
    test_df = pd.read_csv("raw_data/fairface/fairface_label_val.csv")

    idx = train_df[(train_df['race'] == 'East Asian') | (train_df['race'] == 'Southeast Asian')].index
    train_df.loc[idx, 'race'] = 'Asian'

    idx = test_df[(test_df['race'] == 'East Asian') | (test_df['race'] == 'Southeast Asian')].index
    test_df.loc[idx, 'race'] = 'Asian'

    return test_df.sample(3000), train_df

def load_model():

    model = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(224,224,3),
    pooling=None,
    classifier_activation='sigmoid')

    return model

def freeze_layers(model, num_layers=-5):

  # Freeze the layers except the last 5
  for layer in model.layers[:num_layers]:
    layer.trainable = False
  # Check the trainable status of the individual layers
  for layer in model.layers:
    print(layer, layer.trainable)

  return model

def build_model_race(model):

    # Create the model
    model_new = models.Sequential()
    # Add the VGG16 convolutional base model
    model_new.add(model)

    model_new.add(layers.Flatten())
    model_new.add(layers.Dense(500 , activation='relu'))
    model_new.add(layers.Dense(6, activation='softmax'))

    return model_new


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

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def make_data_generators():

    test_df , train_df = load_labels()

    # load data using generator

    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    print("loading training set")

    train_generator = data_generator.flow_from_dataframe(
        train_df,
        "raw_data/fairface/",
        x_col='file',
        y_col='gender',
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=15)

    print("loading validation set")

    validation_generator = data_generator.flow_from_dataframe(
        test_df,
        "raw_data/fairface/",
        x_col='file',
        y_col='gender',
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=15)

    return train_generator, validation_generator


def create_model():
    # build_model for gender
    model = load_model()
    model = freeze_layers(model, num_layers=-2)
    model = build_model_gender(model)
    model = compile_model(model)

    return model

def train_model(model, X_test, X_val, y_test=None, y_val=None):

    es = EarlyStopping(patience=20, restore_best_weights=True)

    model.fit(
            X_test,
            epochs=2,
            batch_size=5,
            validation_data=X_val,
            callbacks=[es],
            verbose=1
        )

    return model


def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('model.joblib')



def save_model(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(reg, 'model.joblib')
    print("saved model.joblib locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")



if __name__ == '__main__':
    # get training data from GCP bucket
    print("loading data")
    train_generator, validation_generator = make_data_generators()


    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)

    print("creating model")
    model = create_model()

    print("traning model")
    trained_model = train_model(model, train_generator, validation_generator)

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(trained_model)
