from facenet_pytorch import MTCNN, extract_face
from PIL import Image

from tensorflow import keras
from tensorflow.python.lib.io import file_io

import os
import io

from google.cloud import storage

import numpy as np
import pandas as pd


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
    file_names = [str(x).split(",")[1].replace(
        f"{subfolders}/", "").strip() for x in \
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
