from facenet_pytorch import MTCNN, extract_face
from PIL import Image

from tensorflow import keras
from tensorflow.python.lib.io import file_io
import os

import numpy as np


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

            # return only faces detcted with 99% or higher
            # certanty
            if prob >= 0.97:

                face = extract_face(img, box)

                out = np.array(face.permute(1, 2, 0))
                out = out.astype('uint8')

                face_list.append(out)

    return face_list


class ModelCheckpointInGcs(keras.callbacks.ModelCheckpoint):
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
