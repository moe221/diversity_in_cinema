from facenet_pytorch import MTCNN, extract_face
from PIL import Image
import numpy as np


def extract_face_mtcnn(image):

    """
    Function which given an image resturns a list of faces all detected
    faces
    """

    mtcnn = MTCNN(keep_all=True,
                      min_face_size=35,
                      post_process=False)
    face_list = []

    print("Extracting faces...")

    img = Image.fromarray(image)

    boxes, probs= mtcnn.detect(img)

    if boxes is not None:
        for box, prob in zip(boxes, probs):

            # return only faces detcted with 99% or higher
            # certanty
            if prob >= 0.99:

                face = extract_face(img, box)

                out = np.array(face.permute(1, 2, 0))
                out = out.astype('uint8')

                face_list.append(out)

    print(f"{len(face_list)} faces detected.")

    return face_list
