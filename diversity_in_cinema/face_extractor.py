from facenet_pytorch import MTCNN, extract_face
from PIL import Image


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

    boxes, probs, _ = mtcnn.detect(img, landmarks=True)

    if boxes is not None:
        for box, prob in zip(boxes, probs):

            # return only faces detcted with 99% or higher
            # certanty
            if prob >= 0.99:

                face = extract_face(img, box)
                face_list.append(face)


    return face_list
