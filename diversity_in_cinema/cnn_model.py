from deepface import DeepFace

def predict_face(face_list):
    #facial analysis
    return DeepFace.analyze(img_path = face_list,
                     detector_backend = 'opencv',
                     actions = ['gender', 'race'],
                     enforce_detection=False)
