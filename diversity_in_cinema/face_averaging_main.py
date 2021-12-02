from face_averaging.face_averaging import average_image
from params import BUCKET_NAME, BUCKET_NAME_STREAMLIT
from utils_cloud import gcp_file_names, gcp_subdir_names, upload_image_to_gcp, bulk_get_image_pixels
from matplotlib.pyplot import imsave, imshow

bucket_location_prefix = 'face_images'

# Get names of folders
movie_folders = gcp_subdir_names(BUCKET_NAME, bucket_location_prefix)

for movie_name in movie_folders:

    print(f'Now Averaging: {movie_name}')
    
    movie_path = bucket_location_prefix + '/' + movie_name
    
    # Get names of paths
    file_names = gcp_file_names(BUCKET_NAME, movie_path)

    # TODO REMOVE THIS LINE. I PUT IT IN TO MAKE IT RUN FAST TO CHECK
    if len(file_names) > 10: file_names = file_names[0:5]

    # Lots of folders are empty!
    if len(file_names) == 0:
        print("There's no faces in this folder!")
        continue

    file_paths = [movie_path + '/' + name for name in file_names]
    
    try:

        images = bulk_get_image_pixels(file_paths, BUCKET_NAME)
    
    except Exception as e:

        print("An error occurred loading the images. Skipping movie. The error:")
        print(e)

    try:

        av = average_image(images, 
                           output_dim = 900, 
                           face_predictor_path = 'face_averaging/shape_predictor_68_face_landmarks.dat')

    except Exception as e:

        print('An error occurred averaging the faces. Skipping movie. The error:')
        print(e)

        # Some errors I've found and not handled:
            # An Image isn't loaded correctly but fail is silent
            # error: (-215:Assertion failed) !ssize.empty() in function 'cv::resize'
            
            # No faces are detected in the frames analyzed
            # list index out of range

        continue
    
    try:

        image_filename = 'faces/' + movie_name + '/' + movie_name + '_avg_face.jpg'

        #imsave(image_filename, av)

        upload_image_to_gcp(av, BUCKET_NAME_STREAMLIT, image_filename)

    except Exception as e:

        print('An error occured while saving the image. The error:')
        print(e)

        print('saving to outputs/ instead.')
        
        stripped_movie_name = ''.join(e for e in movie_name if e.isalnum())

        image_filename = 'face_averaging/outputs/' + stripped_movie_name +'_avg_face.jpg'

        imsave(image_filename, av)

    

'''
Below: One iteration, for testing
'''


# from face_averaging.face_averaging import average_image
# from params import BUCKET_NAME, BUCKET_NAME_STREAMLIT
# from utils_cloud import gcp_file_names, gcp_subdir_names, upload_image_to_gcp, getImagePixels
# import matplotlib.pyplot as plt

# bucket_location_prefix = 'face_images'
# movie_name = 'Hannah Montana: The Movie (2009)'
# movie_path = bucket_location_prefix + '/' + movie_name
# file_names = gcp_file_names(BUCKET_NAME, movie_path)
# file_paths = [movie_path + '/' + name for name in file_names]
# file_paths = file_paths[100:150]
# av = average_image(file_paths, 
#                     output_dim = 900, 
#                     face_predictor_path = 'face_averaging/shape_predictor_68_face_landmarks.dat',
#                     from_google = True,
#                     google_bucket = BUCKET_NAME)
# image_filename = 'faces/' + movie_name + '/' + movie_name + '_avg_face.jpg'
# upload_image_to_gcp(av, BUCKET_NAME_STREAMLIT, image_filename)