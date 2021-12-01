from face_averaging import average_image
from matplotlib.image import imsave
import cv2

movie_folders = ['juno_test', 'presidents'] #TODO replace with links to google drive or wherever

for movie_name in movie_folders:

    print(f'Now Averaging: {movie_name}')
    
    try:

        av = average_image(movie_name, output_dim = 900)

        #plt.imshow(cv2.cvtColor(av, cv2.COLOR_BGR2RGB))

        image_filename = 'outputs/' + movie_name + '_avg_face.jpg'

        imsave(image_filename, cv2.cvtColor(av, cv2.COLOR_BGR2RGB))
    
    except Exception as e:

         print('An error occurred. Skipping movie. The error:')
         print(e)
