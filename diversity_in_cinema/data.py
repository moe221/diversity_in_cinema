import numpy as np
import pandas as pd
from PIL import Image
import requests
from diversity_in_cinema.scraper import grab_frame_url


def get_images(title, frame_interval=2):
    '''Returns a dataframe for a given movie of size of the number of desired frames
    including an image array for each frame'''
    img_lst = []
    frame_lst = []
    title_lst = []
    frame_url = grab_frame_url(title)
    i = 1


    while True:
        split_url = frame_url.split("com-")
        new_frame = split_url[1].replace(f"{split_url[1]}", f"com-{i}.jpg")
        new_url = f"{split_url[0]}{new_frame}"
        response = requests.get(new_url, stream=True)

        if response.status_code != 200:
            break
        image = np.array(Image.open(response.raw))
        img_lst.append(image)
        frame_lst.append(i)
        title_lst.append(title)
        print(i)
        i = i + frame_interval


    img_dict = {
        'title': title_lst,
        'frame_no': frame_lst,
        'img_array': img_lst
    }

    return pd.DataFrame.from_dict(img_dict)
