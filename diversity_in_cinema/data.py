import numpy as np
from PIL import Image
import requests
from tqdm import tqdm
from diversity_in_cinema.scraper import grab_frame_url


def get_images(title, frame_count):
    lst = []
    frame_url = grab_frame_url(title)
    for i in tqdm(range(1, frame_count, 4)):
        split_url = frame_url.split("com-")
        new_frame = split_url[1].replace(f"{split_url[1]}", f"com-{i}.jpg")
        new_url = f"{split_url[0]}{new_frame}"
        response = requests.get(new_url, stream=True)
        if response.status_code != 200:
            break
        #preprocessing: resizing/compression, filtering of only frames with faces
        image = np.array(Image.open(response.raw))
        lst.append(image)
    # need to return dataframe
    return lst
