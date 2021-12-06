import re
import numpy as np
import pandas as pd
import time
from fake_useragent import UserAgent

from selenium import webdriver
from PIL import Image
from selenium.webdriver.chrome.options import Options

from concurrent.futures import ThreadPoolExecutor
from PIL import Image

from io import BytesIO
from diversity_in_cinema.utils import extract_face_mtcnn
from diversity_in_cinema.utils import upload_image_to_gcp
from diversity_in_cinema.params import *
from tqdm import tqdm
import requests

import requests
import httplib2
from bs4 import BeautifulSoup, SoupStrainer


BASE_URL = "https://movie-screencaps.com/movie-directory/"


def get_movies():

    """
    Creates a dictionary of all movies titles as keys with their URLs as values

    """

    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.content, "html.parser")

    titles = []
    urls = []

    for column_section in soup.find_all("ul", class_="links"):

        for movie in column_section.find_all("li"):

            title = movie.find("a").string.strip()
            url_1 = movie.find(href=True)

            # append to lists
            titles.append(title)
            urls.append(url_1["href"].strip())

    movies = dict(zip(titles, urls))

    return movies


def grab_frame_url(title):

    """

    Grabs the URL of the first frame JPG of a given movie for use in retrieving
    all images

    """

    movies = get_movies()
    url = movies[title]
    print(url)
    http = httplib2.Http()

    _, response = http.request(url)


    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    soup = soup.find_all("a", href = True)

    # match url for a frame page
    regex = re.compile(r'(href="(.+?)" title="Go to)')

    try:

        last_page = regex.findall(str(soup))[-1][1]
        http = httplib2.Http()
        _, response = http.request(last_page)

        for link in BeautifulSoup(response,
                                  parse_only=SoupStrainer('a'),
                                  features="html.parser"):

            if link.has_attr('href'):
                if link['href'].endswith(".jpg"):
                    frame_url = link['href']

        last_frame_id = int(frame_url.split(".com-")[1].split(".jpg")[0])

        return frame_url, last_frame_id

    except IndexError:
        return None, 20_000



def frame_urls(title, frame_interval):

    """
    Function which returns a list of all frame urls for a given movie and
    frame interval

    """

    first_frame_url, last_frame_id = grab_frame_url(title)
    split_url = first_frame_url.split("com-")[0]
    frame_urls = [f"{split_url}com-{i}.jpg" for i in range(1,
                                                           last_frame_id,
                                                           frame_interval)]

    return frame_urls


def download_one_frame_bs(url):

    """
    Function which returns an image as a numpy array given an image url

    """

    ua = UserAgent()

    headers = {
            "User-Agent": str(ua.chrome),
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,\
                image/*,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Referer": "//movie-screencaps.com/",
            # ... other headers ...
    }

    frame_number = url.split(".com-")[1].replace(".jpg", "")

    max_retry = 50
    i = 0

    while True:

        try:
            time.sleep(np.random.randint(5, 15))
            headers = {'Accept': 'text/html'}
            response = requests.get(url.strip(), stream=True, headers=headers)
            break

        except:

            if i == max_retry:
                return frame_number, "TimeoutError"

            else:
                print(f"Cannot retrieve frame number {frame_number}.\
                        Trying again in 5 seconds")

                time.sleep(np.random.randint(5, 15))

                i += 1

    if response.status_code != 200:
        print(f"{url} - Not found")
        return None, url

    else:
        image = Image.open(response.raw).quantize(colors=200, method=2)
        image = np.array(image.convert('RGB'))

        return frame_number, image


def download_one_frame_selenium(url):

    """
    Function that uses Selenium to open a webrowser for a given frame link and
    takes a screen shot and saves it as an numpy array
    """

    frame_number = url.split(".com-")[1].replace(".jpg", "")

    options = Options()
    options.add_argument("--headless")

    driver = webdriver.Chrome(r'/Users/Moe/local/bin/chromedriver',
                              options=options)

    driver.get(url)

    if len(driver.get_log('browser')) > 0:

        driver.close()
        print(f"{url} - Not found")
        return None, 400

    else:
        img = driver.get_screenshot_as_png()
        driver.close()

        # save screenshot
        img = Image.open(BytesIO(img)).quantize(colors=250, method=2)
        img = np.array(img.convert('RGB'))

        return frame_number, img


def face_scraper(title, frame_interval, workers):
    """
    Function that given a movie title , identifies and extracts
    faces that are found in every frame based on the frame interval

    Returns a tuple containing a dataframe and the total number of
    scraped frames

    """

    urls = frame_urls(title, frame_interval)

    with ThreadPoolExecutor(max_workers=workers) as executor:

        face_dict = {}
        frame_count = 0
        for frame_id, frame in executor.map(download_one_frame_bs,
                                            urls,
                                            timeout=None):

            frame_count += 1
            if frame_id is None:
                continue

            else:
                print(f"Grabbing frame number {frame_id}\n")

                faces_list = extract_face_mtcnn(frame)
                face_dict[frame_id] = faces_list

                if len(faces_list) > 0:

                    print(f"{len(faces_list)} faces detected.")
                    print("uploading face images to GCP\n")

                    for i, face in tqdm(enumerate(faces_list)):

                        image = Image.fromarray(face)
                        image_name = f"face_images/\
                            {title.strip()}/{frame_id}_face{i}.jpg"

                        upload_image_to_gcp(image, BUCKET_NAME, image_name)
                else:
                    print(f"No faces detected in frame number {frame_id}\n")

            print(f"{frame_count} / {len(urls)} frames scraped!\n")

        faces_df = pd.DataFrame(data={"frame": list(face_dict.keys()),
                                "faces": list(face_dict.values())})

        return faces_df, len(urls)


if __name__ == "__main__":
    # testing
    # results = download_one_frame_selenium("Ace Ventura: Pet Detective (1994)",
    #                                       1000)
    # print(results)

    movies = get_movies()
    for movie in movies.keys():
        grab_frame_url(movie)
