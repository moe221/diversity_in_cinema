import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fake_useragent import UserAgent

from selenium import webdriver
from PIL import Image
from io import BytesIO

import requests

import requests
import httplib2
from bs4 import BeautifulSoup, SoupStrainer


url = "https://movie-screencaps.com/movie-directory/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")


def get_movies():
    '''Creates a dictionary of all movies titles as keys with their URLs as values'''

    titles = []
    urls = []

    for column_section in soup.find_all("ul", class_="links"):
        for movie in column_section.find_all("li"):
            title = movie.find("a").string.strip()
            url_1 = movie.find(href=True)
            titles.append(title)
            urls.append(url_1["href"].strip())

    movies = dict(zip(titles, urls))

    return movies


def grab_frame_url(title):
    '''Grabs the URL of the first frame JPG of a given movie for use in retrieving all images'''

    movies = get_movies()
    url = movies[title]
    http = httplib2.Http()
    status, response = http.request(url)

    for link in BeautifulSoup(response, parse_only=SoupStrainer('a'), features="html.parser"):
        if link.has_attr('href'):
            if '.jpg' in link['href']:
                frame_url = link['href']

                return frame_url


def frame_urls(title, frame_interval=2):
    "Function which retunrs a list of all frame urls of a given movie"

    first_frame_url = grab_frame_url(title)
    split_url = first_frame_url.split("com-")[0]
    frame_urls = [f"{split_url}com-{i}.jpg" for i in range(1, 20_000, frame_interval)]
    return frame_urls


def download_one_frame(url):
    "Function which returns an image array of a frame url"

    ua = UserAgent()


    headers = {
            "User-Agent": str(ua.chrome),
            "Accept": "accept: image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
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
            time.sleep(5)
            headers = {'Accept': 'text/html'}
            response = requests.get(url.strip(), stream=True, headers=headers)
            break
        except:
            if i == max_retry:
                return frame_number, "TimeoutError"
            else:
                print(f"Cannot retrieve frame number {frame_number}. Trying again in 5 seconds")
                time.sleep(7)
                i += 1

    if response.status_code != 200:
        print(f"{url} - No found")
        return None, response.status_code

    else:
        image = Image.open(response.raw).quantize(colors=200, method=2)
        image = np.array(image.convert('RGB'))

        return frame_number, image


def download_all_frames(title, frame_interval=2):

    urls = frame_urls(title, frame_interval)
    frame_list = []
    frame_id_list = []

    with ThreadPoolExecutor(max_workers=50) as executor:


        for frame_id, frame in  executor.map(download_one_frame_selenium,
                                        urls,
                                        timeout=None):

            if frame_id is None:
                continue

            else:
                frame_id_list.append(frame_id)
                frame_list.append(frame)
                print(f"Added frame number {frame_id}")

        print("Exporting to pandas DataFrame...")

        df = pd.DataFrame(data={'Frame_Id':frame_id_list, 'Image':frame_list})
        df["Title"] = [title] * len(df)
        return df.dropna()


def download_one_frame_selenium(url):

    frame_number = url.split(".com-")[1].replace(".jpg", "")

    driver = webdriver.Chrome(r'/Users/Moe/local/bin/chromedriver')
    driver.get(url)

    if len(driver.get_log('browser')) > 0:
        driver.close()
        print(f"{url} - No found")
        return None, response.status_code

    else:
        img = driver.get_screenshot_as_png()
        driver.close()
        img = Image.open(BytesIO(img)).quantize(colors=200, method=2)
        img = np.array(img.convert('RGB'))

        return frame_number, img




if __name__ == "__main__":
    results = download_all_frames("Ace Ventura: Pet Detective (1994)", 1000)
    print(results)
