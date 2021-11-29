import numpy as np
import pandas as pd
import time
from fake_useragent import UserAgent

from selenium import webdriver
from PIL import Image
from selenium.webdriver.chrome.options import Options
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
    _, response = http.request(url)

    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        soup = soup.find_all("a", href = True)

        pages = []
        for x in soup:
            if f"{url}page/" in str(x):
                pages.append(x)
        last_page = pages[-2].get_attribute_list("href")[0]

        http = httplib2.Http()
        _, response = http.request(last_page)

        for link in BeautifulSoup(response, parse_only=SoupStrainer('a'), features="html.parser"):
            if link.has_attr('href'):
                if '.jpg' in link['href']:
                    frame_url = link['href']

        last_frame_id = int(frame_url.split(".com-")[1].replace(".jpg",""))

        return frame_url, last_frame_id

    except:
        return frame_url, 20_000



def frame_urls(title, frame_interval=2):
    "Function which retunrs a list of all frame urls of a given movie"

    first_frame_url, last_frame_id = grab_frame_url(title)
    split_url = first_frame_url.split("com-")[0]
    frame_urls = [f"{split_url}com-{i}.jpg" for i in range(1, last_frame_id, frame_interval)]
    return frame_urls


def download_one_frame_bs(url):
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
            time.sleep(np.random.randint(5, 15))
            headers = {'Accept': 'text/html'}
            response = requests.get(url.strip(), stream=True, headers=headers)
            break
        except:
            if i == max_retry:
                return frame_number, "TimeoutError"
            else:
                print(f"Cannot retrieve frame number {frame_number}. Trying again in 5 seconds")
                time.sleep(np.random.randint(5, 15))
                i += 1

    if response.status_code != 200:
        print(f"{url} - Not found")
        return None, response.status_code

    else:
        image = Image.open(response.raw).quantize(colors=200, method=2)
        image = np.array(image.convert('RGB'))

        return frame_number, image


def download_one_frame_selenium(url):

    frame_number = url.split(".com-")[1].replace(".jpg", "")

    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(r'/Users/Moe/local/bin/chromedriver', options=options)
    driver.get(url)

    if len(driver.get_log('browser')) > 0:
        driver.close()
        print(f"{url} - Not found")
        return None, response.status_code

    else:
        img = driver.get_screenshot_as_png()
        driver.close()
        img = Image.open(BytesIO(img)).quantize(colors=250, method=2)
        img = np.array(img.convert('RGB'))

        return frame_number, img


if __name__ == "__main__":
    results = download_one_frame_selenium("Ace Ventura: Pet Detective (1994)", 1000)
    print(results)
