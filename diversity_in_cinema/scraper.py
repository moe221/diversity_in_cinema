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
    '''Retrieves the URL of the first frame JPG of a given movie'''
    movies = get_movies()
    url = movies[title]
    http = httplib2.Http()
    status, response = http.request(url)
    for link in BeautifulSoup(response, parse_only=SoupStrainer('a')):
        if link.has_attr('href'):
            if '.jpg' in link['href']:
                frame_url = link['href']
                return frame_url
