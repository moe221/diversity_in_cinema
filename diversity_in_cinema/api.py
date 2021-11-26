import requests
import pandas as pd


api_key = '87337046eaf9c07ce51c68d19a21041a'


def fetch_movie_id(movie):
    """
    Get movie title ID from The Movie DB API. Returns error string if not found
    """
    split_movie = movie.lower().split()
    year = split_movie[-1].replace('(','').replace(')','')
    search_terms = split_movie[:-1]
    title = '+'.join(search_terms)
    url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}'

    response = requests.get(url)

    if response.status_code != 200:
        return 'Error: status code not 200'

    data = response.json()

    for index in data['results']:
        if year in data['results'][index]['release_date']:

            return data['results'][index]['id']


def fetch_movie_details(movie):
    """
    Get desired movie details from The Movie DB API. Returns error string if not found
    """

    movie_id = fetch_movie_id(movie)

    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US'

    response = requests.get(url)

    if response.status_code != 200:

        return 'Error: status code not 200'

    data = response.json()

    # pd.DataFrame.from_dict(data)
    pass


def fetch_movie_credits(movie):
    """
    Get desired movie credits from The Movie DB API. Returns error string if not found
    """

    movie_id = fetch_movie_id(movie)

    url = f'https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={api_key}&language=en-US&append_to_response={movie_id}'

    response = requests.get(url)

    if response.status_code != 200:

        return 'Error: status code not 200'

    data = response.json()

    # pd.DataFrame.from_dict(data)
    pass
