import requests
import pandas as pd


api_key = '87337046eaf9c07ce51c68d19a21041a'


def fetch_movie_basic_data(movie):
    """
    Get movie title ID from The Movie DB API. Returns error string if not found
    """
    remove_4k = movie.lower().replace('[4k]', '').strip()
    split_movie = remove_4k.split()
    year = split_movie[-1].replace('(', '').replace(')', '')
    search_terms = split_movie[:-1]
    title = '+'.join(search_terms)
    url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}'

    response = requests.get(url)

    if response.status_code != 200:
        return 'Error: status code not 200'

    data = response.json()

    keep_data = {}

    for index in range(len(data['results'])):
        if year in data['results'][index]['release_date']:
            keep_data['release_date'] = data['results'][index]['release_date']
            keep_data['original_language'] = data['results'][index][
                'original_language']
            keep_data['poster_path'] = data['results'][index]['poster_path']
            return data['results'][index]['id'], keep_data


def fetch_movie_details(movie):
    """
    Get desired movie details from The Movie DB API. Returns error string if not found
    """

    movie_id = fetch_movie_basic_data(movie)

    url = f'https://api.themoviedb.org/3/movie/{movie_id[0]}?api_key={api_key}&language=en-US'

    response = requests.get(url)

    if response.status_code != 200:

        return 'Error: status code not 200'

    data = response.json()

    keep_data = {}

    keep_data['genres'] = data['genres']
    keep_data['spoken_languages'] = data['spoken_languages']
    keep_data['runtime'] = data['runtime']
    keep_data['revenue'] = data['revenue']

    # pd.DataFrame.from_dict(data)
    return keep_data


def fetch_movie_credits(movie):
    """
    Get desired movie credits from The Movie DB API. Returns error string if not found
    """

    movie_id = fetch_movie_basic_data(movie)

    url = f'https://api.themoviedb.org/3/movie/{movie_id[0]}/credits?api_key={api_key}&language=en-US&append_to_response={movie_id}'

    response = requests.get(url)

    if response.status_code != 200:

        return 'Error: status code not 200'

    data = response.json()

    keep_data = {}

    keep_data['cast'] = data['cast']
    keep_data['crew'] = data['crew']

    # pd.DataFrame.from_dict(data)
    return keep_data
