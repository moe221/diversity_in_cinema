from diversity_in_cinema.scraper import *

def test_get_movies():

    movies = get_movies()
    assert type(movies) == dict


def test_grab_frame_url():

    frame_url = grab_frame_url('The A-Team (2010)')
    assert type(frame_url) == str
