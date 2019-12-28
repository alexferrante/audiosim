import math
import sys
import os
import pprint
from random import random
from dotenv import load_dotenv, find_dotenv

import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv(find_dotenv())

CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

client_credentials_manager = SpotifyClientCredentials(CLIENT_ID,CLIENT_SECRET)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song():
    offset = math.floor(random() * 10000)
    results = spotify.search(q=get_random_query(), type='track', offset=offset, limit=1)
    while results['tracks']['items'][0]['preview_url'] is None:
        results = spotify.search(q=get_random_query(), type='track', offset=offset, limit=1)
    return results

def get_random_query():
    chars = 'abcdefghijklmnopqrstuvwxyz'
    rnd_char = chars[math.floor(random() * len(chars))]
    query = ''
    rnd = round(random())
    if rnd == 0:
        query = rnd_char + '%'
    else:
        query = '%' + rnd_char + '%'
    return query

get_song()