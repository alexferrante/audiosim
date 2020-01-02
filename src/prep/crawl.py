import math
import sys
import os
import pprint
from random import random
from dotenv import load_dotenv, find_dotenv

import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials

from utility import Utility
sys.path.append(os.getcwd())
from constants import DB_SIZE

load_dotenv(find_dotenv())

CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

client_credentials_manager = SpotifyClientCredentials(CLIENT_ID,CLIENT_SECRET)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_album_info(id):
    return spotify.album(id)

def get_artist_info(id):
    return spotify.artist(id)

def get_song_util():
    offset = math.floor(random() * 10000)
    results = spotify.search(q=get_random_query(), type='track', offset=offset, limit=1)
    while results['tracks']['items'][0]['preview_url'] is None:
        results = spotify.search(q=get_random_query(), type='track', offset=offset, limit=1)
    artist_id = results['tracks']['items'][0]['album']['artists'][0]['id']
    if len(get_artist_info(artist_id)['genres']) == 0:
        results = get_song_util()
    return results

def get_song():
    results = get_song_util()
    return {'song_id': results['tracks']['items'][0]['id'], 
            'artist_id': results['tracks']['items'][0]['album']['artists'][0]['id'], 
            'album_id': results['tracks']['items'][0]['album']['id'], 
            'preview_url': results['tracks']['items'][0]['preview_url'], 
            'popularity': results['tracks']['items'][0]['popularity'],
            'genres': get_artist_info(results['tracks']['items'][0]['album']['artists'][0]['id'])['genres']}

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

def get_data():
    util = Utility()
    for i in range(DB_SIZE):
        data = get_song()
        util.get_mp3(data['preview_url'], data['song_id'])
        util.create_entry(data['song_id'], data['artist_id'], data['album_id'], data['popularity'], data['genres'])
        

get_data()