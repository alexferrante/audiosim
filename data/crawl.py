import spotipy
import random
import math
import sys

spotify = spotipy.Spotify()

def get_song():
    offset = math.floor(random * 10000)
    results = spotify.search(q=get_random_query(), type='track', offset=offset)
    return results

def get_random_query():
    chars = 'abcdefghijklmnopqrstuvwxyz'
    rnd_char = chars[math.floor(random() * chars.length)]
    query = ''
    rnd = round(random())
    if rnd == 0:
        query = rnd_char + '%'
    else:
        query = '%' + rnd_char + '%'
    return query
