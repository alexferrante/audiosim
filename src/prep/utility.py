import sys
import os

import pandas as pd
import numpy as np

import urllib.request

sys.path.append(os.getcwd())
from constants import AUDIO_PATH

class Utility:
    def __init__(self):
        self.tp = []
        self.df = pd.DataFrame(columns=['song_id', 'artist_id', 'album_id', 'popularity', 'genres'])

    def get_mp3(self, url, id):
        urllib.request.urlretrieve(url + '.mp3', AUDIO_PATH + id + '.mp3')

    def create_entry(self, song_id, artist_id, album_id, popularity, genres):
        entry = tuple([song_id, artist_id, album_id, popularity, genres])
        self.tp.append(entry)
