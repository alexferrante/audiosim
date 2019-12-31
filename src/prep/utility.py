import sys
import os

import pandas as pd
import numpy as np

import urllib.request

sys.path.append(os.getcwd())
from constants import AUDIO_PATH

def get_mp3(url, id):
    urllib.request.urlretrieve(url + '.mp3', AUDIO_PATH + id + '.mp3')

#def create_entry(id, popularity, ):

