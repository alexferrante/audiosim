import sys
import os
import urllib.request

sys.path.append(os.getcwd())
from constants import AUDIO_PATH

def get_mp3(url, id):
    urllib.request.urlretrieve(url + '.mp3', AUDIO_PATH + id + '.mp3')

