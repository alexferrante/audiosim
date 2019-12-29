import urllib.request

AUDIO_PATH = '../../data/raw/audio/'

def get_mp3(url, id):
    urllib.request.urlretrieve(url + '.mp3', AUDIO_PATH + id)

