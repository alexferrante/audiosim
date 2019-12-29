import urllib.request

PATH = 'audio/'

def get_mp3(url, id):
    urllib.request.urlretrieve(url + '.mp3', PATH + id)
