from prep.crawl import get_data
from process.utility import extract_spect

def run():
    spects = []
    song_ids = get_data()
    for i in len(song_ids):
        spects.append(extract_spect(i))

