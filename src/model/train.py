from prep.crawl import get_data
from process.utility import extract_spect

def run():
    spects = []
    print('fetching training data from spotify...')
    song_ids = get_data()
    print('computing spectrograms...')
    for i in len(song_ids):
        spects.append(extract_spect(i))

