init:
	pip install -r requirements.txt

get_data:
	$(MAKE) init
	python src/prep/crawl.py

train: get_data

run: train

clean:
	rm -f data/raw/audio/*.mp3


