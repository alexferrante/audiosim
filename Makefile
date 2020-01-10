init:
	pip install -r requirements.txt

get_data:
	$(MAKE) init
	python src/prep/crawl.py

process_data: 
	python src/process/utility.py

train: get_data

run: train

clean:
	rm -f data/raw/audio/*.mp3
	rm -f data/processed/audio/*.wav


