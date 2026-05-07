.PHONY: setup data train evaluate errors report clean sample

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m src.setup_nltk

sample:
	$(PYTHON) scripts/generate_sample_data.py

data: sample
	$(PYTHON) -m src.load_data --input data/raw/sample_letterboxd_like.csv

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

errors:
	$(PYTHON) -m src.error_analysis

report:
	cd report && latexmk -pdf -interaction=nonstopmode main.tex

clean:
	rm -rf artifacts report/*.aux report/*.log report/*.out report/*.toc report/*.fdb_latexmk report/*.fls report/*.synctex.gz report/*.bbl report/*.blg
