
.PHONY: setup test zip clean

setup:
	python -m pip install -U pip
	pip install -r requirements.txt

test:
	pytest -q || true

zip:
	python -c "import shutil; shutil.make_archive('kaggle-turbokit-full', 'zip', '.')" && 	@echo "Wrote kaggle-turbokit-full.zip"

clean:
	rm -rf outputs __pycache__ .pytest_cache *.egg-info dist build
