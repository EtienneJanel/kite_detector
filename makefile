.PHONY: fmt verify

# Format code using black and isort
fmt:
	PYTHONPATH=$(PWD) black .
	PYTHONPATH=$(PWD) isort .

test: 
	PYTHONPATH=$(PWD) pytest

run:
	PYTHONPATH=$(pwd) uvicorn serving.main:app --host 0.0.0.0 --port 8000 --reload

build:
	docker build -t kite-detector .

compose:
	docker compose up -d