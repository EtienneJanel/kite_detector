.PHONY: fmt verify

# Format code using black and isort
fmt:
	PYTHONPATH=$(PWD) black .
	PYTHONPATH=$(PWD) isort .

test: 
	PYTHONPATH=$(PWD) pytest

run:
	uvicorn serving.main:app --host 0.0.0.0 --port 8000 --reload

dbuild:
	docker build -t kite-detector .

dcompose:
	docker compose up -d

mlflow:
	mlflow ui --port 5005