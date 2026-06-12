.PHONY: install run test

install:
	pip install -r backend/requirements.txt

run:
	cd backend && uvicorn main:app --reload --port 8000

test:
	python -m pytest backend/tests/ -v
