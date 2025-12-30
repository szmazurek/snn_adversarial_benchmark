.PHONY: install format format-doc format-isort format-black lint lint-flake lint-mypy check-all

install:
	pip install --no-cache -r requirements-dev.txt

format-doc:
	docformatter --in-place  .

format-isort:
	isort .

format-black:
	black .

format: format-doc format-isort format-black

lint-flake:
	@echo "--- Running Flake8 ---"
	flake8 .

lint-mypy:
	@echo "--- Running Mypy ---"
	mypy .

lint: lint-flake lint-mypy

test:
	pytest --cov=.

all: format lint test