SRC_DIR := spectral_binaries

.PHONY: clean
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

.PHONY: format
lint:
	flake8 $(SRC_DIR) --ignore=F841,W503

.PHONY: format
format:
	isort $(SRC_DIR)
	black --line-length 79 $(SRC_DIR)