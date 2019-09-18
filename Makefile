doc:
	sphinx-apidoc -f -o ./docs/ ./src/
	sphinx-build -b html ./docs ./docs/_build

test:
	pytest -v --capture=no

lint:
	flake8 ./src/

install_package:
	python -m pip install -e .
