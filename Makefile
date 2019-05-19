all: test

lint: FORCE
	flake8

format: FORCE
	isort -rc .

doctest: FORCE
	$(MAKE) -C docs doctest

test: lint FORCE
	pytest -v test

clean: FORCE
	git clean -dfx -e numpyro.egg-info

docs: FORCE
	$(MAKE) -C docs html

FORCE:
