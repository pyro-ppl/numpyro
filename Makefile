all: test

lint: FORCE
	flake8

format: FORCE
	isort -rc .

test: lint FORCE
	pytest -v test

clean: FORCE
	git clean -dfx -e numpyro.egg-info

docs: FORCE
	$(MAKE) -C docs html

FORCE:
