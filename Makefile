all: test

lint: FORCE
	flake8

test: lint FORCE
	pytest -v test

clean: FORCE
	git clean -dfx -e numpyro.egg-info

FORCE:
