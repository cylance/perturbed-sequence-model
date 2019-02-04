NAME := local_perturbations 
VERSION := $(shell cat VERSION)
SOURCES := $(shell find src -name '*.py')
TARBALL := dist/$(NAME)-$(VERSION).tar.gz

PWD := $(shell pwd)
SITE_PACKAGES=$(PWD)/env/lib/python2.7/site-packages

# Check your code for style, run your unit tests, and make a source tarball
all: test clean-build build

# Run your unit tests after ensuring dependencies are up-to-date
test: env
	env/bin/pytest -sv tests


# Start a watcher that re-runs your unit tests when the tests or the code change
# If this doesn't work, make sure `nose-watch` is in your dev-requirements.txt
watch: env
	source env/bin/activate && env/bin/nosetests --with-watch tests/unit

# Make a compressed tarball of the source files and metadata only (no tests).
# This target will only run if code has changed since the last build.
# Copying the tarball to a remote server is an easy way to move your code around
$(TARBALL): $(SOURCES) env
	env/bin/python setup.py sdist
	@ls dist/* | tail -n 1 | xargs echo "Created source tarball"

# This is an easier-to-type name for the tarball, so you can run `make build`
build: $(TARBALL)

# install python module to virtualenv
install: env
	env/bin/python setup.py install

# Install all dependencies from requirements files
env: requirements.txt dev-requirements.txt | env/tools-installed.flag
	env/bin/pip install pip==9.0.1
	env/bin/pip install -r dev-requirements.txt
	env/bin/pip install -r requirements.txt
# create the .egg-info directory and link the project into the virtualenv making
# any changes to the source available after a load or reload
	
	#other things for development 
	ln -sfn $(PWD)/tests $(SITE_PACKAGES)/tests 
	ln -sfn $(PWD)/experiments $(SITE_PACKAGES)/experiments

	#editable installs 
	env/bin/pip install -e .
# installing packages won't update the modified time on the env folder, so
# unless we do that ourselves then make will constantly rebuild this target
# after either of the requirements files have been edited.
	@touch env


# Force reinstallation of dependencies
force-env: clean-env env
# (This is an example of declaring an alias to run two other targets in order)

# clean random files left from testing, etc, change these as needed
clean: clean-tarball
	rm -rf *csv
	rm -rf *h5
	rm -rf src/$(NAME)/*.h5
	rm -rf *pkl
	rm -rf *png
	rm -rf annoy_index_file
	rm -rf X.np.txt
	rm -rf projecion.np.txt
	rm -rf coverage.xml
	rm -rf nosetests.xml
	rm -rf pylint.out
	rm -rf build/
	rm -rf src/$(NAME).egg-info/

clean-env:
	rm -rf env/

clean-build:
	rm -rf dist/

clean-tarball:
	rm -f $(TARBALL)

# Some more aliases
dev: env

# This is a hack to separate virtualenv initialization from installing deps.
# Doing this allows deps to be updated more quickly.
env/tools-installed.flag:
	virtualenv --no-site-packages env
	@touch env/tools-installed.flag

# #why $$?  see https://www.gnu.org/software/make/manual/html_node/Reference.html#Reference
# eda:
# 	for f in eda/*.py; do env/bin/python "$$f"; done

experiment:
	env/bin/python $(SITE_PACKAGES)/experiments/main.py 

experiment1:
	env/bin/python $(SITE_PACKAGES)/experiments/run_experiment_1.py 

experiment2:
	env/bin/python $(SITE_PACKAGES)/experiments/run_experiment_2.py 


# Mark these targets as 'phony', which means Make considers them always out of
# date so they're always re-run.
.PHONY: test watch clean clean-build clean-env clean-tarball force-env experiment experiment1 experiment2 

# Phony targets are useful in two cases:
#   * when the only point of the target is to clean some filesystem state;
#   * when the target runs tests.
