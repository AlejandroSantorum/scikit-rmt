help:
	@echo "Execute make with one of the next arguments:"
	@echo "\tmake help - print makefile usage"
	@echo "\tmake requirements - list library requirements to be installed"
	@echo "\tmake install_requirements - install library requirements (assert venv is activated)"
	@echo "\tmake activate - command to activate virtual environment"
	@echo "\tmake deactivate - command to deactivate virtual environment"
	@echo "\tmake help_sphinx - print sphinx usage"
	@echo "\tmake apidoc - executes sphinx docs generation"
	@echo "\tmake pytest - executes pytest in the RMTpy library"
	@echo "\tmake coverage - executes pytest-cov in the RMTpy library"
	@echo "\tmake cov_html - executes pytest-cov in the RMTpy library and generates html report files"


activate:
	@echo "To activate virtual environment just execute 'source venv/bin/activate'"


deactivate:
	@echo "To deactivate virtual environment just execute 'deactivate'"


requirements:
	@echo "===> NON-STANDARD LIBRARIES TO BE INSTALLED <==="	
	@echo "\t numpy: Library for efficient mathematical calculus for Python"
	@echo "\t scipy: Python-based library for mathematics, science, and engineering"
	@echo "\t matplotlib: Python plotting library"
	@echo "\t pytest: Python testing library"
	@echo "\t pytest-cov: Python testing coverage library"
	@echo "\t sphinx: Python library for auto-generation of documentation"
	@echo "==> You can install all requirements by executing 'make install_requirements'"


.PHONY: install_requirements
install_requirements:
	pip3 install -r requirements.txt


### pytest & coverage ###
.PHONY: pytest
pytest:
	python3 -m pytest rmtpy

.PHONY: coverage
coverage:
	pytest --cov=rmtpy

.PHONY: cov_html
cov_html:
	pytest --cov-report html --cov=rmtpy

####################################


### Minimal makefile for Sphinx documentation ###
.PHONY: apidoc
apidoc:
	sphinx-apidoc -o docs rmtpy

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help_sphinx:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)