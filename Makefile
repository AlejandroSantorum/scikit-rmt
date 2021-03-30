help:
	@echo "Execute make with one of the next arguments:"
	@echo "\tmake help - print makefile usage"
	@echo "\tmake activate - command to activate virtual environment"
	@echo "\tmake deactivate - command to deactivate virtual environment"
	@echo "\tmake requirements - list library requirements to be installed"
	@echo "\tmake install_requirements - install library requirements (assert venv is activated)"


activate:
	@echo "To activate virtual environment just execute 'source venv/bin/activate'"


deactivate:
	@echo "To deactivate virtual environment just execute 'deactivate'"


requirements:
	@echo "===> NON-STANDARD LIBRARIES TO BE INSTALLED <==="	
	@echo "\t numpy: library for efficient mathematical calculus for Python"
	@echo "\t matplotlib: Python plotting library"
	@echo "\t pytest: Python testing library"
	@echo "==> You can install all requirements by executing 'make install_requirements'"


.PHONY: install_requirements
install_requirements:
	pip3 install -r requirements.txt