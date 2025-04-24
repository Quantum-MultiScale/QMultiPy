#=======================================================================
#                   define the compiler names
#=======================================================================
PYTHON       = /usr/bin/env python
PYLIB_SUFFIX = .cpython-311-darwin.so
FC           = gfortran


#=======================================================================
#                     Target module and source files
#=======================================================================
PYSRC_DIR          := qmultipy


#=======================================================================
#                     recipes
#=======================================================================
.PHONY: all hooks format lint test 

all: format lint test bdist

hooks:
	git config core.hooksPath .git/hooks/
	git config --unset-all core.hooksPath
	cp .githooks/* .git/hooks/
ifeq ($(shell which pre-commit >/dev/null 2>&1 && echo found),found)
	pre-commit install
else
	@test -n "$$TERM" && tput setaf 1 || exit 0
	@echo pre-commit not installed, SKIP configure pre-commit
	@test -n "$$TERM" && tput sgr0 || exit 0
endif


#=======================================================================
#                     Unittest
#=======================================================================
test:
	pytest
	# -rm -rf htmlcov
	# $(PYTHON) -m coverage run -m unittest -v -f
	# -$(PYTHON) -m coverage combine
	# $(PYTHON) -m coverage report
	# $(PYTHON) -m coverage xml
	# $(PYTHON) -m coverage html



#=======================================================================
#                     Lint & Format
#=======================================================================
lint:
	$(PYTHON) -m mypy  $(PYSRC_DIR) test 
	$(PYTHON) -m black $(PYSRC_DIR) test --check -S
	$(PYTHON) -m isort $(PYSRC_DIR) test --check --profile black

format:
	$(PYTHON) -m black -S              $(PYSRC_DIR) test
	$(PYTHON) -m isort --profile black $(PYSRC_DIR) test



#=======================================================================
#                     Manual & Docs
#=======================================================================
manual:
	make -C docs clean
	make -C docs html