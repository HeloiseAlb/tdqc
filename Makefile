###########################
###### TDQC Makefile ######
###########################

ifeq "" "$(TDQC_PYTHONBIN)"
	TDQC_PYTHONBIN=python3
endif

# Clean package
#############################################################################
.PHONY: clean 

clean: 
	@rm -r build 1>/dev/null 2>/dev/null || true 

# Install package
#############################################################################
.PHONY: build 

build:
	@echo "------------------------------------------------ building package ------------------------------------------------"
	@echo ""
	${TDQC_PYTHONBIN} -m pip install . -v


# create documentation
#############################################################################
.PHONY: doc 

doc:
	pdoc tdqcs &

# perform unit tests
#############################################################################
.PHONY: pytest

pytest:
	${TDQC_PYTHONBIN} -m pytest -rfs -v --strict-markers --showlocals --full-trace || true 

all: clean build

install : all

test: pytest
