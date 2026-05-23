.PHONY: build test run_test clean distclean
BUILD_DIR := build/debug
PRESET ?= debug

all: build

build:
	cmake --preset $(PRESET)
	cmake --build --preset $(PRESET)

test: build run_test

run_test:
	ctest --preset $(PRESET)

clean:
	rm -rf $(BUILD_DIR)

distclean: clean
	rm -rf .cache

