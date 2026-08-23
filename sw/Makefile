.PHONY: build test run_test clean distclean \
        common codegen simulator \
        test-common test-codegen test-simulator
BUILD_DIR := build/debug
PRESET ?= debug

all: build

configure:
	cmake --preset $(PRESET)

build: configure
	cmake --build --preset $(PRESET)

test: build run_test

run_test:
	ctest --preset $(PRESET)

common: configure
	cmake --build --preset $(PRESET) --target test_instruction

codegen: configure
	cmake --build --preset $(PRESET) --target macaque_codegen --target test_codegen

simulator: configure
	cmake --build --preset $(PRESET) --target macaque_simulator --target test_simulator

test-common: common
	ctest --preset $(PRESET) -R '^instruction_encoding$$'

test-codegen: codegen
	ctest --preset $(PRESET) -R '^codegen$$'

test-simulator: simulator
	ctest --preset $(PRESET) -R '^simulator$$'

clean:
	rm -rf $(BUILD_DIR)

distclean: clean
	rm -rf .cache
