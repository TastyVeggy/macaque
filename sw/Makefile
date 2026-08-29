.PHONY: build test run_test install clean distclean \
        common codegen simulator runtime \
        test-common test-codegen test-simulator test-runtime
BUILD_DIR := build/debug
PRESET ?= debug
BIN_PREFIX ?= $(abspath ..)

all: install

configure:
	cmake --preset $(PRESET)

build: configure
	cmake --build --preset $(PRESET)

test: build run_test

run_test:
	ctest --preset $(PRESET)

install: build
	cmake --install $(BUILD_DIR) --prefix $(BIN_PREFIX)

common: configure
	cmake --build --preset $(PRESET) --target test_instruction

codegen: configure
	cmake --build --preset $(PRESET) --target macaque_codegen --target test_codegen --target macaque-lower

simulator: configure
	cmake --build --preset $(PRESET) --target macaque_simulator --target test_simulator

runtime: configure
	cmake --build --preset $(PRESET) --target macaque_runtime --target macaque --target test_runtime \
	    --target infer_digit

test-common: common
	ctest --preset $(PRESET) -R '^instruction_encoding$$'

test-codegen: codegen
	ctest --preset $(PRESET) -R '^codegen$$'

test-simulator: simulator
	ctest --preset $(PRESET) -R '^simulator$$'

test-runtime: runtime
	ctest --preset $(PRESET) -R '^runtime$$'

clean:
	rm -rf $(BUILD_DIR)

distclean: clean
	rm -rf .cache
