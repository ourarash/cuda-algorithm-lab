# Convenience wrapper around CMake, CTest, and the tools in tools/. The real
# build lives in CMakeLists.txt; everything is built into $(BUILD_DIR)/bin/<topic>/.
#
#   make                  configure (first time) and build every example
#   make test             run every example at full size; each must PASS
#                         (examples this GPU cannot run are reported as skipped)
#   make sanitize         run every example under compute-sanitizer
#                         (memcheck + racecheck) at --quick sizes
#   make bench            results tables for matmul, reduction, matrix_transpose
#   make roofline         roofline plot of the GEMM ladder (needs matplotlib;
#                         uses Nsight Compute if available)
#   make check-viz        smoke-test the HTML visualizations (needs Node.js)
#   make clean            delete the build directory
#
# Extra CMake options: make CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=80;90'

BUILD_DIR ?= build
CMAKE_ARGS ?=
PYTHON ?= python3

.PHONY: all test sanitize bench roofline check-viz clean

all: $(BUILD_DIR)/CMakeCache.txt
	cmake --build $(BUILD_DIR) --parallel

$(BUILD_DIR)/CMakeCache.txt:
	cmake -S . -B $(BUILD_DIR) $(CMAKE_ARGS)

test: all
	ctest --test-dir $(BUILD_DIR) -L '^run$$' --output-on-failure

sanitize: all
	ctest --test-dir $(BUILD_DIR) -L 'memcheck|racecheck' --output-on-failure

bench: all
	$(PYTHON) tools/bench.py --build $(BUILD_DIR)

roofline: all
	$(PYTHON) tools/roofline.py --build $(BUILD_DIR)

check-viz:
	node tools/check_visualizations.mjs

clean:
	rm -rf $(BUILD_DIR)
