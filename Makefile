# Convenience wrapper around CMake and CTest. The real build lives in
# CMakeLists.txt; everything is built into $(BUILD_DIR)/bin/<topic>/.
#
#   make                  configure (first time) and build every example
#   make test             run every example at full size; each must PASS
#   make sanitize         run every example under compute-sanitizer
#                         (memcheck + racecheck) at --quick sizes
#   make clean            delete the build directory
#
# Extra CMake options: make CMAKE_ARGS='-DCMAKE_CUDA_ARCHITECTURES=80;90'

BUILD_DIR ?= build
CMAKE_ARGS ?=

.PHONY: all test sanitize clean

all: $(BUILD_DIR)/CMakeCache.txt
	cmake --build $(BUILD_DIR) --parallel

$(BUILD_DIR)/CMakeCache.txt:
	cmake -S . -B $(BUILD_DIR) $(CMAKE_ARGS)

test: all
	ctest --test-dir $(BUILD_DIR) -L '^run$$' --output-on-failure

sanitize: all
	ctest --test-dir $(BUILD_DIR) -L 'memcheck|racecheck' --output-on-failure

clean:
	rm -rf $(BUILD_DIR)
