# Makefile for compiling the sorting program

# Compiler
NVCC = nvcc

# Compiler flags
CXXFLAGS = -std=c++11 -O2

# Compute capability
ARCH_FLAGS = -arch=sm_89

SRC = main.cpp data_generator.cpp cpu_sorter.cpp \
      gpu_reference_sorter.cu gpu_bitonic_sorter.cu \
	  gpu_radix_sorter.cu prefix_sum.cu

# Header files
HEADERS = data_generator.h cpu_sorter.h \
          gpu_reference_sorter.h gpu_bitonic_sorter.h \
		  gpu_radix_sorter.h prefix_sum.h

# Executable name
EXEC = sort_program

# Test executable
TEST_EXEC = prefix_sum_test

all: $(EXEC) $(TEST_EXEC)

# Build the main executable
$(EXEC): $(SRC) $(HEADERS)
	$(NVCC) $(CXXFLAGS) $(ARCH_FLAGS) $(SRC) -o $(EXEC)

# Build the prefix sum test
prefix_sum_test: prefix_sum_test.cu prefix_sum.cu prefix_sum.h
	$(NVCC) $(CXXFLAGS) $(ARCH_FLAGS) prefix_sum_test.cu prefix_sum.cu -o $(TEST_EXEC)

# Run the test
test: $(TEST_EXEC)
	./$(TEST_EXEC)

# Clean up
clean:
	rm -f $(EXEC) $(TEST_EXEC)

# Phony targets
.PHONY: all clean test
