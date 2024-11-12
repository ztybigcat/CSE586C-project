## Makefile for compiling the sorting program

# Compiler
NVCC = nvcc

# Compiler flags
CXXFLAGS = -std=c++14 -O2

# Compute capability
ARCH_FLAGS = -arch=sm_70

SRC = main.cpp data_generator.cpp cpu_sorter.cpp \
      gpu_reference_sorter.cu gpu_bitonic_sorter.cu \
	  gpu_radix_sorter.cu

# Header files
HEADERS = data_generator.h cpu_sorter.h \
          gpu_reference_sorter.h gpu_bitonic_sorter.h \
		  gpu_radix_sorter.h

# Executable name
EXEC = sort_program

# Default target
all: $(EXEC)

# Build the executable
$(EXEC): $(SRC) $(HEADERS)
	$(NVCC) $(CXXFLAGS) $(ARCH_FLAGS) $(SRC) -o $(EXEC)

# Clean up
clean:
	rm -f $(EXEC)

# Phony targets
.PHONY: all clean

# Phony targets
.PHONY: all clean
