# Makefile for compiling the sorting program

# Compiler
NVCC = nvcc

# Compiler flags
CXXFLAGS = -std=c++11 -O2

# Compute capability
ARCH_FLAGS = -arch=sm_89

SRC = main.cpp data_generator.cpp cpu_sorter.cpp performance_timer.cpp \
      gpu_reference_sorter.cu gpu_custom_sorter.cu

# Header files
HEADERS = data_generator.h cpu_sorter.h performance_timer.h \
          gpu_reference_sorter.h gpu_custom_sorter.h

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
