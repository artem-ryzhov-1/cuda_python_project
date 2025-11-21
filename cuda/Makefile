# Detect if the environment is local (using shell to check the platform)
LOCAL_RUNTIME := $(shell if [ -n "$$USER" ]; then echo "true"; else echo "false"; fi)

# Set NVCC based on the environment
ifeq ($(LOCAL_RUNTIME), true)
    # Local runtime (e.g., WSL, native Linux)

	# Use the CUDA_HOME or CUDA_PATH environment variable if it is set
	CUDA_PATH := $(shell echo $$CUDA_HOME)

	ifneq ($(CUDA_PATH),)
		# If CUDA_HOME is defined, use it to find nvcc
		NVCC = $(CUDA_PATH)/bin/nvcc
	else
		# Default path if CUDA_HOME is not set
		NVCC = /usr/local/cuda-12.1/bin/nvcc
	endif

else
    # Non-local runtime (e.g., Colab, other cloud platforms)
    NVCC ?= nvcc
endif

#ARCH ?= sm_75 #for Google Colab
#ARCH = sm_86 # for local machine
#ARCH ?= sm_75  # Default, override in build scripts if needed

# JSON include path
JSON_INCLUDE = external

# Compiler flags
#NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -Isrc -I$(JSON_INCLUDE)
NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -Isrc -I$(JSON_INCLUDE) -Xptxas -v

SRC = src/main.cu
TARGET = bin/lindblad_gpu

.PHONY: all clean

all:
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET) -lz

clean:
	rm -f $(TARGET)
