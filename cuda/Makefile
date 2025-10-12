NVCC ?= nvcc

#ARCH ?= sm_75 #for Google Colab
#ARCH = sm_86 # for local machine
#ARCH ?= sm_75  # Default, override in build scripts if needed

NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) -Isrc

SRC = src/main.cu
TARGET = lindblad_gpu

.PHONY: all clean

all:
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
