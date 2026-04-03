NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 06_matmul_tensor_cores

# The WMMA API used in tensor cores requires a Turing architecture (sm_70) or newer.
06_matmul_tensor_cores: 06_matmul_tensor_cores.cu
	$(NVCC) $(CFLAGS) -arch=sm_70 -o $@ $<

clean:
	rm -f 06_matmul_tensor_cores