NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 02_matmul_shared_memory

02_matmul_shared_memory: 02_matmul_shared_memory.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 02_matmul_shared_memory