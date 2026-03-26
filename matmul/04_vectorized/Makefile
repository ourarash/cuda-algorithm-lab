NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 04_matmul_vectorized

04_matmul_vectorized: 04_matmul_vectorized.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 04_matmul_vectorized