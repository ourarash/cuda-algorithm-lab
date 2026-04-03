NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 05_matmul_vectorized

05_matmul_vectorized: 05_matmul_vectorized.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 05_matmul_vectorized