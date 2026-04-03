NVCC = nvcc
CFLAGS = -O3
LDFLAGS = -lcublas

.PHONY: all clean

all: 00_cublas_gemm

00_cublas_gemm: 00_cublas_gemm.cu
	$(NVCC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f 00_cublas_gemm
