NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 03_matmul_register_tiling 04_matmul_2d_register_tiling

03_matmul_register_tiling: 03_matmul_register_tiling.cu
	$(NVCC) $(CFLAGS) -o $@ $<

04_matmul_2d_register_tiling: 04_matmul_2d_register_tiling.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 03_matmul_register_tiling 04_matmul_2d_register_tiling
