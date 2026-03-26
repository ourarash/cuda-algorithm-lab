NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 01_matmul_coalesced

01_matmul_coalesced: 01_matmul_coalesced.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 01_matmul_coalesced