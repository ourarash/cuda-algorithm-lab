NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 00_spmv_coo

00_spmv_coo: 00_spmv_coo.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 00_spmv_coo
