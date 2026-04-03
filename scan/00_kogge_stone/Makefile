NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 00_scan_kogge_stone

00_scan_kogge_stone: 00_scan_kogge_stone.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 00_scan_kogge_stone
