NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 03_scan_blelloch

03_scan_blelloch: 03_scan_blelloch.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 03_scan_blelloch
