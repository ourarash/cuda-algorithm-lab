NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 02_scan_brent_kung

02_scan_brent_kung: 02_scan_brent_kung.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 02_scan_brent_kung
