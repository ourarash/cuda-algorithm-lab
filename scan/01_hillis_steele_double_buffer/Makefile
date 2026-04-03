NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 01_scan_hillis_steele_double_buffer

01_scan_hillis_steele_double_buffer: 01_scan_hillis_steele_double_buffer.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 01_scan_hillis_steele_double_buffer
