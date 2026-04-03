NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 04_scan_blelloch_bank_conflict_free

04_scan_blelloch_bank_conflict_free: 04_scan_blelloch_bank_conflict_free.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 04_scan_blelloch_bank_conflict_free
