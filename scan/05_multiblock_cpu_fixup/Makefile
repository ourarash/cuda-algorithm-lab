NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 05_scan_multiblock_cpu_fixup

05_scan_multiblock_cpu_fixup: 05_scan_multiblock_cpu_fixup.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 05_scan_multiblock_cpu_fixup
