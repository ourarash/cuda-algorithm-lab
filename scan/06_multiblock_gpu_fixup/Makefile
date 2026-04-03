NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 06_scan_multiblock_gpu_fixup

06_scan_multiblock_gpu_fixup: 06_scan_multiblock_gpu_fixup.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 06_scan_multiblock_gpu_fixup
