NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 01_merge_sort

01_merge_sort: 01_merge_sort.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 01_merge_sort
