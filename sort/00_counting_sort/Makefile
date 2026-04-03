NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 00_counting_sort

00_counting_sort: 00_counting_sort.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 00_counting_sort
