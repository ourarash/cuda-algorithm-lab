NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 00_xor_set_symmetric_difference

00_xor_set_symmetric_difference: 00_xor_set_symmetric_difference.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 00_xor_set_symmetric_difference
