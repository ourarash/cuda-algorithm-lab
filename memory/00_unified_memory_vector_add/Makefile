NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 00_unified_memory_vector_add

00_unified_memory_vector_add: 00_unified_memory_vector_add.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 00_unified_memory_vector_add
