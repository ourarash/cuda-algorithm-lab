NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 01_hello_thread_hierarchy

01_hello_thread_hierarchy: 01_hello_thread_hierarchy.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 01_hello_thread_hierarchy
