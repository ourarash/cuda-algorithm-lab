NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 02_vector_add

02_vector_add: 02_vector_add.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 02_vector_add
