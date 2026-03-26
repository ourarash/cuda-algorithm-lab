NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 00_matmul_uncoalesced

00_matmul_uncoalesced: 00_matmul_uncoalesced.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 00_matmul_uncoalesced