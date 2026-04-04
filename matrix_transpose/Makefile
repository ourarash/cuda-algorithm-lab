NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 00_transpose_bank_conflicts

00_transpose_bank_conflicts: 00_transpose_bank_conflicts.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 00_transpose_bank_conflicts
