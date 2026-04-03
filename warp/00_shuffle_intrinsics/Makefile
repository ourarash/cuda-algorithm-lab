NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 00_shuffle_intrinsics

00_shuffle_intrinsics: 00_shuffle_intrinsics.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 00_shuffle_intrinsics
