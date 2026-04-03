NVCC = nvcc
CFLAGS = -O3
LDFLAGS = -lcusparse

.PHONY: all clean

all: 01_spgemm_cusparse

01_spgemm_cusparse: 01_spgemm_cusparse.cu
	$(NVCC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f 01_spgemm_cusparse
