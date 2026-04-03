NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 03_epsilon_sort_manual

03_epsilon_sort_manual: 03_epsilon_sort_manual.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 03_epsilon_sort_manual
