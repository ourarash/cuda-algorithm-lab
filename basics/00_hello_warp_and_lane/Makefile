NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 00_hello_warp_and_lane

00_hello_warp_and_lane: 00_hello_warp_and_lane.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 00_hello_warp_and_lane
