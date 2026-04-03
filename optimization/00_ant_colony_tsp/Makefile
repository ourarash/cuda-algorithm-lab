NVCC = nvcc
CFLAGS = -O3
LDFLAGS = -lcurand

.PHONY: all clean

all: 00_ant_colony_tsp

00_ant_colony_tsp: 00_ant_colony_tsp.cu
	$(NVCC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f 00_ant_colony_tsp
