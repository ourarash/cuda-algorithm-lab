NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 02_epsilon_sort_thrust

02_epsilon_sort_thrust: 02_epsilon_sort_thrust.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 02_epsilon_sort_thrust
