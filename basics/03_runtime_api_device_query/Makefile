NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean

all: 03_runtime_api_device_query

03_runtime_api_device_query: 03_runtime_api_device_query.cu
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f 03_runtime_api_device_query
