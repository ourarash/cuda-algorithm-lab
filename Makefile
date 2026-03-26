SUBDIRS = matmul matmul_siboehm scan sparse
CUDA_FILES = $(wildcard *.cu)
TARGETS = $(CUDA_FILES:.cu=)

NVCC = nvcc
CFLAGS = -O3

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS) $(TARGETS)

# Compile loose .cu files in the root directory
%: %.cu
	$(NVCC) $(CFLAGS) -o $@ $<

$(SUBDIRS):
	@if [ -f $@/Makefile ]; then \
		$(MAKE) -C $@; \
	fi

clean:
	rm -f $(TARGETS)
	@for dir in $(SUBDIRS); do \
		if [ -f $$dir/Makefile ]; then \
			$(MAKE) -C $$dir clean; \
		fi; \
	done