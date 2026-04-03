SUBDIRS = 00_uncoalesced 01_coalesced 02_shared_memory 03_register_tiling 04_vectorized 05_tensor_cores

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
