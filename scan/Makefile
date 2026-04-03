SUBDIRS = \
	00_kogge_stone \
	01_hillis_steele_double_buffer \
	02_brent_kung \
	03_blelloch \
	04_blelloch_bank_conflict_free \
	05_multiblock_cpu_fixup \
	06_multiblock_gpu_fixup

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
