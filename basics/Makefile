SUBDIRS = \
	00_hello_warp_and_lane \
	01_hello_thread_hierarchy \
	02_vector_add \
	03_runtime_api_device_query

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
