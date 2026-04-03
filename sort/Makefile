SUBDIRS = \
	00_counting_sort \
	01_merge_sort \
	02_epsilon_sort_thrust \
	03_epsilon_sort_manual

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
