MAKE := @$(MAKE) --quiet --file $(lastword $(MAKEFILE_LIST))
TEST_LOC := 'utils/'

test:
	$(MAKE) clean
	nosetests --config .noserc $(TEST_LOC)
	$(MAKE) clean

clean:
	find . -name '*.pyc' -type f -delete

.PHONY: test clean
