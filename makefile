TEST_LOC := 'utils/'

test:
	nosetests --config .noserc $(TEST_LOC)

clean:
	find . -name '*.pyc' -type f -delete

.PHONY: test clean
