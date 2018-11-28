MAKE := @$(MAKE) --quiet --file $(lastword $(MAKEFILE_LIST))
TEST_LOC := utils

test:
	$(MAKE) clean
	TF_CPP_MIN_LOG_LEVEL=3 nosetests --config .noserc --nologcapture $(TEST_LOC)
	$(MAKE) clean

clean:
	find . -name '*.pyc' -type f -delete

.PHONY: test clean
