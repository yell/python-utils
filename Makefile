MAKE := @$(MAKE) --quiet --file $(lastword $(MAKEFILE_LIST))
TEST := utils

test:
	$(MAKE) clean
	TF_CPP_MIN_LOG_LEVEL=3 nosetests --config .noserc --nologcapture $(TEST)
	$(MAKE) clean

clean:
	find . -name '*.pyc' -type f -delete
	rm -rf test_model*

.PHONY: *
