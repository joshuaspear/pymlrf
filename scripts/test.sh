#!/bin/bash -eux

# create temporary directory for tests
mkdir -p tests/test_data

pytest tests -p no:warnings -v

# clean up
rm -r tests/test_data