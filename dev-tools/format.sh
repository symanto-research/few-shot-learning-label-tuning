#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place --exclude=__init__.py .
isort .
black -l 80 .

