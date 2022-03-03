#!/usr/bin/env bash

set -e
set -x

mypy .
flake8 --ignore=E501,W503,E402 .
black --check -l 80 .
