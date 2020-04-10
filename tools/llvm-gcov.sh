#!/bin/bash
if [ "$1" = "-v" ]; then
  echo "llvm-cov-wrapper 4.2.1"
  exit 0
else
  exec llvm-cov gcov "$@"
fi
