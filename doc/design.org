# Fast simple LMM resolver

This LMM resolver is a simplified version of FaST-LMM. It is part of genenetwork.org.

## Multi-core support

## GPU CUDA support

CUDA support is automatic, provided a GPU is installed and
dependencies are working.  pylmm tests for those dependencies and
checks for RAM on the GPU. If these fail the program falls back to the
multi-core edition.

