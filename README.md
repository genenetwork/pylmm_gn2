# pylmm - A lightweight linear mixed-model solver

This is an improved and faster (multi-core+CUDA) version of pylmm (a
form of the FaST-LMM algorithm by Lippert et al.) which is part of the
[Genenetwork2](https://github.com/genenetwork) project. pylmm_gn2 can
parse data in
[R/qtl2 format](http://kbroman.org/qtl2/assets/vignettes/input_files.html)
as input. E.g.,

    ./bin/runlmm.py --control data/rqtl/iron.yaml --pheno data/rqtl/iron_pheno.csv --geno data/rqtl/iron_geno.csv rqtl

(note if that if this command segfaults you need to upgrade
openblas, see ...)

pylmm is a fast and lightweight linear mixed-model (LMM) solver for
use in genome-wide association studies (GWAS). The original is derived
from FaST-LMM and that code base can be found
[here](https://github.com/nickFurlotte/pylmm). Prof. Karl Broman wrote
a comparison with his
[R/lmmlite](http://kbroman.org/lmmlite/assets/lmmlite.html).

## Install

This edition of pylmm has been tested on Python 2.7.10 with openblas
on GNU Guix. Dependencies for single and multi-core use are

    numpy
    scipy

To use these libs on GNU Guix you may need to add the following Python
path

    export PYTHONPATH="$HOME/.guix-profile/lib/python2.7/site-packages"

For CUDA dependencies are (in addition to a functional NVIDIA CUDA
installation)

    pycuda
    scikits.cuda (latest from github!)

For CUDA support you may need to add something like

    export CUDA_ROOT=/usr/local/cuda-7.0
    export PATH=$CUDA_ROOT/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_ROOT/lib64

Test CUDA support with

    ./bin/cuda_properties.py

## Run

By default pylmm tries to find CUDA and run pylmm on the GPU. If there
is no CUDA support multi-core numpy is used. To avoid CUDA use
--no-cuda. To force BLAS use the --blas switch. Examples

    ./bin/runlmm.py --help

    ./bin/runlmm.py --control data/rqtl/iron.yaml --pheno data/rqtl/iron_pheno.csv --geno data/rqtl/iron_geno.csv rqtl --pheno-column=1

and for a more extensive test

    ./bin/runlmm.py \
      --geno=../test_gn_pylmm/data/test8000.geno \
      --pheno=../test_gn_pylmm/data/test8000.pheno run

## License

pylmm is offered under the GNU Affero GPL (https://www.gnu.org/licenses/why-affero-gpl.html).
See also LICENSE.txt file that comes with the source.
