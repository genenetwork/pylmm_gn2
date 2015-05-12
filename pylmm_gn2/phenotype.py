# Phenotype routines

# Copyright (C) 2013  Nicholas A. Furlotte (nick.furlotte@gmail.com)
# Copyright (C) 2015  Pjotr Prins (pjotr.prins@thebird.nl)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import numpy as np

# ---- A trick to decide on the environment:
try:
    from wqflask.my_pylmm.pyLMM import chunks
    from gn2 import uses, progress_set_func
except ImportError:
    has_gn2=False
    import standalone as handlers
    from standalone import uses, progress_set_func

progress,debug,info,mprint = uses('progress','debug','info','mprint')

def remove_missing(n,y,g):
    """
    Remove missing data from matrices, make sure the genotype data has
    individuals as rows
    """
    assert(y is not None)
    assert y.shape[0] == g.shape[0],"y (n) %d, g (n,m) %s" % (y.shape[0],g.shape)

    y1 = y
    g1 = g
    v = np.isnan(y)
    keep = True - v
    if v.sum():
        info("runlmm.py: Cleaning the phenotype vector and genotype matrix by removing %d individuals...\n" % (v.sum()))
        y1 = y[keep]
        g1 = g[keep,:]
        n = y1.shape[0]
    return n,y1,g1,keep

def remove_missing_new(n,y):
    """
    Remove missing data. Returns new n,y,keep
    """
    assert(y is not None)
    y1 = y
    v = np.isnan(y)
    keep = True - v
    if v.sum():
        info("runlmm.py: Cleaning the phenotype vector by removing %d individuals" % (v.sum()))
        y1 = y[keep]
        n = y1.shape[0]
    return n,y1,keep

