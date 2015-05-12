# Standard file readers
#
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
import os
import numpy as np
import csv

def kinship(fn):
    K1 = []
    print fn
    with open(fn,'r') as tsvin:
        assert(tsvin.readline().strip() == "# Kinship format version 1.0")
        tsvin.readline()
        tsvin.readline()
        tsv = csv.reader(tsvin, delimiter='\t')
        for row in tsv:
            ns = np.genfromtxt(row[1:])
            K1.append(ns) # <--- slow
    K = np.array(K1)
    return K

def pheno(fn):
    Y1 = []
    print fn
    with open(fn,'r') as tsvin:
        assert(tsvin.readline().strip() == "# Phenotype format version 1.0")
        tsvin.readline()
        tsvin.readline()
        tsvin.readline()
        tsv = csv.reader(tsvin, delimiter='\t')
        for row in tsv:
            ns = np.genfromtxt(row[1:])
            Y1.append(ns) # <--- slow
    Y = np.array(Y1)
    return Y

def geno(fn):
    G1 = []
    hab_mapper = {'A':0,'H':1,'B':2,'-':3}
    pylmm_mapper = [ 0.0, 0.5, 1.0, float('nan') ]

    print fn
    with open(fn,'r') as tsvin:
        line = tsvin.readline().strip()
        assert line == "# Genotype format version 1.0", line
        tsvin.readline()
        tsvin.readline()
        tsvin.readline()
        tsvin.readline()
        tsv = csv.reader(tsvin, delimiter='\t')
        for row in tsv:
            # print(row)
            id = row[0]
            gs = list(row[1])
            # print id,gs
            gs2 = [pylmm_mapper[hab_mapper[g]] for g in gs]
            # print id,gs2
            # ns = np.genfromtxt(row[1:])
            G1.append(gs2) # <--- slow
    G = np.array(G1)
    return G

def geno(fn):
    G1 = []
    for id,values in geno_iter(fn):
        G1.append(values) # <--- slow
    G = np.array(G1)
    return G

def geno_callback(fn,func):
    hab_mapper = {'A':0,'H':1,'B':2,'-':3}
    pylmm_mapper = [ 0.0, 0.5, 1.0, float('nan') ]

    print fn
    with open(fn,'r') as tsvin:
        assert(tsvin.readline().strip() == "# Genotype format version 1.0")
        tsvin.readline()
        tsvin.readline()
        tsvin.readline()
        tsvin.readline()
        tsv = csv.reader(tsvin, delimiter='\t')
        for row in tsv:
            id = row[0]
            gs = list(row[1])
            gs2 = [pylmm_mapper[hab_mapper[g]] for g in gs]
            func(id,gs2) 

def geno_iter(fn):
    """
    Yield a tuple of snpid and values
    """
    hab_mapper = {'A':0,'H':1,'B':2,'-':3}
    pylmm_mapper = [ 0.0, 0.5, 1.0, float('nan') ]

    print fn
    with open(fn,'r') as tsvin:
        assert(tsvin.readline().strip() == "# Genotype format version 1.0")
        tsvin.readline()
        tsvin.readline()
        tsvin.readline()
        tsvin.readline()
        tsv = csv.reader(tsvin, delimiter='\t')
        for row in tsv:
            id = row[0]
            gs = list(row[1])
            gs2 = [pylmm_mapper[hab_mapper[g]] for g in gs]
            yield (id,gs2) 
