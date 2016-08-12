# R/qtl file reader
#
# Copyright (C) 2015,2016  Pjotr Prins (pjotr.prins@thebird.nl)
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import os
import numpy as np
import yaml
import csv
import re

def control(fn):
    """
    Load the Rqtl2 control file
    """
    with open(fn, "r") as f:
       ctrl = yaml.load(f)
       return ctrl

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
    """
    Fetch the Rqtl2 phenotypes and return matrix

    [[   61.92   153.16]
     [   88.33   178.58]
     [   58.     131.91]
     ...]]

    and list of column headers, e.g. ['liver', 'spleen']

    FIXME: only passes in the first list, also handle missing values
    """
    Y1 = []
    ynames = None
    print fn
    with open(fn,'r') as tsvin:
        tsv = csv.reader(tsvin)
        ynames = tsv.next()[1:]
        p = re.compile('\.')
        for n in ynames:
            assert not p.match(n), "Phenotype header %s appears to contain number" % n
        for row in tsv:
            ns = np.genfromtxt(row[1:])
            Y1.append(ns[0]) # <--- slow
    Y = np.array(Y1)
    return Y,ynames

def geno(fn, ctrl):
    """
    Read Rqtl2 genotype file and list of marker names.
    """
    G1 = []
    gnames = None
    # for pylmm convert 'genotypes': {'A': 1, 'H': 2, 'B': 3}, 'na.strings': ['-', 'NA'] to
    #  [ 0.0, 0.5, 1.0, NaN ]
    if 'na.strings' not in ctrl:
        ctrl['na.strings'] = ['-', 'NA']
    # hab_mapper = ctrl['genotypes']
    hab_mapper = {}
    for k in ctrl['genotypes']:
        hab_mapper[k] = int(ctrl['genotypes'][k])

    print hab_mapper
    idx = len(hab_mapper)
    assert(idx == 3), hab_mapper  # this is what we expect now
    # Note R/qtl hab_mapper is 1 indexed
    pylmm_mapper = [ None, 0.0, 0.5, 1.0 ]
    for s in ctrl['na.strings']:
        idx += 1
        hab_mapper[s] = idx
        pylmm_mapper.append(float('nan'))
    print 'hab_mapper', hab_mapper
    print 'pylmm_mapper', pylmm_mapper
    print fn
    with open(fn,'r') as tsvin:
        tsv = csv.reader(tsvin)
        gnames = tsv.next()[1:]
        print gnames
        for row in tsv:
            # print(row)
            id = row[0]
            gs = row[1:]
            # print id,gs
            # Convert all items to genotype values
            gs2 = [pylmm_mapper[hab_mapper[g]] for g in gs]
            # print id,gs2
            # ns = np.genfromtxt(row[1:])
            G1.append(gs2) # <--- slow
    G = np.array(G1)
    if 'geno_transposed' in ctrl and ctrl['geno_transposed']:
      return G,gnames
    return G.T,gnames

def geno_callback(fn,func):
    hab_mapper = {'A':0,'H':1,'B':2,'-':3}
    pylmm_mapper = [ 0.0, 0.5, 1.0, float('nan') ]

    raise "NYI"
    print fn
    with open(fn,'r') as csvin:
        assert(csvin.readline().strip() == "# Genotype format version 1.0")
        csvin.readline()
        csvin.readline()
        csvin.readline()
        csvin.readline()
        tsv = csv.reader(csvin, delimiter='\t')
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
    with open(fn,'r') as csvin:
        assert(csvin.readline().strip() == "# Genotype format version 1.0")
        csvin.readline()
        csvin.readline()
        csvin.readline()
        csvin.readline()
        tsv = csv.reader(csvin, delimiter='\t')
        for row in tsv:
            id = row[0]
            gs = list(row[1])
            gs2 = [pylmm_mapper[hab_mapper[g]] for g in gs]
            yield (id,gs2)
