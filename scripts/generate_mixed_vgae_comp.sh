#!/usr/bin/env bash

mkdir -p data/mixed

# python code/randkcnf.py data/mixed/1 600 3 5 21 0 # yes
# python code/domset.py data/mixed/1 600 5 0.2 2 600 # yes
# python code/kclique.py data/mixed/1 600 5 0.2 3 1200 # yes

# python code/randkcnf.py data/mixed/1 600 3 10 43 1800 # yes
# python code/domset.py data/mixed/1 600 7 0.2 3 2400 # yes
# python code/kclique.py data/mixed/1 600 10 0.1 3 3600 # yes
# python code/randkcnf.py data/mixed/1 600 3 13 49 4200 # yes
# python code/domset.py data/mixed/1 600 9 0.2 5 4800 # yes
# python code/kclique.py data/mixed/1 600 12 0.3 4 5400 # yes

# python code/kcolor.py data/mixed/1comp 600 5 0.5 3 6000
# python code/kcolor.py data/mixed/1comp 600 10 0.5 3 6600
# python code/kcolor.py data/mixed/1comp 600 15 0.5 4 7200


# python code/randkcnf.py data/mixed/2 600 3 10 43 0
# python code/domset.py data/mixed/2 600 7 0.2 3 600
# python code/kclique.py data/mixed/2 600 10 0.1 3 1200
# python code/kcolor.py data/mixed/2comp 600 12 0.5 4 1200
