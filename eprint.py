# The only one short + flexible + portable + readable way to print to stderr
# according to Leon on StackOverflow in 2013

from __future__ import print_function
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
