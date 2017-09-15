import sys
sys.path.append("../")
import numpy as np
from rindex import IndexVectorSciPy as IV


n = 1000
c = 0

civ = IV.IndexVector(dim=1000, n=100)
v = np.array(civ.create_index_vector_from_ordered_context(["hello"]))


for i in range (0,n):
	w = np.array(civ.create_index_vector_from_context(["hello"]))
	if not np.array_equal(v,w):
		c=c+1

print("Found %i different Vectors for \"hello\"" % (c) )
