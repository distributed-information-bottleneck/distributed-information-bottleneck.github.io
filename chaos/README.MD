# Code accompanying "Optimized measurements of chaotic dynamical systems via the information bottleneck" (2023)
Here we include the code to generate trajectory data for the chaotic systems in the paper (`chaos_data.py`), the C++ code to estimate the entropy rate of a sequence with the context tree weighting (CTW) method, and an iPython notebook that builds the machine learning models and trains them to optimize a measurement process (`Chaos_experiments.ipynb`).

## Infinite depth context tree weighting

To estimate the entropy rate of symbolic sequences, we implemented the infinite depth variant of CTW in C++.  Doing so provided a speedup of about 100x over a python implementation we had previously.  The C++ code can be compiled to a python extension using the following at your command line: 
```
python setup.py build_ext -i
```

Thereafter, the entropy rate of a sequence `seq` with an alphabet size (number of possible symbols) `alphabet_size` can be estimated with
```
from ctw import estimate_entropy

alphabet_size = 4
seq = np.random.randint(alphabet_size, size=10**5)

estimate_entropy(seq, alphabet_size)
```

Note: there's a hard-coded maximum depth to the tree, currently set to 512.  Periodic sequences make the tree grow in size quickly, and sequences of O(1M) symbols or longer can start to exhaust memory.