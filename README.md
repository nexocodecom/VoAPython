# VoAPython
Implementation of tools proposed in the paper "Validation of Association" by B. Ćmiel and T. Ledwina
https://arxiv.org/pdf/1904.06519

# Installation
pip install git+https://github.com/nexocodecom/VoAPython.git

# Example 
```python
import numpy as np
import voa
import random
from voa import create_Q_plot


X = np.array(list(range(100)))
Y = np.array([x**2 for x in X])

print(X)
print(Y)

# Create Q plot
result = create_Q_plot(X, Y, k_plot_grid=100, MC=100, display=False)

# The result contains the plot and related data
fig = result['Q_plot']
fig.show()
```
