import numpy as np
import pandas as pd
import jupyter
import matplotlib.pyplot as plt
import matplotlib.cm as cm




import scipy
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

learning_rate = 1e-4

num_iterations = 2500

dropout = 0.5
batch_size = 50

validation_size = 2000
image_to_display = 10

data = pd.read_csv('.../input/train.csv')
print('data({0[0]}, {0[1]})'.format(data.shape))
print(data.head())
