import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time

DNA_RANGE = 256
DNA_SIZE = 28*28        # DNA length

# load datasets MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

pop = np.random.randint(DNA_RANGE, size=(DNA_SIZE)).reshape(28,28)   # initialize the pop DNA

def mse(signal,noised_signal):
    return np.average( (signal-noised_signal)**2 )

xx_test = []
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow( x_test[0] ^ np.array(pop/3**i, dtype='int8') , cmap='gray')
    plt.title( '%d mse:%.2f' % (i, mse(x_test[0], x_test[0] ^ np.array(pop/3**i, dtype='int8') ) ) )
    plt.axis('off')
plt.show()
