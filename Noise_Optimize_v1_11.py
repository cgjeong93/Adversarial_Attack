import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# for multiprocessing
import multiprocessing as mp
from multiprocessing import Process, Queue

CPU_COUNTS = mp.cpu_count()

# load datasets MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_ = x_train.copy()
x_test_ = x_test.copy()

x_train = x_train / 255.
x_test = x_test / 255.

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
print(x_test.shape)

# simple CNN Model similar with VGGNet
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same'),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=500, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=500, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# fitting
history = model.fit(x_train, y_train, epochs=5, batch_size=100, validation_split=0.25)

test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=100)

# Adding rank and tournament selections by Choi, T
# Adding one- and two-point crossovers by Choi, T
# Adding sharing method by Choi, T

"""
Visualize Genetic Algorithm to find a maximum point in a function.
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""
#DNA_RANGE = 2
DNA_RANGE = 256
DNA_SIZE = 28*28*1           # DNA length

POP_SIZE = 100          # population size
CROSS_RATE = 0.9        # mating probability (DNA crossover)
MUTATION_RATE = 0.01   # mutation probability
N_GENERATIONS = 200

SLICE_SIZE = 500*CPU_COUNTS

def mse(signal,noised_signal):
    return np.average( (signal-noised_signal)**2 )
def image_xor( image_a, image_b ):
    return image_a ^ image_b

def get_noised_image_and_mse( signal_set, noise , q_test:Queue , q_diff:Queue ):
    xx_test = [] 
    xx_diff = []
    for i in range(len(signal_set)):
        xx_test.append( image_xor(signal_set[i] , noise) ) # make errored image by exclusive or(XOR) operator
        xx_diff.append( mse(signal_set[i] , noise )) # get difference as squared scale
    q_test.put(xx_diff)
    q_diff.put(xx_diff)

# add noise image at test dataset and get accuracy at pre-trained model
def get_fitness(pop):
    fitness = []
    for i in range(POP_SIZE):
        q_test = Queue()
        q_diff = Queue()
        p = []
        # make process
        for j in range(CPU_COUNTS):
            p.append(Process(target=get_noised_image_and_mse, args=(x_test_[CPU_COUNTS*j:CPU_COUNTS*(j+1)], pop[i],q_test, q_diff))) 
        for j in range(CPU_COUNTS):
            p[j].start()
            p[j].join()
        #xx_test , xx_diff = get_noised_image_and_mse( x_test_[:SLICE_SIZE], pop[i])
        xx_test, xx_diff = [], []
        for _ in range(CPU_COUNTS):
            xx_test.append(q_test.get()) 
            xx_diff.append(q_diff.get())


        # get accuracy rate
        loss, acc = model.evaluate(
            x = np.array(xx_test, dtype=np.float).reshape(-1,28,28,1) / 255. ,
            y = y_test[:SLICE_SIZE],
            batch_size=100,
            verbose=0)
        if(np.average(xx_diff) < 2.25e2):
            fitness.append(test_acc - acc - np.average(xx_diff)/1e1 )
        else:
            fitness.append( -9.9e10 )
    print(np.average(fitness))

    return fitness

#def get_fitness(pred_acc, pred_mse):
#    return (test_acc - np.array(pred_acc))*100

# convert binary DNA to decimal and normalize
def translateDNA(pop):
    pop = pop.reshape(POP_SIZE,28,28,1)
    return pop

# Added by Choi, T for EA lectures
def tournament_select(pop, fitness, tournament_size=2):
    idx = []
    for _ in range(POP_SIZE):
        participants = np.random.choice(
            np.arange(POP_SIZE), size=tournament_size, replace=False)
        participants_fitness = list(np.array(fitness)[participants])
        winner = participants_fitness.index(max(participants_fitness))
        idx.append(participants[winner])
    return pop[idx]


def crossover(parent, pop):  # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        # select another individual from pop
        i_ = np.random.randint(0, POP_SIZE, size=1)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)    # choose crossover points
        # mating and produce one child
        parent[cross_points] = pop[i_, cross_points]
    return parent


def one_point_crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        # select another individual from pop
        i_ = np.random.randint(0, POP_SIZE, size=1)
        j_ = np.random.randint(1, DNA_SIZE - 1, size=1)
        flag = True if np.random.randint(0, 2) < 0.5 else False
        cross_points = [flag] * DNA_SIZE
        cross_points[int(j_):] = [not flag] * len(cross_points[int(j_):])
        # mating and produce one child
        parent[cross_points] = pop[i_, cross_points]
    return parent


def two_point_crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        # select another individual from pop
        i_ = np.random.randint(0, POP_SIZE, size=1)
        j_ = np.sort(np.random.choice(
            np.arange(DNA_SIZE) - 2, size=2, replace=False) + 1)
        flag = True if np.random.randint(0, 2) < 0.5 else False
        cross_points = [flag] * DNA_SIZE
        cross_points[int(j_[0]):int(j_[1])] = [not flag] * \
            len(cross_points[int(j_[0]):int(j_[1])])
        # mating and produce one child
        parent[cross_points] = pop[i_, cross_points]
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = np.random.randint(DNA_RANGE)
    return child


pop = np.random.randint(DNA_RANGE, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA

M_ = [] # maximum of generation
A_ = [] # average of 
N_ = [] # minimum of

for _ in range(N_GENERATIONS):
    # GA part (evolution)

    fitness = get_fitness(translateDNA(pop))

    pop = tournament_select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child   # parent is replaced by its child

    # max,avg,min sets
    M_.append(np.max(fitness))
    A_.append(np.average(fitness))
    N_.append(np.min(fitness))
    print(_+1, 'Gens :', M_[-1], A_[-1], N_[-1])
pop = translateDNA(pop)

# drawing result
for i in range(3):
    plt.subplot(3,3,i*3+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.subplot(3,3,i*3+2)
    fnt = list(fitness)
    print((pop[fnt.index(max(fnt))]).reshape(28,28,1).shape)
    plt.imshow((pop[fnt.index(max(fnt))]).reshape(28,28)*255 ,cmap='gray')
    plt.subplot(3,3,i*3+3) # merged image
    plt.imshow((x_test_[i] ^ pop[fnt.index(max(fnt))]).reshape(28,28)*255 ,cmap='gray')
plt.show()

plt.plot(M_, 'r-', label='Max')
plt.plot(A_, 'g-', label='Avg')
plt.plot(N_, 'b-', label='Min')
plt.show()