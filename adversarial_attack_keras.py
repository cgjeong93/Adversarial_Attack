import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta

import numpy as np
import matplotlib.pyplot as plt
import time

# for multiprocessing
import multiprocessing as mp
from multiprocessing import Process, Queue

CPU_COUNTS = mp.cpu_count()

#DNA_RANGE = 2
DNA_RANGE = 256
DNA_SIZE = 32*32*3           # DNA length

POP_SIZE = 250          # population size
CROSS_RATE = 0.9        # mating probability (DNA crossover)
MUTATION_RATE = 0.05   # mutation probability
N_GENERATIONS = 200

UNIT_SIZE = 100
SLICE_SIZE = UNIT_SIZE*CPU_COUNTS

TEST_BATCH_SIZE = SLICE_SIZE

epsilon = 0.01

"""
Visualize Genetic Algorithm to find a maximum point in a function.
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""

def mse(signal,noised_signal):
    return np.average( (signal-noised_signal)**2 )
def image_epsilon( image_a, image_b ):
    result = (image_a * (1-epsilon) ) + (image_b * epsilon)
    return result
    # return image_a ^ image_b

def min_max_normalization( image ):
    temp = np.array(image)
    temp = (temp - temp.min()) / temp.max()
    return temp

def get_noised_image_and_mse( signal_set, noise , q_test:Queue , q_diff:Queue ):
    xx_test = [] 
    xx_diff = []
    for i in range(len(signal_set)):
        xx_test.append( image_epsilon(signal_set[i] , noise) ) # make errored image by exclusive or(XOR) operator
        xx_diff.append( mse(signal_set[i] , noise )) # get difference as squared scale
    return xx_test, xx_diff
def get_noised_image( signal_set, noise ):
    xx_test = [] 
    for i in range(len(signal_set)):
        xx_test.append( image_epsilon(signal_set[i] , noise) ) # make errored image by exclusive or(XOR) operator
    return xx_test



# add noise image at test dataset and get accuracy at pre-trained model
def get_fitness(model, pop, x_test, y_test, test_acc):
    fitness = []
    for i in range(POP_SIZE):
        xx_test = np.array(get_noised_image(x_test[:SLICE_SIZE], pop[i]), dtype=np.float)/255.
        
        # get accuracy rate
        loss, acc = model.evaluate(xx_test[:SLICE_SIZE], y_test[:SLICE_SIZE], batch_size=1000, verbose=0)
        # if(np.average(xx_diff) < 1.25e3):
        fitness.append(test_acc - acc)# - np.average(xx_diff)/1e4 )
        # else:
        #     fitness.append(test_acc - acc - np.average(xx_diff)/1e1 )
        #print("{:.2}secs".format(time.time()-start))
    print('Acc: {:.6f}'.format(acc),end=' ')

    return fitness

#def get_fitness(pred_acc, pred_mse):
#    return (test_acc - np.array(pred_acc))*100

# convert binary DNA to decimal and normalize
def translateDNA(pop):
    pop = pop.reshape(POP_SIZE,32,32,3)
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



def main():
    model = tf.keras.models.load_model("keras_cifar10_VGG19.h5")
    model.compile(optimizer=Adadelta(learning_rate=1.0), loss='sparse_categorical_crossentropy',metrics=['acc'])

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    test_loss, test_acc = model.evaluate(x_test/255. , y_test, batch_size=1000)
    print( 'Test accuracy : ' , '%.2f' % test_acc )

    pop = np.random.randint(DNA_RANGE, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA

    M_ = [] # maximum of generation
    A_ = [] # average of 
    N_ = [] # minimum of

    for _ in range(N_GENERATIONS):
        # GA part (evolution)
        start = time.time()

        fitness = get_fitness(
            model, translateDNA(pop), 
            x_test, y_test, test_acc)

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
        print(_+1, 'Gens :', '%.6f'% M_[-1], '%.6f'% A_[-1], '%.6f'% N_[-1], " %.2f secs"%(time.time() - start))

    pop = translateDNA(pop)
    idx_img = np.random.randint(9998)
    for i in range(3):
        temp_img = np.array(x_test[idx_img])
        plt.subplot(3,3,i*3+1)
        plt.imshow(temp_img)
        plt.subplot(3,3,i*3+2)
        #fnt = list(fitness) #fnt.index(max(fnt))
        plt.imshow(pop[0])
        plt.subplot(3,3,i*3+3) # merged image
        temp_img = image_epsilon(temp_img , pop[0 ])
        temp_img = np.array(temp_img , dtype=np.uint8)
        plt.imshow( temp_img )
    plt.show()

    plt.plot(M_, 'r-', label='Max')
    plt.plot(A_, 'g-', label='Avg')
    plt.plot(N_, 'b-', label='Min')
    plt.show()


if __name__ == '__main__':
    main()