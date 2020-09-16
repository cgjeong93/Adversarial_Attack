from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision

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

POP_SIZE = 150          # population size
CROSS_RATE = 0.9        # mating probability (DNA crossover)
MUTATION_RATE = 0.05   # mutation probability
N_GENERATIONS = 50

UNIT_SIZE = 100
SLICE_SIZE = UNIT_SIZE*CPU_COUNTS

TEST_BATCH_SIZE = SLICE_SIZE

epsilon = 0.01


# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test(model, device, test_loader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)

# Adding rank and tournament selections by Choi, T
# Adding one- and two-point crossovers by Choi, T
# Adding sharing method by Choi, T

"""
Visualize Genetic Algorithm to find a maximum point in a function.
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""

def mse(signal,noised_signal):
    return np.average( (signal-noised_signal)**2 )
def image_xor( image_a, image_b ):
    return image_a * (1-epsilon) + image_b * epsilon
    # return image_a ^ image_b

def get_noised_image_and_mse( signal_set, noise , q_test:Queue , q_diff:Queue ):
    xx_test = [] 
    #xx_diff = []
    for i in range(len(signal_set)):
        xx_test.append( image_xor(signal_set[i] , noise) ) # make errored image by exclusive or(XOR) operator
        #xx_diff.append( mse(signal_set[i] , noise )) # get difference as squared scale
    q_test.put(xx_test)
    #q_diff.put(xx_diff)
    return xx_test#, xx_diff

# add noise image at test dataset and get accuracy at pre-trained model
def get_fitness(load_model, pop, dataset, device, test_acc, kwargs, transform):
    fitness = []
    for i in range(POP_SIZE):
        #start = time.time()
        # sharing Queue
        q_test = Queue()
        q_diff = Queue()
        p = [] # process handler

        # dataset
        adv_dataset = datasets.CIFAR10('../cifar10_data', train=False, download=False,
                    transform=transform)

        # slicing
        adv_dataset.data = dataset.data[:SLICE_SIZE]
        adv_dataset.targets = dataset.targets[:SLICE_SIZE]
        x_test_ = adv_dataset.data
        #x_test_.to(device)

        # make process
        for j in range(CPU_COUNTS):
            p.append(Process(target=get_noised_image_and_mse, args=(x_test_[UNIT_SIZE*j:UNIT_SIZE*(j+1)], pop[i],q_test, q_diff))) 
            p[j].start()

    
        xx_test, xx_diff = [], []
        # xx_test , xx_diff = get_noised_image_and_mse( x_test_[:SLICE_SIZE], pop[i])
        
        
        for _ in range(CPU_COUNTS):
            xx_test.append(q_test.get()) 
            #xx_diff.append(q_diff.get())
        adv_dataset.data = np.array(xx_test,dtype=np.uint8).reshape(-1,32,32,3)

        data_loader = torch.utils.data.DataLoader(adv_dataset, **kwargs)
        # get accuracy rate
        acc = test(load_model, device, data_loader)
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
    # Use GPU
    device = torch.device("cuda")

    kwargs = {'batch_size': TEST_BATCH_SIZE}
    kwargs.update({'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False},
                    )

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    # original datasets
    dataset = datasets.CIFAR10('../cifar10_data', train=False, download=False,
                       transform=transform)

    # slicing
    dataset.data = dataset.data[:SLICE_SIZE]
    dataset.targets = dataset.targets[:SLICE_SIZE]
    # test_loader
    test_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    # Load Model
    load_model = VGG('VGG19').to(device)
    load_model.load_state_dict(torch.load("cifar_cnn.pt"))
    load_model.eval()
    test_acc = test(load_model, device, test_loader)
    print( 'Test accuracy : ' , '%.2f' % test_acc )

 
    pop = np.random.randint(DNA_RANGE, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA

    M_ = [] # maximum of generation
    A_ = [] # average of 
    N_ = [] # minimum of

    for _ in range(N_GENERATIONS):
        # GA part (evolution)
        start = time.time()

        fitness = get_fitness(
            load_model, translateDNA(pop), 
            dataset, device, test_acc, kwargs, transform)

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
        print(_+1, 'Gens :', M_[-1], A_[-1], N_[-1], " %.2f secs"%(time.time() - start))

    pop = translateDNA(pop)

    temp_img = np.array(dataset.data[0]).reshape(32,32,3)
    for i in range(3):
        plt.subplot(3,3,i*3+1)
        plt.imshow(temp_img)
        plt.subplot(3,3,i*3+2)
        fnt = list(fitness)
        plt.imshow((pop[fnt.index(max(fnt))]).reshape(32,32,3))
        plt.subplot(3,3,i*3+3) # merged image
        plt.imshow(image_xor(temp_img , pop[fnt.index(max(fnt))].reshape(32,32,3)).reshape(32,32,3))
    plt.show()

    plt.plot(M_, 'r-', label='Max')
    plt.plot(A_, 'g-', label='Avg')
    plt.plot(N_, 'b-', label='Min')
    plt.show()


if __name__ == '__main__':
    main()