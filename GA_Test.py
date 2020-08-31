import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TotalWeightCapacity = 13743

DNA_SIZE = 500           # DNA length
POP_SIZE = 100          # population size
CROSS_RATE = 0.8        # mating probability (DNA crossover)
MUTATION_RATE = 0.003   # mutation probability
N_GENERATIONS = 100

#load data slicing
df = pd.read_csv('C:\\Project\\Pratice\\TestData(0-1Knapsack).txt', sep='\t')
item = df.to_numpy()[:,1:].swapaxes(0,1)



# find non-zero fitness for selection
def get_fitness(pop):
    weight_pop = pop.dot(item[0])
    profit_pop = pop.dot(item[1])
    for i in range(POP_SIZE):
        if weight_pop[i] > TotalWeightCapacity: # penalty
            profit_pop[i] = 1e-3 

    return profit_pop


# Added by Choi, T for EA lectures
def get_sharing_fitness(pop, min_dist=1.5):
    fitness = get_fitness(pop)

    for i in range(POP_SIZE):
        denom = 1
        for j in range(POP_SIZE):
            dist = (pop[i] != pop[j]).sum()
            if dist < min_dist:
                denom += 1 - dist / min_dist
        fitness[i] /= denom
    return fitness


def select(pop, fitness):   # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


# Added by Choi, T for EA lectures
def rank_select(pop, fitness):
    # Efficient method to calculate the rank vector of a list in Python
    # https://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    def rank_simple(vector):
        return sorted(range(len(vector)), key=vector.__getitem__)

    def rankdata(a):
        n = len(a)
        ivec = rank_simple(a)
        svec = [a[rank] for rank in ivec]
        sumranks = 0
        dupcount = 0
        newarray = [0]*n
        for i in range(n):
            sumranks += i
            dupcount += 1
            if i == n-1 or svec[i] != svec[i+1]:
                averank = sumranks / float(dupcount) + 1
                for j in range(i-dupcount+1, i+1):
                    newarray[ivec[j]] = averank
                sumranks = 0
                dupcount = 0
        return newarray

    rank_fitness = rankdata(fitness)
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=list(map(lambda x: x / sum(rank_fitness), rank_fitness)))
    return pop[idx]


# Added by Choi, T for EA lectures
def tournament_select(pop, fitness, tournament_size=2):
    idx = []
    for _ in range(POP_SIZE):
        participants = np.random.choice(
            np.arange(POP_SIZE), size=tournament_size, replace=False)
        participants_fitness = list(np.array(fitness)[participants])
        winner = participants_fitness.index(np.max(participants_fitness))
        idx.append(participants[winner])
    return pop[idx]


def crossover(parent, pop):  # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        # select another individual from pop
        i_ = np.random.randint(0, POP_SIZE, size=1)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(
            np.bool)    # choose crossover points
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
            child[point] = 1 if child[point] == 0 else 0
    return child

# for graph
x = range(1,POP_SIZE+1,1)
N_TRIALS = 30

# test 30 times
result_dict = {}

for (crs, i) in [[crossover,1], [one_point_crossover,2], [two_point_crossover,3]]:
    # crossover functions
    if crs == crossover:
        crossover_name = 'crossover'
    elif crs == one_point_crossover:
        crossover_name = 'one_point_crossover'
    else:
        crossover_name = 'two_point_crossover'
    
    for (slt, j) in [[select,1], [tournament_select,2], [rank_select,3]]:
        # select functions
        if slt == select:
            select_name = 'select'
        elif slt == tournament_select:
            select_name = 'tournament_select'
        else:
            select_name = 'rank_select'
        
        # test 30 times
        trial_M = []
        trial_A = []
        trial_N = []
        for non in range(N_TRIALS):
            pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE) )   # initialize the pop DNA

            M_ = [] # maximum of generation
            A_ = [] # average of 
            N_ = [] # minimum of

            for _ in range(N_GENERATIONS):
                # GA part (evolution)
                fitness = get_fitness(pop)

                pop = slt(pop, fitness)
                pop_copy = pop.copy()
                for parent in pop:
                    child = crs(parent, pop_copy)
                    child = mutate(child)
                    parent[:] = child   # parent is replaced by its child

                # max.avg.min(except 0 value)
                M_.append(np.max(fitness))
                
                a = []
                for k in range(POP_SIZE):
                    if fitness[k] > 0:
                        a.append(fitness[k])
                A_.append(np.average(a))
                N_.append(np.min(a))
            trial_M.append(M_)
            trial_A.append(A_)
            trial_N.append(N_)
            print(crossover_name , '&' , select_name , 'Trial : %3d' % (non+1))
        
        x = range(1,POP_SIZE+1)
        plt.subplot(3, 3, (i-1)*3+j)
        plt.title(crossover_name + '&' +select_name)
        plt.plot(x,np.average(trial_M,axis=0),'r-',label='Max')
        plt.plot(x,np.average(trial_A,axis=0),'g-',label='Avg')
        plt.plot(x,np.average(trial_N,axis=0),'b-',label='Min')
        plt.legend()
plt.show()