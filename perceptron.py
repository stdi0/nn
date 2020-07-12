import numpy as np
from random import random
from time import sleep

# The number of neurons per layer:

ENTERS_QTY = 10
HIDDEN_QTY = 10
OUTERS_QTY = 10

class MultilayerPerceptron():

    def __init__(self):
        self.wEH = np.array([[random()*0.3+0.1 for _ in range(ENTERS_QTY)] for _ in range(HIDDEN_QTY)])
        self.wHO = np.array([[random()*0.3+0.1 for _ in range(HIDDEN_QTY)] for _ in range(OUTERS_QTY)])

    def sigmoid(self, x, derive=False):
	    if derive:
	        return x * (1.0 - x)
	    return 1.0 / (1 + np.exp(-x))

    def calc_outer(self, enters):
        hidden = np.dot(enters, self.wEH)
        for i, j in enumerate(hidden):
        	hidden[i] = self.sigmoid(j)
        
        self.hidden = hidden

        outer = np.dot(hidden, self.wHO)
       	for i, j in enumerate(outer):
        	outer[i] = float('{:.3f}'.format(self.sigmoid(j)))

        return outer

    def study(self, patterns, answers, number_of_iterations):
        for iteration in range(number_of_iterations):
            for i, j in enumerate(patterns):
            	print('era %d' % iteration)
            	outer = self.calc_outer(j)
            	local_err = answers[i] - outer
            	err = np.dot(local_err, self.wHO.transpose())

            	for e in range(ENTERS_QTY):
            		for h in range(HIDDEN_QTY):
            			self.wEH[e][h] += 0.1 * err[h] * self.sigmoid(self.hidden[h], True) * j[e]
            	for h in range(HIDDEN_QTY):
            		for o in range(OUTERS_QTY):
            			self.wHO[h][o] += 0.1 * local_err[o] * self.sigmoid(outer[o], True) * self.hidden[h]

# Training dataset:

a =   [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] # 1+2
_a =  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
a1 =  [1, 0, 1, 0, 0, 0, 0, 0, 0, 0] # 1+3
_a1 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
a2 =  [1, 0, 0, 1, 0, 0, 0, 0, 0, 0] # 1+4
_a2 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
a3 =  [1, 0, 0, 0, 1, 0, 0, 0, 0, 0] # 1+5
_a3 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
a4 =  [1, 0, 0, 0, 0, 1, 0, 0, 0, 0] # 1+6
_a4 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
a5 =  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0] # 1+7
_a5 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
a6 =  [1, 0, 0, 0, 0, 0, 0, 1, 0, 0] # 1+8
_a6 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
a7 =  [1, 0, 0, 0, 0, 0, 0, 0, 1, 0] # 1+9
_a7 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

b =   [0, 1, 1, 0, 0, 0, 0, 0, 0, 0] # 2+3
_b =  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
b1 =  [0, 1, 0, 1, 0, 0, 0, 0, 0, 0] # 2+4
_b1 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
b2 =  [0, 1, 0, 0, 1, 0, 0, 0, 0, 0] # 2+5
_b2 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
b3 =  [0, 1, 0, 0, 0, 1, 0, 0, 0, 0] # 2+6
_b3 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
b4 =  [0, 1, 0, 0, 0, 0, 1, 0, 0, 0] # 2+7
_b4 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
b5 =  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0] # 2+8
_b5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

c =   [0, 0, 1, 1, 0, 0, 0, 0, 0, 0] # 3+4
_c =  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
c1 =  [0, 0, 1, 0, 1, 0, 0, 0, 0, 0] # 3+5
_c1 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
c2 =  [0, 0, 1, 0, 0, 1, 0, 0, 0, 0] # 3+6
_c2 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
c3 =  [0, 0, 1, 0, 0, 0, 1, 0, 0, 0] # 3+7
_c3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

d =  [0, 0, 0, 1, 1, 0, 0, 0, 0, 0] # 4+5
_d = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
d1=  [0, 0, 0, 1, 0, 1, 0, 0, 0, 0] # 4+6
_d1 =[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

patterns = np.array([a, a1, a2, a3, a4, a5, a6, a7, b, b1, b2, b3, b4, b5, c, c1, c2, c3, d, d1])
answers = np.array([_a, _a1, _a2, _a3, _a4, _a5, _a6, _a7, _b, _b1, _b2, _b3, _b4, _b5, _c, _c1, _c2, _c3, _d, _d1])

n = MultilayerPerceptron()

print('Random starting synaptic weights: ')
print(n.wEH)
print(n.wHO)

n.study(patterns, answers, 10000)

print('New synaptic weights after training: ')
print(n.wEH)
print(n.wHO)

print('Test the neural network:')

print('[1, 1, 0, 0, 0, 0, 0, 0, 0, 0] # 1+2:', n.calc_outer([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
print('[1, 0, 1, 0, 0, 0, 0, 0, 0, 0] # 1+3:', n.calc_outer([1, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
print('[1, 0, 0, 1, 0, 0, 0, 0, 0, 0] # 1+4:', n.calc_outer([1, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
print('[1, 0, 0, 0, 1, 0, 0, 0, 0, 0] # 1+5:', n.calc_outer([1, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
print('[1, 0, 0, 0, 0, 1, 0, 0, 0, 0] # 1+6:', n.calc_outer([1, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
print('[1, 0, 0, 0, 0, 0, 1, 0, 0, 0] # 1+7:', n.calc_outer([1, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
print('[1, 0, 0, 0, 0, 0, 0, 1, 0, 0] # 1+8:', n.calc_outer([1, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
print('[1, 0, 0, 0, 0, 0, 0, 0, 1, 0] # 1+9:', n.calc_outer([1, 0, 0, 0, 0, 0, 0, 0, 1, 0]))

print('[0, 1, 1, 0, 0, 0, 0, 0, 0, 0] # 2+3:', n.calc_outer([0, 1, 1, 0, 0, 0, 0, 0, 0, 0]))
print('[0, 1, 0, 1, 0, 0, 0, 0, 0, 0] # 2+4:', n.calc_outer([0, 1, 0, 1, 0, 0, 0, 0, 0, 0]))
print('[0, 1, 0, 0, 1, 0, 0, 0, 0, 0] # 2+5:', n.calc_outer([0, 1, 0, 0, 1, 0, 0, 0, 0, 0]))
print('[0, 1, 0, 0, 0, 1, 0, 0, 0, 0] # 2+6:', n.calc_outer([0, 1, 0, 0, 0, 1, 0, 0, 0, 0]))
print('[0, 1, 0, 0, 0, 0, 1, 0, 0, 0] # 2+7:', n.calc_outer([0, 1, 0, 0, 0, 0, 1, 0, 0, 0]))
print('[0, 1, 0, 0, 0, 0, 0, 1, 0, 0] # 2+8:', n.calc_outer([0, 1, 0, 0, 0, 0, 0, 1, 0, 0]))

print('[0, 0, 1, 1, 0, 0, 0, 0, 0, 0] # 3+4:', n.calc_outer([0, 0, 1, 1, 0, 0, 0, 0, 0, 0]))
print('[0, 0, 1, 0, 1, 0, 0, 0, 0, 0] # 3+5:', n.calc_outer([0, 0, 1, 0, 1, 0, 0, 0, 0, 0]))
print('[0, 0, 1, 0, 0, 1, 0, 0, 0, 0] # 3+6:', n.calc_outer([0, 0, 1, 0, 0, 1, 0, 0, 0, 0]))
print('[0, 0, 1, 0, 0, 0, 1, 0, 0, 0] # 3+7:', n.calc_outer([0, 0, 1, 0, 0, 0, 1, 0, 0, 0]))

print('[0, 0, 0, 1, 1, 0, 0, 0, 0, 0] # 4+5:', n.calc_outer([0, 0, 0, 1, 1, 0, 0, 0, 0, 0]))
print('[0, 0, 0, 1, 0, 1, 0, 0, 0, 0] # 4+6:', n.calc_outer([0, 0, 0, 1, 0, 1, 0, 0, 0, 0]))
