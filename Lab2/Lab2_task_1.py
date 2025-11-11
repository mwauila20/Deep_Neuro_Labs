import time

from random import randint

import math as M

#Задание 1

list_1 = []
list_length = 10

min_value = 1

max_value = 20

for i in range(list_length):
    list_1.append(randint(min_value,max_value))
print(list_1)

import random

list_2 = random.sample(range(1,20),10)
print(list_2)

sum=0
for i in list_1:
    if( i % 2 == 0 ):
        sum += i
print(sum)

