#GROUP MEMBERS: LUKA VUJEVA, DONG UK KIM, PIERCE BALKO, CAMERON STOCKTON

#the necessary libraries were imported
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#the data was imported for each circuit setup
data_large = np.loadtxt('data_large.txt')
data_small = np.loadtxt('data_small.txt')