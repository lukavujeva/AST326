#GROUP MEMBERS: LUKA VUJEVA, DONG UK KIM, PIERCE BALKO, CAMERON STOCKTON

#the necessary libraries were imported
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import matplotlib.mlab as mlab

#the data was imported for each data group
data_large = np.loadtxt('data_large.txt')
data_small = np.loadtxt('data_small.txt')
distance_data = np.loadtxt('distance_measurement.txt')
countrate = np.loadtxt('photon_countrate.txt')

###############################################################################
############################## PROBLEM 1 ######################################
###############################################################################

#the distance data was spliced into its values of distance and measurement error

distance = np.ndarray.flatten(distance_data[:, 0:1])
error = np.ndarray.flatten(distance_data[:, 1:2])

#the number of total measurements was extracted from the data
N = len(distance)

#the mean of each dataset was calculated
mean = (1/N)*(np.sum(distance))

#the standard deviation was calculated

std = np.sqrt(np.sum((distance - mean)**2)/(N-1))

#now for the weighted mean

weight  = 1 / (error*2)

weighted_mean = np.sum(weight*distance)/np.sum(weight)

#now for the wei
weighted_std = weighted_mean/np.sum(weight*distance)


#The distance data was plotted just as it was in fig. 3
plt.plot(distance, linestyle = 'none', marker = '.', color = 'red')
plt.xlabel('Measurement')
plt.ylabel('Distance (pc)')
plt.title('Measurements vs Distance')
plt.savefig('data_p1_plot.pdf')
plt.show()

#the plots of each dataset were shown as a histogram with the gaussian and poisson distributions plotted on top
plt.hist(data_large, rwidth = std, bins = 14, color = 'red')

plt.title('Distance vs Number of Measurements')
plt.xlabel('Distance (pc)')
plt.ylabel('Number of Measurements')
plt.savefig('data_p1_hist.pdf')
plt.show()

print("Therefore the standard deviation is: ", std)
std_2 = (1/(N-1))


###############################################################################
############################## PROBLEM 2 ######################################
###############################################################################

#the number of total measurements was extracted from the data
N_countrate = len(countrate)

#the mean of each dataset was calculated
mean_countrate = (1/N_countrate)*(np.sum(countrate))

#the standard deviation was calculated
std_countrate = np.sqrt(np.sum((countrate - mean_countrate)**2)/(N_countrate-1))

#the plots of each dataset were shown as a histogram with the gaussian and poisson distributions plotted on top
plt.hist(countrate, bins = 8, normed = True,  color = 'green')

# plot Poisson distribution
plt.plot(countrate, poisson.pmf(countrate, mean_countrate), linestyle='none', marker='o', 
         color='red', label='Poisson Distribution')
sigma = mean_countrate**(1/2)

plt.title('Number of Photons vs Number of Measurements (Normalized)')
plt.xlabel('Number of Photons')
plt.ylabel('Number of Measurements')
plt.savefig('data_p2_hist.pdf')
plt.show()

print("Therefore the standard deviation is: ", std_countrate)
###############################################################################
############################## PROBLEM 3 ######################################
###############################################################################

#the number of total measurements was extracted from the data
N = len(data_large)

#the mean of each dataset was calculated
mean_large = (1/N)*(np.sum(data_large))
mean_small = (1/N)*(np.sum(data_small))

#the standard deviations were calculated for both datasets
std_large = np.sqrt(np.sum((data_large - mean_large)**2)/(N-1))

std_small = np.sqrt(np.sum((data_small - mean_small)**2)/(N-1))

#The large distance data was plotted just as it was in fig. 3
plt.plot(data_large, linestyle = 'none', marker = '.', color = 'blue')
plt.xlabel('Measurement')
plt.ylabel('Distance (pc)')
plt.title('Large Measurements vs Distance')
plt.savefig('data_large_p3_plot.pdf')
plt.show()

#The small distance data was plotted just as it was in fig. 3
plt.plot(data_small, linestyle = 'none', marker = '.', color = 'blue')
plt.xlabel('Measurement')
plt.ylabel('Distance (pc)')
plt.title('Small Measurements vs Distance')
plt.savefig('data_small_p3_plot.pdf')
plt.show()

#the plots of each dataset were shown as a histogram with the gaussian and poisson distributions plotted on top
plt.hist(data_large, bins = 14, normed = True)

# plot Gaussian distribution
sigma_large = mean_large **(1/2)
x = np.linspace(mean_large-3*sigma_large, mean_large+3*sigma_large, 100)

plt.plot(x, mlab.normpdf(x, mean_large, sigma_large), color='yellow', linewidth='1.0', 
         label='Gaussian Distribution')

# plot Poisson distribution
plt.plot(data_large, poisson.pmf(data_large, mean_large), linestyle='none', marker='o', 
         color='red', label='Poisson Distribution')
sigma = mean_large**(1/2)

plt.title('Distance vs Measurements for Large Data (Normalized)')
plt.xlabel('Distance (pc)')
plt.ylabel('Number of Measurements')
plt.legend()
plt.savefig('data_large_hist.pdf')
plt.show()



plt.hist(data_small, bins = 12, normed = True)

# plot Gaussian distribution
sigma_small = mean_small **(1/2)
x = np.linspace(mean_small-3*sigma_small, mean_small+3*sigma_small, 100)

plt.plot(x, mlab.normpdf(x, mean_small, sigma_small), color='yellow', linewidth='1.0', 
         label='Gaussian Distribution')

# plot Poisson distribution
plt.plot(data_small, poisson.pmf(data_small, mean_small), linestyle='none', marker='o', 
         color='red', label='Poisson Distribution')
sigma = mean_small**(1/2)

plt.title('Distance vs Measurements for Small Data (Normalized)')

plt.xlabel('Distance (pc)')
plt.ylabel('Number of Measurements')
plt.legend()
plt.savefig('data_small_hist.pdf')
plt.show()

print("Therefore the standard deviation of the large dataset is, ", std_large)
print("Therefore the standard deviation of the large dataset is, ", std_small)
