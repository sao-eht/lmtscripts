import corr, adc5g, httplib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import sys, time


#if0_hot = np.load('dataSamp_hotload_if0full.npy')
#if1_hot = np.load('dataSamp_hotload_if1full.npy')
#if0_cold = np.load('dataSamp_coldload_if0full.npy')
#if1_cold = np.load('dataSamp_coldload_if1full.npy')

# read in the filenames
if len(sys.argv) < 5:
    if0_hot_filename = 'dataSamp_hotload_if0full.npy'
    if1_hot_filename = 'dataSamp_hotload_if1full.npy'
    if0_cold_filename = 'dataSamp_coldload_if0full.npy'
    if1_cold_filename = 'dataSamp_coldload_if1full.npy'
else:
    if0_hot_filename = sys.argv[1]
    if1_hot_filename = sys.argv[2]
    if0_cold_filename = sys.argv[3]
    if1_cold_filename = sys.argv[4]
 
# read in the length of the snippets we want to analyze   
if len(sys.argv) < 6:
    snippetLen = 512
else:
    snippetLen = sys.argv[5]

# read in sampling rate
if len(sys.argv) < 7:
    samp_rate = 4096e6
else:
    snippetLen = sys.argv[6] 


# load the data
if0_hot = np.load(if0_hot_filename)
if1_hot = np.load(if1_hot_filename)
if0_cold = np.load(if0_cold_filename)
if1_cold = np.load(if1_cold_filename)

# set the offset from lindy's code
offset = 1; 

# get the number of snippets and the original number of samples
nSnippets = if0_hot.shape[0]/snippetLen
nRep = if0_hot.shape[1]

#reshape the samples so that each is of length snippetLen
if0_hot = if0_hot.T.reshape(nSnippets*nRep, snippetLen) - offset; 
if1_hot = if1_hot.T.reshape(nSnippets*nRep, snippetLen) - offset; 
if0_cold = if0_cold.T.reshape(nSnippets*nRep, snippetLen) - offset; 
if1_cold = if1_cold.T.reshape(nSnippets*nRep, snippetLen) - offset; 

# get the power of each snippet
if0_hot_power = abs(np.fft.rfft(if0_hot))**2
if1_hot_power = abs(np.fft.rfft(if1_hot))**2
if0_cold_power = abs(np.fft.rfft(if0_cold))**2
if1_cold_power = abs(np.fft.rfft(if1_cold))**2

# take the average power spectrum
if0_hot_power_average = np.mean(if0_hot_power, axis=0)
if1_hot_power_average = np.mean(if1_hot_power, axis=0)
if0_cold_power_average = np.mean(if0_cold_power, axis=0)
if1_cold_power_average = np.mean(if1_cold_power, axis=0)

# get the frequencies associated with the power spectrum
freq = np.fft.fftfreq(snippetLen, d=1./samp_rate)
freq = freq[0:len(if0_hot_power_average)]; 
freq[-1] = -freq[-1]




# plot the ratios 
plt.figure(1)
plt.subplot(211)
plt.plot(freq,if0_hot_power_average/if0_cold_power_average)
plt.ylim(0, np.max(if0_hot_power_average/if0_cold_power_average)); 
plt.title('IF0 Ratio of Hot vs Cold Power')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Ratio')

plt.subplot(212)
plt.plot(freq,if1_hot_power_average/if1_cold_power_average)
plt.ylim(0, np.max(if1_hot_power_average/if1_cold_power_average)); 
plt.title('IF1 Ratio of Hot vs Cold Power')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Ratio')

plt.show()



