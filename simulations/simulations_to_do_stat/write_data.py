import numpy as np
import matplotlib.pyplot as plt
import glob

npyfiles = []
for file in glob.glob("*.npy"):
    npyfiles.append(file)
print(npyfiles[1:])
list_data = []
for index, file in enumerate(npyfiles):
    list_data.append(np.load(file))


data_array = np.average(a=list_data,axis=0)


print("\nData summary:\n", data_array)
print("\nData shape:\n", data_array.shape)

data_array_last_rewards = data_array[:,-1]
#data_array_last_rewards = np.array([max(0,data_array_last_rewards[i]) for i in range(len(data_array_last_rewards))])
#data_array_last_rewards = data_array_last_rewards[0:100]

print('average start:{}'.format(np.average(data_array_last_rewards[0:100])))
print('average last:{}'.format(np.average(data_array_last_rewards[-100:])))

print('standard derivation start:{}'.format(np.std(data_array_last_rewards[0:100])))
print('standard derivation end:{}'.format(np.std(data_array_last_rewards[-100:])))
print('Best reward: {}'.format(max(data_array_last_rewards)))

#print("\nData summary:\n", data_array_last_rewards)
print("\nData shape:\n", data_array_last_rewards.shape)

#"""
xaxis = np.arange(data_array.shape[0])
plt.show()
#plt.title("Evolution of the local reward during training for a 4-qubit system.")
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.scatter(xaxis,data_array_last_rewards, marker = '+', c = 'g')
#plt.plot(xaxis,data_array_last_rewards, marker = '+', c = 'g')


plt.savefig('local_rewards.png')

"""
data_average = np.zeros(data_array_last_rewards.shape[0]-100)
for idx, value in enumerate(data_array[0:-100]):
    data_average[idx] = np.average(data_array_last_rewards[idx:idx+100])


#data_average = np.zeros(int(data_array_last_rewards.shape[0]/100))
#for idx in np.arange(0,int(data_array.shape[0]/100-1)):
#    data_average[idx] = np.average(data_array_last_rewards[idx*100:(idx+1)*100])



xaxis_ave = np.arange(data_average.shape[0])
plt.show()
plt.title("Evolution of the average local reward during training for a N-qubit system.")
plt.xlabel('Episode')
plt.ylabel('average Rewards over 100')
#plt.scatter(xaxis,data_average, marker = '+', c = 'g')

plt.plot(xaxis_ave,data_average, marker = '+', c = 'g')
plt.savefig('local_rewards_ave.png')
"""