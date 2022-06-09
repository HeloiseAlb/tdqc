import numpy as np
import matplotlib.pyplot as plt

data_array = np.load('rewards.npy')

print("\nData summary:\n", data_array)
print("\nData shape:\n", data_array.shape)

data_array_last_rewards = data_array[:,-1]
data_array_last_rewards = data_array_last_rewards[0:100]

print('average:{}'.format(sum(data_array_last_rewards)/data_array_last_rewards.shape[0]))
print('standard derivation:{}'.format(np.std(data_array_last_rewards)))
print("\nData summary:\n", data_array_last_rewards)
print("\nData shape:\n", data_array_last_rewards.shape)

xaxis = np.arange(data_array_last_rewards.shape[0])
plt.show()
plt.title("Evolution of the local reward during training for a 4-qubit system.")
plt.xlabel('Episode')
plt.ylabel('Rewards')
#plt.scatter(xaxis,data_array_last_rewards, marker = '+', c = 'g')

plt.plot(xaxis,data_array_last_rewards, marker = '+', c = 'g')

plt.savefig('local_rewards.png')

