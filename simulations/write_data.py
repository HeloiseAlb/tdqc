#%%
import numpy as np
import matplotlib.pyplot as plt

data_array = np.load('rewardsNoPD_N6episode50000simulations0t_final0.8ok.npy')

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
plt.scatter(xaxis, data_array_last_rewards, marker = '+', c = 'g')
#plt.plot(xaxis,data_array_last_rewards, marker = '+', c = 'g')


#plt.savefig('local_rewards.png')
#"""
data_average = np.zeros(data_array_last_rewards.shape[0]-100)
for idx, value in enumerate(data_array[0:-100]):
    data_average[idx] = np.average(data_array_last_rewards[idx:idx+100])


#data_average = np.zeros(int(data_array_last_rewards.shape[0]/100))
#for idx in np.arange(0,int(data_array.shape[0]/100-1)):
#    data_average[idx] = np.average(data_array_last_rewards[idx*100:(idx+1)*100])



xaxis_ave = np.arange(data_average.shape[0])
plt.show()
plt.title("Evolution of the local reward (average over 100) during training for a 6-qubit system.")
plt.xlabel('Episode')
plt.ylabel('average local reward over 100 episodes')
#plt.scatter(xaxis,data_average, marker = '+', c = 'g')

plt.plot(xaxis_ave,data_average, marker = '+', c = 'g')
#plt.savefig('local_rewards_ave.png')

#"""
# %%
fidelities_several_circuits = np.array([0.6721171862867633,0.4901307435777814	, 0.22560642664465058	,0.2927422854325783	,0.17899370307109774	,0.14853528033044516,0.08776751294322593	,0.22219540680442373,0.3136491509103273,0.30025882198287634])
fidelities_ed_dql = np.array([0.6721171862867633, 0.55960608,0.56282425,0.52890962,0.43585315,0.57339609,0.52625495,0.54413414])
fidelities_several_circuits_t2 = np.array([0.5596060534411742,0.27667333738509203,0.23905020869816584,0.18610511855834655,0.17412193921179597])
xaxis = np.arange(1,fidelities_several_circuits.shape[0]+1)
xaxis2 = np.arange(1,fidelities_ed_dql.shape[0]+1)
xaxis3 = np.arange(2,fidelities_ed_dql.shape[0]+2,2)

plt.show()
#plt.title("Evolution of the local reward during training for a 4-qubit system.")
plt.xlabel('Times')
plt.ylabel('Fidelities')
plt.plot(xaxis,fidelities_several_circuits, marker = '+', c = 'g')
plt.plot(xaxis2,fidelities_ed_dql, marker = '+', c = 'r')
plt.plot(xaxis3,fidelities_several_circuits_t2, marker = '+', c = 'c')

#plt.plot(xaxis,data_array_last_rewards, marker = '+', c = 'g')

# %%
fidelities_several_circuits = np.array([0.6315579363239511, 0.4272137024427569, 0.14044484965132753, 0.24374324905223155, 0.17840050800116813, 0.1305493311671747, 0.05344007715917143, 0.2441355281806593, 0.32545358828850235, 0.32738204742350663])


xaxis = np.arange(fidelities_several_circuits.shape[0])
plt.show()
#plt.title("Evolution of the local reward during training for a 4-qubit system.")
plt.xlabel('Times')
plt.ylabel('Fidelities')
plt.plot(xaxis,fidelities_several_circuits, marker = '+', c = 'g')
#plt.plot(xaxis,data_array_last_rewards, marker = '+', c = 'g')

# %%

# %%
# Plots of the fidelity when applying several time the circuit obtained from DQL 
# and the circuit obtained by Trotterization. The block forming the block unit 
# have been obtained for the simulation of t_final = 0.06. Applying n times the 
# circuit should provide the final state for the final time n*t_final. 

# Following the fidelities for positiveDefinite = True
fidelities_several_circuits_dql = np.array([0.7128275632858276, 0.5116266012191772, 0.33018091320991516, 0.13852520287036896, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ])
fidelities_circuit_same_depth_trotter = np.array([0.1927478313446045, 0.0, 0.0, 0.0, 0.0, 0.017091745510697365, 0.37677833437919617, 0.42947739362716675, 0.192281112074852, 0.0])

# Following the fidelities for positiveDefinite = False
fidelities_several_circuits_dql = np.array([0.7128275632858276, 0.5116266012191772, 0.33018091320991516, 0.13852520287036896, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ])
fidelities_circuit_same_depth_trotter2 = np.array([0.9835176467895508, 0.9708316326141357, 0.9598742723464966, 0.9507444500923157, 0.9439265131950378, 0.9398025274276733, 0.9382265210151672, 0.9383816123008728, 0.9398238062858582, 0.9428160190582275])

# 

#fidelities_several_circuits_t2 = np.array([0.5596060534411742,0.27667333738509203,0.23905020869816584,0.18610511855834655,0.17412193921179597])
xaxis = 0.06*np.arange(1,fidelities_several_circuits_dql.shape[0]+1)
xaxis2 = 0.06*np.arange(1,fidelities_circuit_same_depth_trotter.shape[0]+1)
#xaxis3 = np.arange(2,fidelities_ed_dql.shape[0]+2,2)

plt.show()
#plt.title("Evolution of the local reward during training for a 4-qubit system.")
plt.xlabel('Times')
plt.ylabel('Local reward')
plt.plot(xaxis,fidelities_several_circuits_dql, marker = '+', c = 'g', label="dql")
plt.plot(xaxis2,fidelities_circuit_same_depth_trotter2, marker = '+', c = 'r', label="Trotter")
#plt.plot(xaxis3,fidelities_several_circuits_t2, marker = '+', c = 'c')
plt.legend(loc="upper right")
plt.xticks(xaxis)
#plt.plot(xaxis,data_array_last_rewards, marker = '+', c = 'g')

# %%
