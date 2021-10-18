from netpyne.network.pop import Pop
import torch 
import numpy as np
from netpyne import specs, sim
import time
import json

import sim_complex
import matplotlib as mpl
import matplotlib.pyplot as plt
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.analysis.plot import pairplot


from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import analysis as analysis

start_time = time.time()

#NetPyNE simulator

b = np.linspace(sim_complex.pop, sim_complex.pop*len(sim_complex.netParams.popParams), len(sim_complex.netParams.popParams))


def simulator(params):
    params = np.asarray(params)

    #The tuning parameters will be the weight parameters
    sim_complex.netParams.stimTargetParams['bkg->all']['weight'] = params[0]
    sim_complex.netParams.connParams['E->all']['probability'] = params[1]
    sim_complex.netParams.connParams['E->all']['weight'] = params[2]
    sim_complex.netParams.connParams['I->E']['probability'] = params[3]
    sim_complex.netParams.connParams['I->E']['weight'] = params[4]
    sim_complex.netParams.connParams['E->I']['weight'] = params[5]

    #Runs the simulation with the above given parameters
    sim_complex.run()

    simdata = sim.allSimData['popRates']
    poprates = np.zeros(len(sim_complex.netParams.popParams))
    j = 0
    for i in simdata.values():
        poprates[j] = (i)
        j += 1

    plotraces = np.array(sim.simData['V_soma']['cell_0'])
    time = np.array(sim.simData['t'])
    hist = np.histogram(poprates, b)[0] #replaces the need for a summary statistics class/fitness function class
    return dict(stats = hist, time = time, pop = poprates, traces = plotraces)


#Simulation Wrapper: Uses only histogram statistics
def simulation_wrapper(params):
    obs = simulator(params)
    summstats = torch.as_tensor(obs['stats'])
    return summstats

#Target Values 
pop_target = [20, 3, 20, 3, 20, 3]
observable_baseline_stats = torch.as_tensor(np.histogram(pop_target, b)[0])

#Results from SNLE with only 1000 simulations
# Posterior Sample Param: [0.18673387 0.07158194 0.00334429 0.5259729  0.00097797 0.00197567]
# Pop Rate Estimates: [19.6  4.2 20.6  3.  21.2  3.6]

#Results from SNRE with only 1000 simulations
# Posterior Sample Param: [0.1994602  0.06239155 0.00471359 0.24544932 0.00085426 0.00172087]
# Pop Rate Estimates: [20.2  5.  23.2  3.6 21.6  3.6]

#Results from SNPE with only 1000 simulations
# Posterior Sample Param: [0.1664557  0.18847144 0.00292594 0.40350556 0.00132117 0.00158131]
# Pop Rate Estimates: [21.4  3.6 22.4  3.6 22.8  3.8]



#Prior distribution Setup
prior_min = np.array([0.05, 0.05, 0.001,0.2, 0.0005, 0.0005])
prior_max = np.array([0.5, 0.5, 0.1, 0.8, 0.2, 0.2])

#Unifrom Distribution setup 
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), 
                                    high=torch.as_tensor(prior_max))

#Choose the option of running single-round or multi-round inference
inference_type = 'single'

if inference_type == 'single':
    posterior = infer(simulation_wrapper, prior, method='SNLE', 
                    num_simulations=15000, num_workers=8)
    samples = posterior.sample((10000,),
                                x = observable_baseline_stats)
    posterior_sample = posterior.sample((1,),
                                            x = observable_baseline_stats).numpy()

elif inference_type == 'multi':
    #Number of rounds that you want to run your inference
    num_rounds = 2
    #Driver for the multi-rounds inference
    for _ in range(num_rounds):
        posterior = infer(simulation_wrapper, prior, method='SNPE', 
                    num_simulations=15000, num_workers=56)
        prior = posterior.set_default_x(observable_baseline_stats)
        samples = posterior.sample((10000,), x = observable_baseline_stats)

    posterior_sample = posterior.sample((1,),
                        x = observable_baseline_stats).numpy()

else:
    print('Wrong Input for Inference Type')

# Plot Observed and Posterior

#Gives the optimized paramters Here

op_param = posterior_sample[0]

x = simulator(op_param)
t = x['time']

#How to compare the poprates plot traces to the estimated one? Since we gave target one we cannot really do this since we do not have the target parameters


print('Posterior Sample Param:', op_param)
print('Pop Rate Estimates:', x['pop'])
# plt.figure(1, figsize=(16,14))

# gs = mpl.gridspec.GridSpec(2,1,height_ratios=[4,1])
# ax = plt.subplot(gs[0])

# plt.plot(t, x['traces'], '--', lw=2, label='posterior sample')
# plt.xlabel('time (ms)')
# plt.ylabel('voltage (mV)')
# plt.title('Complex Network')

# ax = plt.gca()
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.3, 1), 
#           loc='upper right')
# plt.legend()
# plt.savefig('observation_vs_posterior.png')

# plt.figure(2)
# _ = analysis.pairplot(samples, limits=[[0.0,0.4],[0.0,0.4],[0.0,0.01],[0,1.0],[0.0,0.01], [0.0,0.01]], 
#                    figsize=(16,14))  

# plt.legend()
# plt.savefig('PairPlot.png')

# print("Program took", time.time() - start_time, "seconds to run")