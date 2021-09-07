import torch 
import numpy as np
from netpyne import specs, sim
import time

import sim_complex
import matplotlib as mpl
import matplotlib.pyplot as plt
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.analysis.plot import pairplot

start_time = time.time()

#NetPyNE simulator
def simulator(params):
    params = np.asarray(params)

    sim_complex.netParams.stimTargetParams['bkg->all']['weight'] = params[0]
    sim_complex.netParams.connParams['E->all']['probability'] = params[1]
    sim_complex.netParams.connParams['E->all']['weight'] = params[2]
    sim_complex.netParams.connParams['I->E']['probability'] = params[3]
    sim_complex.netParams.connParams['I->E']['weight'] = params[4]

    #Runs the simulation with the above given parameters
    sim_complex.run()

    spiketimes = np.array(sim.simData['spkt'])
    plotraces = np.array(sim.simData['V_soma']['cell_0'])
    time = np.array(sim.simData['t'])
    hist = np.histogram(spiketimes, int(sim_complex.cfg.duration))[0] #replaces the need for a summary statistics class/fitness function class
    return dict(stats = hist, time = time, traces = plotraces)


#Simulation Wrapper: Uses only histogram statistics
def simulation_wrapper(params):
    obs = simulator(params)
    summstats = torch.as_tensor(obs['stats'])
    return summstats

#Ground Truth Parameters
baseline_param = np.array([0.1,0.1,0.005,0.4,0.001])
baseline_simulator = simulator(baseline_param)
observable_baseline_stats = torch.as_tensor(baseline_simulator['stats'])

#Prior distribution Setup
prior_min = np.array([0.05, 0.05, 0.001,0.2, 0.0005])
prior_max = np.array([0.2, 0.2, 0.007, 0.6, 0.002])

#Unifrom Distribution setup 
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), 
                                    high=torch.as_tensor(prior_max))

#Choose the option of running single-round or multi-round inference
inference_type = 'single'

if inference_type == 'single':
    posterior = infer(simulation_wrapper, prior, method='SNPE', 
                    num_simulations=5000, num_workers=48)
    samples = posterior.sample((10000,),
                                x = observable_baseline_stats)
    posterior_sample = posterior.sample((1,),
                                            x = observable_baseline_stats).numpy()

elif inference_type == 'multi':
    #Number of rounds that you want to run your inference
    num_rounds = 4
    #Driver for the multi-rounds inference
    for _ in range(num_rounds):
        posterior = infer(simulation_wrapper, prior, method='SNPE', 
                    num_simulations=2500, num_workers=48)
        prior = posterior.set_default_x(observable_baseline_stats)
        samples = posterior.sample((10000,), x = observable_baseline_stats)

    posterior_sample = posterior.sample((1,),
                        x = observable_baseline_stats).numpy()
else:
    print('Wrong Input for Inference Type')

# Plot Observed and Posterior
x = simulator(posterior_sample[0])
t = baseline_simulator['time']

print('Posterior Sample:', posterior_sample[0])

plt.figure(1, figsize=(16,14))

gs = mpl.gridspec.GridSpec(2,1,height_ratios=[4,1])
ax = plt.subplot(gs[0])

plt.plot(t, baseline_simulator['traces'], '-r' ,lw=2, label='observation')
plt.plot(t, x['traces'], '--', lw=2, label='posterior sample')
plt.xlabel('time (ms)')
plt.ylabel('voltage (mV)')
plt.title('Complex Network')

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.3, 1), 
          loc='upper right')
plt.legend()
plt.savefig('observation_vs_posterior.png')

plt.figure(2)
_ = analysis.pairplot(samples, limits=[[0.0,0.4],[0.0,0.4],[0.0,0.01],[0,1.0],[0.0,0.01]], 
                   figsize=(16,14))  

plt.legend()
plt.savefig('PairPlot.png')

print("Program took", time.time() - start_time, "seconds to run")

