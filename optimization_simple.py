import torch 
import numpy as np
from netpyne import sim
import sim_simple

import matplotlib as mpl
import matplotlib.pyplot as plt
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import time


start_time = time.time()

#NetPyNE simulator
def simulator(params):
    params = np.asarray(params)

    # sim_simple.netParams.stimSourceParams['background']['noise'] = params[0]
    sim_simple.netParams.connParams['P1 -> P2']['probability'] = params[0]
    sim_simple.netParams.connParams['P1 -> P2']['weight'] = params[1]
    sim_simple.netParams.connParams['P1 -> P2']['delay'] =params[2]

    """
    Way to simulate this without always running it with the displays?
     
    Need to see if there is an alternative to create simulations
    """
    # sim.createSimulateAnalyze(netParams = sim_simple.netParams, simConfig = sim_simple.cfg)
    sim_simple.run()

    spiketimes = np.array(sim.simData['spkt'])
    plotraces = np.array(sim.simData['V_soma']['cell_0'])
    time = np.array(sim.simData['t'])
    hist = np.histogram(spiketimes, int(sim_simple.cfg.duration))[0] #replaces the need for a summary statistics class/fitness function class
    return dict(stats = hist, time = time, traces = plotraces)


def simulation_wrapper(params):
    obs = simulator(params)
    summstats = torch.as_tensor(obs['stats'])
    return summstats

#Ground Truth Parameters
baseline_param = np.array([0.2, 0.025, 2])
baseline_simulator = simulator(baseline_param)
observable_baseline_stats = torch.as_tensor(baseline_simulator['stats'])


#Inference

prior_min = np.array([0.01, 0.001, 1])
prior_max = np.array([0.5, 0.1, 20])

#Unifrom Distribution setup 
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), 
                                    high=torch.as_tensor(prior_max))

#Choose the option of running single-round or multi-round inference
inference_type = 'single'

if inference_type == 'single':
    posterior = infer(simulation_wrapper, prior, method='SNPE', 
                    num_simulations=2500, num_workers=8)
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
                    num_simulations=2500, num_workers=8)
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
_ = analysis.pairplot(samples, limits=[[0.0,0.5],[0.0,0.12],[0,21]], 
                   figsize=(16,14))  

plt.legend()
plt.savefig('PairPlot.png')

print("Program took", time.time() - start_time, "seconds to run")
#Next step is to plot either the distribution or the likelihood estimator
