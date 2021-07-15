import torch 
import numpy as np
from netpyne import sim
import sim_single
import matplotlib as mpl
import matplotlib.pyplot as plt
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer

#NetPyNE simulator
def simulator(params):
    params = np.asarray(params)

    sim_single.netParams.stimSourceParams['background']['noise'] = params
    """
    Way to simulate this without always running it with the displays?
     
    Need to see if there is an alternative to create simulations
    """
    # sim.createSimulateAnalyze(netParams = sim_single.netParams, simConfig = sim_single.cfg)
    sim_single.run()

    spiketimes = np.array(sim.simData['spkt'])
    plotraces = np.array(sim.simData['V_soma']['cell_0'])
    time = np.array(sim.simData['t'])
    hist = np.histogram(spiketimes, int(sim_single.cfg.duration))[0] #replaces the need for a summary statistics class/fitness function class
    return dict(stats = hist, time = time, traces = plotraces)


def simulation_wrapper(params):
    obs = simulator(params)
    summstats = torch.as_tensor(obs['stats'])
    return summstats

#Ground Truth Parameters
baseline_param = np.array([.8])
baseline_simulator = simulator(baseline_param)
observable_baseline_stats = torch.as_tensor(baseline_simulator['stats'])


#Inference

prior_min = np.array([0])
prior_max = np.array([2])

prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), 
                                    high=torch.as_tensor(prior_max))

posterior = infer(simulation_wrapper, prior, method='SNPE', 
                  num_simulations=1000, num_workers=4)


samples = posterior.sample((10000,),
                            x = observable_baseline_stats)


posterior_sample = posterior.sample((1,),
                                        x = observable_baseline_stats).numpy()



#Plot Observed and Posterior

fig = plt.figure(figsize=(12,10))

gs = mpl.gridspec.GridSpec(2,1,height_ratios=[4,1])
ax = plt.subplot(gs[0])

#x is simulation of Posterior
x = simulator(posterior_sample)
t = baseline_simulator['time']

plt.plot(t, baseline_simulator['traces'], '-r' ,lw=2, label='observation')
plt.plot(t, x['traces'], '--', lw=2, label='posterior sample')

plt.xlabel('time (ms)')
plt.ylabel('voltage (mV)')

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.3, 1), 
          loc='upper right')

plt.legend()
plt.show()
