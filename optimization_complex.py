import torch 
import numpy as np
from netpyne import specs, sim

import sim_complex
import matplotlib as mpl
import matplotlib.pyplot as plt
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.analysis.plot import pairplot

#NetPyNE simulator

def simulator(params):
    params = np.asarray(params)

    sim_complex.netParams.stimTargetParams['bkg->all']['weight'] = params[0]
    sim_complex.netParams.connParams['E->all']['probability'] = params[1]
    sim_complex.netParams.connParams['E->all']['weight'] = params[2]
    sim_complex.netParams.connParams['I->E']['probability'] = params[3]
    sim_complex.netParams.connParams['I->E']['weight'] = params[4]

    sim_complex.run()

    spiketimes = np.array(sim.simData['spkt'])
    plotraces = np.array(sim.simData['V_soma']['cell_0'])
    time = np.array(sim.simData['t'])
    hist = np.histogram(spiketimes, int(sim_single_cell.cfg.duration))[0] #replaces the need for a summary statistics class/fitness function class
    return dict(stats = hist, time = time, traces = plotraces)



def simulation_wrapper(params):
    obs = simulator(params)
    summstats = torch.as_tensor(obs['stats'])
    return summstats

#Ground Truth Parameters
baseline_param = np.array([0.1,0.1,0.005,0.4,0.001])
baseline_simulator = simulator(baseline_param)
observable_baseline_stats = torch.as_tensor(baseline_simulator['stats'])


#Inference
prior_min = np.array([0.05, 0.05, 0.001,0.2, 0.0005])
prior_max = np.array([0.2, 0.2, 0.007, 0.6, 0.002])

prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), 
                                    high=torch.as_tensor(prior_max))

posterior = infer(simulation_wrapper, prior, method='SNRE', 
                  num_simulations=500, num_workers=16)


samples = posterior.sample((10000,),
                            x = observable_baseline_stats)


posterior_sample = posterior.sample((1,),
                                        x = observable_baseline_stats).numpy()




# Plot Observed and Posterior
x = simulator(posterior_sample[0])
t = baseline_simulator['time']
print(posterior_sample[0])

plt.figure(1, figsize=(12,10))

gs = mpl.gridspec.GridSpec(2,1,height_ratios=[4,1])
ax = plt.subplot(gs[0])

#x is simulation of Posterior

plt.plot(t, baseline_simulator['traces'], '-r' ,lw=2, label='observation')
plt.plot(t, x['traces'], '--', lw=2, label='posterior sample')
plt.xlabel('time (ms)')
plt.ylabel('voltage (mV)')
plt.title('Network: 60 Neurons of 6 Populations')

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.3, 1), 
          loc='upper right')


plt.show()
