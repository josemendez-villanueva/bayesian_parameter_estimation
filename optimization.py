from os import stat
from delfi.simulator.BaseSimulator import BaseSimulator
import numpy as np
from netpyne import specs, sim
from numpy.core.arrayprint import StructuredVoidFormat
from numpy.lib.function_base import sinc
import sim_single_cell


#Might need to get the spikes or etc....
def simulator(params):
    #parameters that you want to test
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    params = np.array(params)
    num_params = len(params) # changed this from params[:,0] since of the way that params is set up.....

    #for loop to connect the above paramters into your simulation and instantiate it

    #instantiate first simulation which is a guarantee to have your n set up to give spike time results back

    sim_single_cell.netParams.stimSourceParams['background']['noise'] = params[0]
    sim_single_cell.cfg.filename = 'output' + str(0)    
    sim.createSimulateAnalyze(netParams = sim_single_cell.netParams, simConfig = sim_single_cell.cfg) 

    n = len(sim.simData['spkt'])
    sim_samples = np.zeros((num_params, n))

    # one issue that can arise is if there are different n's per simulation.... will focus on that later on...
    # Have to re-run for zero in order to put into the dictionary until I can find where to pull simData from with netPyNE functionality
    # Do not want to have to open a json file and search through that is why I am just doing it from the simulations that are generated in file
    for i in range(0, len(params)):
        sim_single_cell.netParams.stimSourceParams['background']['noise'] = params[i]
        sim_single_cell.cfg.filename = 'output' + str(i)    
        sim.createSimulateAnalyze(netParams = sim_single_cell.netParams, simConfig = sim_single_cell.cfg)  
        for j in range(n):   
            spikes = float((sim.simData['spkt'][j]))
            sim_samples[i][j] = spikes
            #Might need to return a histogram in order to make this run better or just return the time steps as well 

    print(sim_samples)
    return sim_samples
  

# params = [[.2],[.3],[.9]]
# simulator(params)



# this class will be using the delfi wrapper

class single_cell(BaseSimulator):

    def __init__(self, seed = None):

        dim_param = 3 #this is the dimension of the params np array
        super().__init__(dim_param = dim_param, seed = seed)
        self.netpynesimulation = simulator

    def single_parameter(self, params):
        params = np.asarray(params)
        assert params.ndim == 1 #checking dimension is only taking in one set of parameters to be evaulated
        states = self.netpynesimulation(params)

        return {"data" : states}

# #From the above need to make sure that states is giving the  correct data that we need ot else will have to manually export
import delfi.distribution as dd

seed_p = 1

# #The following will be the Prior Dist
# #Using uniform distribution for one variable
prior_min = [.2]
prior_max = [5]

prior = dd.Uniform(lower = prior_min, upper = prior_max, seed = seed_p)

# #Don't need to create a summary statistics class due to netpyne recording number of spikes, times, and the rest should be easily found if needed
# #Now to create teh generator class

import delfi.generator as dg
from delfi.summarystats import Identity

m = single_cell()
s = Identity()

# #Before implementing the below, you need to have your simulator, prior, and summary statistic done

g = dg.Default(model = m, prior = prior, summary = s)

# #defining a baseline and its statistics... will be needed later

t_params = np.array([.4]) #Denoting this one as the "real" parameter
t_param_name = ['noise']

obs = m.single_parameter(t_params)
obs_stats = s.calc([obs])

# #Hyper-parameters/inference set up here

# #seed for inference
seed_inf = 1

# #summary statistics parameters????

# #training

n_processes = 10 

pilot_samples = 10

# #training schedules
n_train = 10
n_rounds = 4

# #fitting schedule
minibatch = 5
epochs = 10
val_frac = 0.05

# #network setup

n_hiddens = [50,50]

# #convenience 

prior_norm = True

# #MAF parameters
density = 'maf'
n_mades = 5

import delfi.inference as infer
# inference object

res = infer.SNPEC(g,
                obs=obs_stats,
                n_hiddens=n_hiddens,
                seed=seed_inf,
                pilot_samples=pilot_samples,
                n_mades=n_mades,
                prior_norm=prior_norm,
                density=density)



loglik, _, posterior = res.run(
                    n_train=n_train,
                    n_rounds=n_rounds,
                    minibatch=minibatch,
                    epochs=epochs,
                    silent_fail=False,
                    proposal='prior',
                    val_frac=val_frac,
                    verbose=True,)





