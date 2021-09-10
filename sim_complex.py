from netpyne import specs, sim, analysis
from netpyne.specs import Dict
import numpy as np

netParams = specs.NetParams()
cfg = specs.SimConfig()


param = np.array([0,0,0,0,0])

netParams.sizeX = 100 # x-dimension (horizontal length) size in um
netParams.sizeY = 1000 # y-dimension (vertical height or cortical depth) size in um
netParams.sizeZ = 100 # z-dimension (horizontal length) size in um
netParams.propVelocity = 100.0 # propagation velocity (um/ms)
netParams.probLengthConst = 150 # length constant for conncfg. probability (um)


netParams.cellParams['E'] = {
    'secs': {'soma':
            {'geom': {'diam': 15, 'L': 14, 'Ra': 120.0},
            'mechs': {'hh': {'gnabar': 0.13, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}}}}}



netParams.cellParams['I'] = {
    'secs': {'soma':
            {'geom': {'diam': 10.0, 'L': 9.0, 'Ra': 110.0},
            'mechs': {'hh': {'gnabar': 0.11, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}}}}}

#Population parameters
pop = 50
netParams.popParams['E2'] = {'cellType': 'E', 'numCells': pop, 'yRange': [100,300], 'cellModel': 'HH'}
netParams.popParams['I2'] = {'cellType': 'I', 'numCells': pop, 'yRange': [100,300], 'cellModel': 'HH'}
netParams.popParams['E4'] = {'cellType': 'E', 'numCells': pop, 'yRange': [300,600], 'cellModel': 'HH'}
netParams.popParams['I4'] = {'cellType': 'I', 'numCells': pop, 'yRange': [300,600], 'cellModel': 'HH'}
netParams.popParams['E5'] = {'cellType': 'E', 'numCells': pop, 'ynormRange': [0.6,1.0], 'cellModel': 'HH'}
netParams.popParams['I5'] = {'cellType': 'I', 'numCells': pop, 'ynormRange': [0.6,1.0], 'cellModel': 'HH'}


# Synaptic mechanism parameters
netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': 0.8, 'tau2': 5.3, 'e': 0}  # NMDA synaptic mechanism
netParams.synMechParams['inh'] = {'mod': 'Exp2Syn', 'tau1': 0.6, 'tau2': 8.5, 'e': -75}  # GABA synaptic mechanism


# Stimulation parameters
netParams.stimSourceParams['bkg'] = {'type': 'NetStim', 'rate': 20, 'noise': 0.3}
netParams.stimTargetParams['bkg->all'] = {'source': 'bkg', 'conds': {'cellType': ['E','I']}, 'weight': param[0], 'delay': 'max(1, normal(5,2))', 'synMech': 'exc'}


## Cell connectivity rules
netParams.connParams['E->all'] = {
	  'preConds': {'cellType': 'E'}, 'postConds': {'y': [100,1000]},  #  E -> all (100-1000 um)
	  'probability': param[1],                  # probability of connection
	  'weight': param[2],         # synaptic weight
	  'delay': 0.8,      # transmission delay (ms)
	  'synMech': 'exc'}                     # synaptic mechanism

netParams.connParams['I->E'] = {
	  'preConds': {'cellType': 'I'}, 'postConds': {'pop': ['E2','E4','E5']},       #  I -> E
	  'probability': param[3],   # probability of connection
	  'weight': param[4],                                      # synaptic weight
	  'delay': 0.8,                      # transmission delay (ms)
	  'synMech': 'inh'}     


# Create the recoding of data/simulation/plotting

cfg.duration = 1*1e3        # Duration of the simulation, in ms
cfg.dt = 0.025              # Internal integration timestep to use
cfg.verbose = False         # Show detailed messages
cfg.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
cfg.recordStep = 0.1        # Step size in ms to save data (eg. V traces, LFP, etc      
cfg.saveFolder = 'output_folder'
cfg.filename = 'output'
cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net']
cfg.saveJson = False
cfg.printPopAvgRates = False


cfg.analysis['plotTraces'] = {'include': [0], 'saveFig': False}  # Plot recorded traces for this list of cells

def run():
    sim.createSimulateAnalyze(netParams = netParams, simConfig = cfg)
