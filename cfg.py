from netpyne import specs

cfg = specs.SimConfig()


# Create the recoding of data/simulation/plotting

cfg.duration = 1*1e3        
cfg.dt = 0.025              
cfg.verbose = False         
cfg.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}} 
cfg.recordStep = 0.1        
cfg.filename = 'output_noise'      
cfg.saveDataInclude = ['simData']
cfg.saveJson = True
cfg.printPopAvgRates = True

cfg.analysis['plotTraces'] = {'include': [0], 'saveFig': True}  
cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net']

cfg.noise = 0.8