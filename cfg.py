from netpyne import specs

cfg = specs.SimConfig()


# Create the recoding of data/simulation/plotting

cfg.duration = 1*1e3        # Duration of the simulation, in ms
cfg.dt = 0.025              # Internal integration timestep to use
cfg.verbose = False         # Show detailed messages
cfg.recordTraces = {'V_soma':{'sec':'soma','loc':0.5,'var':'v'}}  # Dict with traces to record
cfg.recordStep = 0.1        # Step size in ms to save data (eg. V traces, LFP, etc)
cfg.filename = 'output'       # Set file output name
cfg.saveJson = True
cfg.printPopAvgRates = True

cfg.analysis['plotTraces'] = {'include': [0], 'saveFig': True}  # Plot recorded traces for this list of cells

cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net']

cfg.noise = 0.8