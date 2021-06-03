from netpyne import specs, sim

try:
    from __main__ import cfg  
except:
    from cfg import cfg  


netParams = specs.NetParams()

# This will create the basic cell model that we will be using: Creating both the soma and a dendritic section
netParams.cellParams['PYR'] = {
    'secs': {'soma':
            {'geom': {'diam': 18.8, 'L': 18.8, 'Ra': 123.0},
            'mechs': {'hh': {'gnabar': 0.12, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}}}},
            'dend':
            {'geom': {'diam': 5.0, 'L': 150.0, 'Ra': 150.0, 'cm': 1},
            'mechs': {'pas': {'g': 0.0000357, 'e': 0}}}}

# The following will be creating a population of one pyramidal cell

netParams.popParams['P1'] = {'cellType': 'PYR', 'numCells': 1}

# Create synapses

netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': 0.1, 'tau2': 5.0, 'e': 0}

# Now to create Background Noise and the target for it

netParams.stimSourceParams['background'] = {'type': 'NetStim', 'rate': 10, 'noise': cfg.noise}
netParams.stimTargetParams['background->Pop'] = {'source': 'background', 'conds': {'cellType': 'PYR'}, 'weight': 0.03, 'delay': 10, 'synMech': 'exc'}


