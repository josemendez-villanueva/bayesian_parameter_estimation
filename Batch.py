from netpyne import specs
from netpyne.batch import Batch

def batchnoise():
        # Create variable of type ordered dictionary (NetPyNE's customized version)
        params = specs.ODict()

        # fill in with parameters to explore and range of values (key has to coincide with a variable in simConfig)
        params['noise'] = [.2, .5, .9]

        # create Batch object with parameters to modify, and specifying files to use
        b = Batch(params=params, cfgFile='cfg.py', netParamsFile='netParams.py',)

        # Set output folder, grid method (all param combinations), and run configuration
        b.batchLabel = 'noise_analysis'
        b.saveFolder = 'noise_data'
        b.method = 'grid'
        b.runCfg = {'type': 'mpi_bulletin',
                                'script': 'init.py',
                                'skip': True}

        # Run batch simulations
        b.run()

# Main code
if __name__ == '__main__':
        batchnoise()