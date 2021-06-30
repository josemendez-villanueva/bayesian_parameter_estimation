from netpyne import specs
from netpyne.batch import Batch

def batchnoise():
     
        params = specs.ODict()

        params['noise'] = [.2, .5, .9] #noise paramters that are given

        



 
        b = Batch(params=params, cfgFile='cfg.py', netParamsFile='netParams.py',)


        b.batchLabel = 'noise_analysis'
        b.saveFolder = 'noise_data'
        b.method = 'delfi'
        b.runCfg = {'type': 'mpi_bulletin',
                                'script': 'init.py',
                                'skip': True}


        b.run()

# Main code
if __name__ == '__main__':
        batchnoise()