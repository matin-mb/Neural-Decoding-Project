import lindi
import numpy as np
import h5py

# Dandi archive session URL
URL = 'https://api.dandiarchive.org/api/assets/7c541451-875d-4398-b993-6e6a5aaa4910/download/'

def load_neurodata(url=URL):
    """Load neurodata from the given URL using lindi."""
    f = lindi.LindiH5pyFile.from_hdf5_file(url)
    X = f['/units']
    return {
        'electrodes': X['electrodes'],
        'electrodes_index': X['electrodes_index'],
        'id': X['id'],
        'spike_times': X['spike_times'],
        'spike_times_index': X['spike_times_index'],
        'waveforms': X['waveforms'],
        'waveforms_index': X['waveforms_index'],
        'waveforms_index_index': X['waveforms_index_index']
    }