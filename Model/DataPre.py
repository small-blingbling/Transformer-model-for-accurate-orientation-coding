import hdf5storage
import numpy as np

def Data_Pre(path):
    order_fig_path = f'{path}/order_fig_1st_100.mat'
    resp_avg_path = f'{path}/G4_RespAvg.mat'

    data_FigAll = hdf5storage.loadmat(order_fig_path)
    target = data_FigAll['order_fig_1st_100']
    target = np.expand_dims(target,axis=1)

    data_Response = hdf5storage.loadmat(resp_avg_path)
    source = data_Response['G4_RespAvg']

    s  = (source-np.min(source))/(np.max(source)-np.min(source))


    return s*10, target