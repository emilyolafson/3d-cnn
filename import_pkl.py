import pickle
import numpy
import scipy.io as sio

for i in range(1,23):
    with open('/Users/emilyolafson/Documents/Thesis/colossus_syncs/SUB1_23_data/NeMo_SUB1_23/shen_regional/SUB' + str(i) + '_lesion_1mmMNI_nemo_output_chacovol_shen268_mean.pkl', 'r+b') as f:
        data = pickle.load(f)
        numpy.savetxt('SUB' + str(i) + '_lesion_1mmMNI_shen268_mean_chacovol.csv', data, delimiter=",")
    with open('/Users/emilyolafson/Documents/Thesis/colossus_syncs/SUB1_23_data/NeMo_SUB1_23/shen_regional/SUB' + str(i) + '_lesion_1mmMNI_nemo_output_chacoconn_shen268_mean.pkl', 'r+b') as e:
        data2 = pickle.load(e)
        print(data2.shape)
        sio.savemat('SUB' + str(i) +'_lesion_1mmMNI_shen268_mean_chacoconn.mat', {'SUB'+str(i)+'chacoconn':data2})
