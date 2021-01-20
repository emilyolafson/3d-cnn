import pickle
import numpy
import scipy.io as sio


for i in range(1,23):
    with open('/Users/emilyolafson/GIT/fc-from-sc/SUB' + str(i) + '_lesion1mm_nemo_output_chacoconn_fs86subj_mean.pkl', 'r+b') as e:

        data2 = pickle.load(e)
        sio.savemat('SUB' + str(i) +'_lesion_1mmMNI_fs86subj_mean_chacoconn.mat', {'SUB'+str(i)+'chacoconn':data2})