
import numpy as np


class NormalizeMult():
    def __init__(self,dataset) :
        self.data =dataset

   
    def normalizemult(self):
        #normalize 
        data = np.array(self.data)
        normalize = np.arange(2*data.shape[1],dtype='float64')

        normalize = normalize.reshape(data.shape[1],2)
        # print(normalize.shape)
        for i in range(0,data.shape[1]):
            list = data[:,i]
            listlow,listhigh =  np.percentile(list, [0, 100])
            # print(i)
            normalize[i,0] = listlow
            normalize[i,1] = listhigh
            delta = listhigh - listlow
            if delta != 0:
                for j in range(0,data.shape[0]):
                    data[j,i]  =  (data[j,i] - listlow)/delta
        #np.save("normalize.npy",normalize)
        return  data,normalize