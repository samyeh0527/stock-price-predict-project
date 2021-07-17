import numpy as np

class FNormalizeMult():
    def __init__(self,dataset,normalize) :
        self.data = dataset
        self.normalize= normalize
    def FNormalizeMult_(self):
        data = np.array(self.data)
        for i in  range(0,data.shape[1]):
            listlow =  self.normalize[i,0]
            listhigh = self.normalize[i,1]
            delta = listhigh - listlow
            if delta != 0:
                #第j行
                for j in range(0,data.shape[0]):
                    data[j,i]  =  data[j,i]*delta + listlow

        return data


if __name__ == '__main__':
    FNormalizeMult()