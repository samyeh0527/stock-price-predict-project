import matplotlib.pyplot as plt



class visualization():
    def __init__(self,name,originalvalue,predict) :
        self.originalvalue = originalvalue
        self.predict = predict
        self.name = name

    def visualization(self):
        plt.plot(self.originalvalue)       
        plt.plot(self.predict,'r--')
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.savefig(self.name)
        plt.show()


