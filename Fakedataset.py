#fack dataset 
from numpy.core.numeric import False_
import LoadDataset
import statistics


class Fakedataset():
    def __init__(self,dataset)  :
        self.dataset=dataset
        self.Add = None
        self.Upper_lower_limit = dataset[-1]
        self.interval=1



    def Add_fake_oneday_mean(self):
        #AVG
        Statistics_mean = statistics.mean(self.dataset[-5:])
        if Statistics_mean > (self.Upper_lower_limit *1.1) or Statistics_mean < (self.Upper_lower_limit *    0.9) :
            print("Upper_lower_limit = ",self.Upper_lower_limit ,"\nNow Value is %.2f" %Statistics_mean)
            return False
        else:
            print("original dataset " ,self.dataset)
            self.dataset.append(Statistics_mean)
            print("\nAfter modification dataset " ,self.dataset)
            return Statistics_mean


    def Add_fake_oneday_median(self):
        #中位數
        Statistics_median = statistics.median(self.dataset[-2:])
        if Statistics_median > (self.Upper_lower_limit *1.1) or Statistics_median < (self.Upper_lower_limit *    0.9) :
            print("Upper_lower_limit = ",self.Upper_lower_limit ,"\nNow Value is %.2f" %Statistics_median)
        else:
            print("original dataset " ,self.dataset)
            self.dataset.append(Statistics_median)
            print("\nAfter modification dataset " ,self.dataset)
            return Statistics_median


    def Add_fake_oneday_median_grouped(self):
        #中位數
        Statistics_median_grouped = statistics.median_grouped(self.dataset[-2:],self.interval)
        if Statistics_median_grouped > (self.Upper_lower_limit * 1.1) or Statistics_median_grouped < (self.Upper_lower_limit * 0.9) :
            print("Upper_lower_limit = ",self.Upper_lower_limit ,"\nNow Value is %.2f" %Statistics_median_grouped)
        else:
            print("original dataset " ,self.dataset)
            self.dataset.append(Statistics_median_grouped)
            print("\nAfter modification dataset " ,self.dataset)
            return Statistics_median_grouped
    def Add_fake_oneday_mode(self):
        """
        Return the most common data point from discrete or nominal data. 
        
        The mode (when it exists) is the most typical value, and is a robust measure of central location.
        """
        
        print("*Note five data for statistics.mode")
        Statistics_mode = statistics.mode(self.dataset[-5:])
        if Statistics_mode > (self.Upper_lower_limit * 1.1) or Statistics_mode < (self.Upper_lower_limit * 0.9):
            print("Upper_lower_limit = ",self.Upper_lower_limit ,"\nNow Value is %.2f" %Statistics_mode)
        else:
            print("original dataset " ,self.dataset)
            self.dataset.append(Statistics_mode)
            print("\nAfter modification dataset " ,self.dataset)
            return Statistics_mode
if __name__ == '__main__':

    t1 = [20,23,26,27,26,25,24]
    test = Fakedataset(t1).Add_fake_oneday_median_grouped()

    print( test) 
    