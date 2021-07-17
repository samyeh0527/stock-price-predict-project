# BiGRU
Stock 2344Tw and 4919TW for BiGRU preidct price
"""
            1.---CreateDataset.py               =>  {input:}    dataset,look_bac
            |                                   =>  {ootput:}   TrainX,  Train_Y 
            |
            2.---LoadDataset.py                 =>  {input:}    path1,path2
            |                                   =>  {output:}   testVaild,  trainVaild,  dataset
            |
main.py-----3.---SplitTrainAndTest.py           =>  {input:}    dataset,    testVaild,  trainVaild
            |                                       {output:}   train,  test,  testVaild,   trainVaild
            |
            4.---Normalize.py                   =>  {iunput:}   data
            |                                       {output:}   data,   normalize
            |
            5.---Attention_model.py             =>  {input:}    Dims,   TimeSteps
            |                                       {output:}   model
            |
            |
            6.---Fnormalize.py                  =>  {input:}    data,   normalize      
            |                                       {output:}   data
            |
            7.---Attention_function.py          =>  {input:}    inputs
            |                                       {output:}   output_attention_mul
"""

provide conda env yml for use .
you can find the conda list full pakge version 
this case using GPU 1660s base Cuda
