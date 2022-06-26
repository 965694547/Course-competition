import numpy as np
import pandas as pd
from LAC import LAC
lac = LAC(mode='rank')
test=pd.read_csv('test_A.tsv',sep='\t',error_bad_lines=False)
#dictionary = pd.DataFrame(columns = ["A", "B"])
#i = 0
dictionary_A=list()
dictionary_B=list()
for row in range(0,test.shape[0]):
    seg_result_A = lac.run(test.loc[row, 'A'])
    dictionary_A = dictionary_A+seg_result_A
    seg_result_B = lac.run(test.loc[row, 'B'])
    dictionary_B = dictionary_B + seg_result_B
    #np.c_[dictionary,seg_result_A+seg_result_B]
    #dictionary = dictionary+seg_result_A+seg_result_B
    if(len(dictionary_A)%3000==0):
        np.save('test_A_LAC.npy', dictionary_A)
        np.save('test_B_LAC.npy', dictionary_B)
np.save('test_A_LAC.npy', dictionary_A)
np.save('test_B_LAC.npy', dictionary_B)
dictionary_A=np.load('test_A_LAC.npy')
dictionary_B=np.load('test_B_LAC.npy')