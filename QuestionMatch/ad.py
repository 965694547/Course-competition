import numpy as np
import pandas as pd
dictionary_A=np.load('test_A_LAC.npy',allow_pickle=True)
dictionary_B=np.load('test_B_LAC.npy',allow_pickle=True)
test=pd.read_csv('test_A.tsv',sep='\t',error_bad_lines=False)
adv_adj = pd.DataFrame(columns = ["A", "B"])
i=0
for row in range(0,test.shape[0]):
    flag = 1
    if len(dictionary_A[3*row])>len(dictionary_B[3*row]):
        lens= len(dictionary_B[3*row])
        indexs = 0
        for indexl in range(0,len(dictionary_A[3*row])):
            if indexs<lens and dictionary_A[3*row][indexl]==dictionary_B[3*row][indexs]:
                indexs = indexs+1
            elif dictionary_A[3*row+1][indexl]!='ad' and dictionary_A[3*row+1][indexl]!='a':
                flag =0
                break
            elif dictionary_A[3*row+1][indexl]=='ad' or dictionary_A[3*row+1][indexl]=='a':
                continue
    if len(dictionary_A[3*row])<len(dictionary_B[3*row]):
        lens= len(dictionary_A[3*row])
        indexs = 0
        for indexl in range(0,len(dictionary_B[3*row])):
            if indexs<lens and dictionary_B[3*row][indexl]==dictionary_A[3*row][indexs]:
                indexs = indexs+1
            elif dictionary_B[3*row+1][indexl]!='ad' and dictionary_B[3*row+1][indexl]!='a':
                flag = 0
                break
            elif dictionary_B[3*row+1][indexl]=='ad' or dictionary_B[3*row+1][indexl]=='a':
                continue
    if len(dictionary_A[3 * row]) == len(dictionary_B[3 * row]):
        continue
    if flag :
        adv_adj.loc[i, "A"] = test.loc[row,'A']
        adv_adj.loc[i, "B"] = test.loc[row,'B']
        i = i+1