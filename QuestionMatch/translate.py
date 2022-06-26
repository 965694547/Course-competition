import pandas as pd
from nlpcda import baidu_translate
en_lib=pd.read_csv('quora_duplicate_questions.tsv',sep='\t')
new_train = pd.DataFrame(columns = ["A", "B", "C"])
i =0
for row in range(0,en_lib.shape[0]):
    en1 = en_lib.loc[row, 'question1']
    en2 = en_lib.loc[row, 'question2']
    zh1 = baidu_translate(content=en1, appid='20211007000966000', secretKey='qdjGQdcNI4VzhurLUIaq', t_from='en',
                           t_to='zh')
    zh2 = baidu_translate(content=en2, appid='20211007000966000', secretKey='qdjGQdcNI4VzhurLUIaq', t_from='en',
                           t_to='zh')
    if zh1!=zh2:
        new_train.loc[i, 'A'] = zh1
        new_train.loc[i, 'B'] = zh2
        new_train.loc[i, 'C'] = en_lib.loc[row, 'is_duplicate']
        i=i+1
        if i%1000=0:
            new_train.to_csv('new_train.tsv', header=0, index=0, sep='\t')
            new_train.to_csv('new_train.csv', header=0, index=0)
new_train.to_csv('new_train.tsv',header=0,index=0,sep='\t')
new_train.to_csv('new_train.csv',header=0,index=0)