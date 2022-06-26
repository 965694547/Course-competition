import pandas as pd
import numpy as np


#test=input.T.T[['A','B']]
#test.to_csv('test.tsv',header=0,index=0,sep='\t')
#比较输入和标注差别
input=pd.read_csv('dev.tsv',sep='\t')
output=pd.read_csv('predict_result.tsv',sep='\t')
result = np.where(input['C']==output['C'],'same','different')
df_result = pd.DataFrame(result)
df = pd.concat([input, df_result], axis=1)
df.to_csv('compare.csv',header=0,index=0)


#标注词性
import pandas as pd
from  snownlp import SnowNLP
#posseg_list=SnowNLP('我爱自然语言处理')
#print(' '.join('%s/%s'%(word,tag)for(word,tag)in posseg_list.tags))
#train=pd.read_csv('train.tsv',sep='\t')
dev=pd.read_csv('dev.tsv',sep='\t')
for row in range(0,dev.shape[0]):
    dev.loc[row,'A'] = ' '.join('%s/%s'%(word,tag)for(word,tag)in SnowNLP(dev.loc[row,'A']).tags)
    dev.loc[row,'B'] = ' '.join('%s/%s'%(word,tag)for(word,tag)in SnowNLP(dev.loc[row,'B']).tags)
dev.to_csv('dev_tag.tsv',header=0,index=0,sep='\t')
train1=pd.read_csv('./data/BQ/train',sep='\t',error_bad_lines=False)
train2=pd.read_csv('./data/LCQMC/train',sep='\t',error_bad_lines=False)
train3=pd.read_csv('./data/OPPO/train',sep='\t',error_bad_lines=False)
#train = pd.concat([train1, train2,train3], axis=0)
for row in range(0,train1.shape[0]):
    train1.loc[row,'A'] = ' '.join('%s/%s'%(word,tag)for(word,tag)in SnowNLP(train1.loc[row,'A']).tags)
    train1.loc[row,'B'] = ' '.join('%s/%s'%(word,tag)for(word,tag)in SnowNLP(train1.loc[row,'B']).tags)
for row in range(0, train2.shape[0]):
    train2.loc[row,'A'] = ' '.join('%s/%s'%(word,tag)for(word,tag)in SnowNLP(train2.loc[row,'A']).tags)
    train2.loc[row,'B'] = ' '.join('%s/%s'%(word,tag)for(word,tag)in SnowNLP(train2.loc[row,'B']).tags)
for row in range(126437, train3.shape[0]):
    train3.loc[row,'A'] = ' '.join('%s/%s' % (word, tag) for (word, tag) in SnowNLP(train3.loc[row, 'A']).tags)
    train3.loc[row,'B'] = ' '.join('%s/%s' % (word, tag) for (word, tag) in SnowNLP(train3.loc[row, 'B']).tags)
train = pd.concat([train1, train2,train3], axis=0)
train.to_csv('train_tag.tsv',header=0,index=0,sep='\t')

#词云替换
import numpy as np
import pandas as pd
from  snownlp import SnowNLP
dictionary = pd.DataFrame(columns = ["A", "B"])
train1=pd.read_csv('./data/BQ/train',sep='\t',error_bad_lines=False)
train2=pd.read_csv('./data/LCQMC/train',sep='\t',error_bad_lines=False)
train3=pd.read_csv('./data/OPPO/train',sep='\t',error_bad_lines=False)
i = 0
for row in range(0,train1.shape[0]):
    for (word, tag) in SnowNLP(train1.loc[row, 'A']).tags:
        new = pd.DataFrame({'A':word,'B':tag},index=[i])
        i = i+1
        dictionary= dictionary.append(new,ignore_index=True)
    null_row = pd.DataFrame({'A': np.NaN, 'B':np.NaN},index = [i])
    i = i + 1
    dictionary= dictionary.append(null_row,ignore_index=True)# 空行，用于分割
    for (word, tag) in SnowNLP(train1.loc[row, 'B']).tags:
        new = pd.DataFrame({'A':word,'B':tag},index=[i])
        i = i+1
        dictionary= dictionary.append(new,ignore_index=True)
    null_row = pd.DataFrame({'A': np.NaN, 'B': np.NaN},index = [i])
    i = i + 1
    dictionary= dictionary.append(null_row,ignore_index=True)
for row in range(0,train2.shape[0]):
    for (word, tag) in SnowNLP(train2.loc[row, 'A']).tags:
        new = pd.DataFrame({'A':word,'B':tag},index=[i])
        i = i+1
        dictionary= dictionary.append(new,ignore_index=True)
    null_row = pd.DataFrame({'A': np.NaN, 'B':np.NaN},index = [i])
    i = i + 1
    dictionary= dictionary.append(null_row,ignore_index=True)# 空行，用于分割
    for (word, tag) in SnowNLP(train2.loc[row, 'B']).tags:
        new = pd.DataFrame({'A':word,'B':tag},index=[i])
        i = i+1
        dictionary= dictionary.append(new,ignore_index=True)
    null_row = pd.DataFrame({'A': np.NaN, 'B': np.NaN},index = [i])
    i = i + 1
    dictionary= dictionary.append(null_row,ignore_index=True)#
for row in range(0,train3.shape[0]):
    for (word, tag) in SnowNLP(train3.loc[row, 'A']).tags:
        new = pd.DataFrame({'A':word,'B':tag},index=[i])
        i = i+1
        dictionary= dictionary.append(new,ignore_index=True)
    null_row = pd.DataFrame({'A': np.NaN, 'B':np.NaN},index = [i])
    i = i + 1
    dictionary= dictionary.append(null_row,ignore_index=True)# 空行，用于分割
    for (word, tag) in SnowNLP(train3.loc[row, 'B']).tags:
        new = pd.DataFrame({'A':word,'B':tag},index=[i])
        i = i+1
        dictionary= dictionary.append(new,ignore_index=True)
    null_row = pd.DataFrame({'A': np.NaN, 'B': np.NaN},index = [i])
    i = i + 1
    dictionary= dictionary.append(null_row,ignore_index=True)

dictionary.to_csv('dictionary.tsv',header=0,index=0,sep='\t')
dictionary.to_csv('dictionary.csv',header=0,index=0)


import paddle
from paddlenlp.embeddings import TokenEmbedding, list_embedding_name
token_embedding = TokenEmbedding(embedding_name="w2v.baidu_encyclopedia.target.word-word.dim300")
score = token_embedding.cosine_sim("中国", "美国")

import numpy as np
import pandas as pd
import random
new_train = pd.DataFrame(columns = ["A", "B"])
dictionary=pd.read_csv('dictionary.tsv',sep='\t',error_bad_lines=False)
dictionary_n = pd.read_csv('dictionary_n.tsv',sep='\t',error_bad_lines=False)
temp_A =''#存储当前句子
temp_B =''
flag =1 #是否已经调换名词
word=''
subword=''
yuzhi = 5#替换词与原词的差距
i = 0
for row in range(0,dictionary.shape[0]):

    if dictionary.loc[row,'A'] != np.NaN:
        word = dictionary.loc[row,'A']
        if dictionary.loc[row,'A'] == '/n' and flag :
            flag = 0
            while  token_embedding.cosine_sim(dictionary.loc[row,'A'],word)>yuzhi :
                word = dictionary_n.loc[random.randrange(0,dictionary_n.shap[0]),'A']
        temp_A = temp_A + dictionary.loc[row,'A']
        temp_B = temp_B + word
    else:
        if flag == 0 :
            new_row = pd.DataFrame({'A': temp_A, 'B': temp_B}, index=[i])
            i = i + 1
            dictionary = dictionary.append(new_row, ignore_index=True)
        flag = 1
        temp_A = ''  # 存储当前句子
        temp_B = ''


a.find('/y ')

#########替换近义词
import pandas as pd
from synonyms  import nearby
from  snownlp import SnowNLP
dictionary = pd.DataFrame(columns = ["A", "B"])
train1=pd.read_csv('./data/BQ/train',sep='\t',error_bad_lines=False)
train2=pd.read_csv('./data/LCQMC/train',sep='\t',error_bad_lines=False)
train3=pd.read_csv('./data/OPPO/train',sep='\t',error_bad_lines=False)
i = 0
class_AORB = 'A'
#print("识别: %s%s" % (nearby("识别")))
for row in range(0,train1.shape[0]):
    j = 0
    sentence = ''
    flag_change = 1 #为1说明没有替换、为0说明已经替换
    for (word, tag) in SnowNLP(train1.loc[row, class_AORB]).tags:
        if (tag == 'n' or tag == 'a') and flag_change ==1 :
            flag_change = 0
            change = nearby(word)
            if (len(change[0]) >= 2):
                if(change[1][1]>=0.85):
                    word = change[0][1]
                else:
                    flag_change = 1
            else:
                flag_change = 1
        sentence = sentence + word
    if flag_change == 0:
        dictionary.loc[i, 'A'] = train1.loc[row, class_AORB]
        dictionary.loc[i, 'B'] = sentence
        i = i + 1
class_AORB = 'B'
for row in range(0,train1.shape[0]):
    j = 0
    sentence = ''
    flag_change = 1 #为1说明没有替换、为0说明已经替换
    for (word, tag) in SnowNLP(train1.loc[row, class_AORB]).tags:
        if (tag == 'n' or tag == 'a') and flag_change ==1 :
            flag_change = 0
            change = nearby(word)
            if (len(change[0]) >= 2):
                if(change[1][1]>=0.85):
                    word = change[0][1]
                else:
                    flag_change = 1
            else:
                flag_change = 1
        sentence = sentence + word
    if flag_change == 0:
        dictionary.loc[i, 'A'] = train1.loc[row, class_AORB]
        dictionary.loc[i, 'B'] = sentence
        i = i + 1

class_AORB = 'A'
#print("识别: %s%s" % (nearby("识别")))
for row in range(0,train2.shape[0]):
    j = 0
    sentence = ''
    flag_change = 1 #为1说明没有替换、为0说明已经替换
    for (word, tag) in SnowNLP(train2.loc[row, class_AORB]).tags:
        if (tag == 'n' or tag == 'a') and flag_change ==1 :
            flag_change = 0
            change = nearby(word)
            if (len(change[0]) >= 2):
                if(change[1][1]>=0.85):
                    word = change[0][1]
                else:
                    flag_change = 1
            else:
                flag_change = 1
        sentence = sentence + word
    if flag_change == 0:
        dictionary.loc[i, 'A'] = train2.loc[row, class_AORB]
        dictionary.loc[i, 'B'] = sentence
        i = i + 1
class_AORB = 'B'
for row in range(0,train2.shape[0]):
    j = 0
    sentence = ''
    flag_change = 1 #为1说明没有替换、为0说明已经替换
    for (word, tag) in SnowNLP(train2.loc[row, class_AORB]).tags:
        if (tag == 'n' or tag == 'a') and flag_change ==1 :
            flag_change = 0
            change = nearby(word)
            if (len(change[0]) >= 2):
                if(change[1][1]>=0.85):
                    word = change[0][1]
                else:
                    flag_change = 1
            else:
                flag_change = 1
        sentence = sentence + word
    if flag_change == 0:
        dictionary.loc[i, 'A'] = train2.loc[row, class_AORB]
        dictionary.loc[i, 'B'] = sentence
        i = i + 1

for row in range(0,train3.shape[0]):
    j = 0
    sentence = ''
    flag_change = 1 #为1说明没有替换、为0说明已经替换
    for (word, tag) in SnowNLP(train3.loc[row, class_AORB]).tags:
        if (tag == 'n' or tag == 'a') and flag_change ==1 :
            flag_change = 0
            change = nearby(word)
            if (len(change[0]) >= 2):
                if(change[1][1]>=0.85):
                    word = change[0][1]
                else:
                    flag_change = 1
            else:
                flag_change = 1
        sentence = sentence + word
    if flag_change == 0:
        dictionary.loc[i, 'A'] = train3.loc[row, class_AORB]
        dictionary.loc[i, 'B'] = sentence
        i = i + 1
class_AORB = 'B'
for row in range(0,train1.shape[0]):
    j = 0
    sentence = ''
    flag_change = 1 #为1说明没有替换、为0说明已经替换
    for (word, tag) in SnowNLP(train1.loc[row, class_AORB]).tags:
        if (tag == 'n' or tag == 'a') and flag_change ==1 :
            flag_change = 0
            change = nearby(word)
            if (len(change[0]) >= 2):
                if(change[1][1]>=0.85):
                    word = change[0][1]
                else:
                    flag_change = 1
            else:
                flag_change = 1
        sentence = sentence + word
    if flag_change == 0:
        dictionary.loc[i, 'A'] = train1.loc[row, class_AORB]
        dictionary.loc[i, 'B'] = sentence
        i = i + 1
dictionary.to_csv('dictionary.tsv',header=0,index=0,sep='\t')

from paddlenlp.taskflow import TaskFlow
lac = TaskFlow("lexical_analysis")

from paddlenlp  import taskflow
lac =  taskflow.Taskflow("lexical_analysis")
lac("LAC是个优秀的分词工具")

####中英互译
from googletrans import Translator
def googletrans(content='一个免费的谷歌翻译API', t_from='zh-cn', t_to='en'):
    translator = Translator()
    s = translator.translate(text=content, dest=t_to,src=t_from)
    return s.text
##baidu
import pandas as pd
from nlpcda import baidu_translate
en_lib=pd.read_csv('./import data/quora_duplicate_questions.tsv',sep='\t')
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
new_train.to_csv('new_train.tsv',header=0,index=0,sep='\t')
new_train.to_csv('new_train.csv',header=0,index=0)

####中英互译
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
        if i%1000==0:
            new_train.to_csv('new_train.tsv', header=0, index=0, sep='\t')
            new_train.to_csv('new_train.csv', header=0, index=0)
new_train.to_csv('new_train.tsv',header=0,index=0,sep='\t')
new_train.to_csv('new_train.csv',header=0,index=0)

#jieba分词
import jieba.posseg as pseg
words =pseg.cut("我爱北京天安门")
for w in words:
    w.word, w.flag
from LAC import LAC
lac = LAC(mode='rank')
text = u"LAC是个优秀的分词工具"
seg_result = lac.run(text)

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
    if(len(dictionary_A)%1000==0):
        np.save('test_A_LAC.npy', dictionary_A)
        np.save('test_B_LAC.npy', dictionary_B)
np.save('test_A_LAC.npy', dictionary_A)
np.save('test_B_LAC.npy', dictionary_B)
dictionary_A=np.load('test_A_LAC.npy',allow_pickle=True)
dictionary_B=np.load('test_B_LAC.npy',allow_pickle=True)
dataDf_A=pd.DataFrame(dictionary_A)
dataDf_B=pd.DataFrame(dictionary_B)
new_dataDf = pd.DataFrame(columns = ["A", "B", "C","D", "E", "F"])
i=0
for row in range(0,dataDf_A.shape[0]):
    new_dataDf.loc[i, 'A'] = dataDf_A.loc[3*i,0]
    new_dataDf.loc[i, 'B'] = dataDf_A.loc[3*i+1,0]
    new_dataDf.loc[i, 'C'] = dataDf_A.loc[3*i+2,0]
    new_dataDf.loc[i, 'D'] = dataDf_B.loc[3*i,0]
    new_dataDf.loc[i, 'E'] = dataDf_B.loc[3*i+1,0]
    new_dataDf.loc[i, 'F'] = dataDf_B.loc[3*i+2,0]
    i=i+1
new_dataDf.to_csv('new_dataDf.tsv',header=0,index=0,sep='\t')

###对应词修改
import pandas as pd
test=pd.read_csv('test_A.tsv',sep='\t',error_bad_lines=False)
new_dataDf=pd.read_csv('new_dataDf.tsv',sep='\t',error_bad_lines=False)
dictionary_A=np.load('test_A_LAC.npy',allow_pickle=True)
dictionary_B=np.load('test_B_LAC.npy',allow_pickle=True)
Corresponding = pd.DataFrame(columns = ["A", "B"])
Corresponding_left = pd.DataFrame(columns = ["A", "B"])
i=0
for row in range(0,new_dataDf.shape[0]):
    if(new_dataDf.loc[row,'B']==new_dataDf.loc[row,'E']):
        Corresponding_left.loc[i, 'A'] = test.loc[row, 'A']
        Corresponding_left.loc[i, 'B'] = test.loc[row, 'B']
        for j in range(0,len(dictionary_A[3*row])):
            if dictionary_A[3*row][j]!=dictionary_B[3*row][j]:
                Corresponding.loc[i, 'A']=dictionary_A[3*row][j]
                Corresponding.loc[i, 'B']=dictionary_B[3*row][j]
        i=i+1
Corresponding.to_csv('Corresponding.tsv',header=0,index=0,sep='\t')
Corresponding_left.to_csv('Corresponding_left.tsv',header=0,index=0,sep='\t')

output = pd.read_csv('predict_result.csv',error_bad_lines=False)
output_result = pd.DataFrame(columns = ["A", "B","C","D","E"])
for row in range(3632,output.shape[0]):
    output_result.loc[row,'A']=Corresponding_left.loc[row,'A']
    output_result.loc[row, 'B'] = Corresponding_left.loc[row, 'B']
    output_result.loc[row, 'C'] = Corresponding.loc[row, 'A']
    output_result.loc[row, 'D'] = Corresponding.loc[row, 'B']
    output_result.loc[row, 'E'] = output.loc[row-1, '1']
output_result.to_csv('output_result.tsv',header=0,index=0,sep='\t')

#处理单副词形容词
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
            elif dictionary_A[3*row+1][indexl]!='nr' and dictionary_A[3*row+1][indexl]!='ns':
                flag =0
                break
            elif dictionary_A[3*row+1][indexl]=='nr' or dictionary_A[3*row+1][indexl]=='ns':
                continue
    if len(dictionary_A[3*row])<len(dictionary_B[3*row]):
        lens= len(dictionary_A[3*row])
        indexs = 0
        for indexl in range(0,len(dictionary_B[3*row])):
            if indexs<lens and dictionary_B[3*row][indexl]==dictionary_A[3*row][indexs]:
                indexs = indexs+1
            elif dictionary_B[3*row+1][indexl]!='nr' and dictionary_B[3*row+1][indexl]!='ns':
                flag = 0
                break
            elif dictionary_B[3*row+1][indexl]=='nr' or dictionary_B[3*row+1][indexl]=='ns':
                continue
    if len(dictionary_A[3 * row]) == len(dictionary_B[3 * row]):
        continue
    if flag :
        adv_adj.loc[i, "A"] = test.loc[row,'A']
        adv_adj.loc[i, "B"] = test.loc[row,'B']
        i = i+1
adv_adj.to_csv('adv_adj.tsv', header=0, index=0, sep='\t')


###不一样的词语
    set_word_A = set(dictionary_A[3*row])
    set_word_B = set(dictionary_B[3*row])
    differ_word=set_word_A^set_word_B
    #set_part_A = set(dictionary_A[3 * row+1])
    #set_part_B = set(dictionary_B[3 * row+1])
    #differ_part=set_part_A^set_part_B
    for item in differ_word:
        if item in dictionary_A[3*row]:
            index=dictionary_A[3 * row].index(item)
            if dictionary_A[3 * row+1][index] == 'a' or dictionary_A[3 * row+1][index] == 'ad':
                adv_adj.loc[i,"A"]=dictionary_A[3*row]
                adv_adj.loc[i, "B"] = dictionary_B[3 * row]
                i=i+1
        else:
            index = dictionary_B[3 * row].index(item)
            if dictionary_B[3 * row+1][index] == 'a' or dictionary_B[3 * row+1][index] == 'ad':
                adv_adj.loc[i,"A"]=dictionary_A[3*row]
                adv_adj.loc[i, "B"] = dictionary_B[3 * row]
                i = i + 1
    adv_adj.to_csv('adv_adj.tsv', header=0, index=0, sep='\t')

#处理特定相同句
import pandas as pd
import numpy as np
#from paddlenlp.embeddings import TokenEmbedding
#wordemb = TokenEmbedding("w2v.baidu_encyclopedia.target.word-word.dim300")
test=pd.read_csv('test_A.tsv',sep='\t',error_bad_lines=False)
new_dataDf=pd.read_csv('new_dataDf.tsv',sep='\t',error_bad_lines=False)
dictionary_A=np.load('test_A_LAC.npy',allow_pickle=True)
dictionary_B=np.load('test_B_LAC.npy',allow_pickle=True)
Corresponding = pd.DataFrame(columns = ["A", "B",'C'])
Corresponding_left = pd.DataFrame(columns = ["A", "B"])
#comparelist = ['f','s','t','nr','ns','nt','nw','nz','PER','LOC','ORG','TIME0']
comparelist = ['xc','v']
i=0
for row in range(0,new_dataDf.shape[0]):
    if(new_dataDf.loc[row,'B']==new_dataDf.loc[row,'E']):
        flag = 0
        for j in range(0,len(dictionary_A[3*row])):
            if dictionary_A[3*row][j]!=dictionary_B[3*row][j]:
                for label in comparelist:
                    if dictionary_A[3*row+1][j]==label:
                        break
                        #for word in dictionary_A[3*row][j]:
                if not set(dictionary_A[3*row][j]) & set(dictionary_B[3*row][j]):
                        Corresponding.loc[i, 'A']=dictionary_A[3*row][j]
                        Corresponding.loc[i, 'B']=dictionary_B[3*row][j]
                        Corresponding.loc[i, 'C']=dictionary_B[3*row+1][j]
                        flag = flag+1
        if flag ==1:
            Corresponding_left.loc[i, 'A'] = test.loc[row, 'A']
            Corresponding_left.loc[i, 'B'] = test.loc[row, 'B']
            i=i+1
output_result = pd.DataFrame(columns = ["A", "B","C","D","E"])
for row in range(0,Corresponding_left.shape[0]-1):
    output_result.loc[row,'A']=Corresponding_left.loc[row,'A']
    output_result.loc[row, 'B'] = Corresponding_left.loc[row, 'B']
    output_result.loc[row, 'C'] = Corresponding.loc[row, 'A']
    output_result.loc[row, 'D'] = Corresponding.loc[row, 'B']
    output_result.loc[row, 'E'] = Corresponding.loc[row, 'C']
output_result.to_csv('output_result.tsv',header=0,index=0,sep='\t')

for row in range(0,output_result.shape[0]-1):
    output_result.loc[row, 'E'] = wordemb.cosine_sim(output_result.loc[row, 'C'], output_result.loc[row, 'D'])
output_result.to_csv('output_result.tsv',header=0,index=0,sep='\t')
output_result.to_csv('output_result.csv',header=0,index=0)
