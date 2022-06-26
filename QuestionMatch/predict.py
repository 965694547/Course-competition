# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import argparse
import sys
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad

import pandas as pd
from data import create_dataloader, read_text_pair, convert_example
from model import QuestionMatching
from pypinyin import lazy_pinyin
from LAC import LAC
lac = LAC(mode='lac')
# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="The full path of input file")
parser.add_argument("--result_file", type=str, required=True, help="The result file name")
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable
comparelist = ['xc', 'v']

def predict(model, data_loader):
    """
    Predicts the data labels.

    Args:
        model (obj:`QuestionMatching`): A model to calculate whether the question pair is semantic similar or not.
        data_loaer (obj:`List(Example)`): The processed data ids of text pair: [query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids]
    Returns:
        results(obj:`List`): cosine similarity of text pairs.
    """
    batch_logits = []

    model.eval()

    with paddle.no_grad():
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data

            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)

            batch_logit, _ = model(
                input_ids=input_ids, token_type_ids=token_type_ids)

            batch_logits.append(batch_logit.numpy())

        batch_logits = np.concatenate(batch_logits, axis=0)

        return batch_logits

def is_num(item):
    for i in item:
        if(not(i >= '0' and i <= '9')):
            return False
    return True
if __name__ == "__main__":
    paddle.set_device(args.device)

    pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(
        'ernie-gram-zh')
    tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained(
        'ernie-gram-zh')

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_test=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids
    ): [data for data in fn(samples)]

    test_ds = load_dataset(
        read_text_pair, data_path=args.input_file, is_test=True, lazy=False)

    test_data_loader = create_dataloader(
        test_ds,
        mode='predict',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    model = QuestionMatching(pretrained_model)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    y_probs = predict(model, test_data_loader)
    y_preds = np.argmax(y_probs, axis=1)


    file_test = open(args.input_file, encoding='utf-8')

    with open(args.input_file, 'w', encoding="utf-8") as f:
        for y_pred in y_preds:
            line = file_test.readline()
            line = line.split('\t')
            q1 = line[0]
            q2 = line[-1][:-1]

            ##单词匹配_专有名词
            flag = 0 #当且仅当差一个词的时候修改为0
            seg_result_A = lac.run(q1)
            seg_result_B = lac.run(q2)
            if (seg_result_A[1] == seg_result_B[1]):
                for j in range(0, len(seg_result_A[0])):
                    if seg_result_A[0][j] != seg_result_B[0][j]:
                        if not set(seg_result_A[0][j]) & set(seg_result_B[0][j]):
                            flag = flag + 1
                            for label in comparelist:#去除容易意思重叠的动词和助词
                                if seg_result_A[1][j] == label:
                                    flag = flag - 1
                if flag == 1:
                    y_pred = 0


            ##单词匹配
            #lac = LAC(mode='rank')
            '''seg_result_A = lac.run(q1)
            seg_result_B = lac.run(q2)
            input_word1 = ''
            input_word2 = ''
            if (seg_result_A[1] == seg_result_B[1]):
                for j in range(0, len(seg_result_A[0])):
                    if seg_result_A[0][j] != seg_result_B[0][j]:
                        input_word1 = input_word1+seg_result_A[0][j]
                        input_word2 = input_word2+seg_result_B[0][j]
                if(input_word1!='' and input_word2!=''):
                    temp = pd.DataFrame(columns=["A", "B"])
                    temp.loc[0, 'A'] = input_word1
                    temp.loc[0, 'B'] = input_word2
                    temp.to_csv('temp.tsv', header=0, index=0, sep='\t')
                    test_ds = load_dataset(
                        read_text_pair, data_path='temp.tsv', is_test=True, lazy=False)
                    test_data_loader = create_dataloader(
                        test_ds,
                        mode='predict',
                        batch_size=1,
                        batchify_fn=batchify_fn,
                        trans_fn=trans_func)
                    y_probs = predict(model, test_data_loader)
                    y_out = np.argmax(y_probs, axis=1)
                    y_pred = y_out.tolist()[0]
                    os.remove("temp.tsv")'''

            ##插入形容词和副词
            '''flag = 1
            seg_result_A = lac.run(q1)
            seg_result_B = lac.run(q2)
            if len(seg_result_A[0]) > len(seg_result_B[0]):
                lens = len(seg_result_B[0])
                indexs = 0
                for indexl in range(0, len(seg_result_A[0])):
                    if indexs < lens and seg_result_A[0][indexl] == seg_result_B[0][indexs]:
                        indexs = indexs + 1
                    elif seg_result_A[1][indexl] != 'ad' and seg_result_A[1][indexl] != 'a':
                        flag = 0
                        break
                    #elif seg_result_A[1][indexl] == 'ad' or seg_result_A[1][indexl] == 'a':
                    #    continue
            if len(seg_result_A[0]) < len(seg_result_B[0]):
                lens = len(seg_result_A[0])
                indexs = 0
                for indexl in range(0, len(seg_result_B[0])):
                    if indexs < lens and seg_result_B[0][indexl] == seg_result_A[0][indexs]:
                        indexs = indexs + 1
                    elif seg_result_B[1][indexl] != 'ad' and seg_result_B[1][indexl] != 'a':
                        flag = 0
                        break
                    #elif seg_result_B[1][indexl] == 'ad' or seg_result_B[1][indexl] == 'a':
                    #    continue
            if len(seg_result_A[0]) != len(seg_result_B[0]) and flag:
                y_pred = 0'''

            ##错别字
            pinyin1 = lazy_pinyin(q1)
            pinyin2 = lazy_pinyin(q2)
            if(pinyin1 == pinyin2):
                y_pred = 1

            ##数字
            lac_q1 = lac.run(q1)
            lac_q2 = lac.run(q2)
            num_list1 = []
            num_list2 = []
            for item1 in lac_q1[0]:
                if is_num(item1):
                    num_list1.append(item1)
            for item2 in lac_q2[0]:
                if is_num(item2):
                    num_list2.append(item2)
            if (len(num_list1) > 0 and len(num_list2) > 0):
                num_list1.sort()
                num_list2.sort()
                if (num_list1 != num_list2 and len(num_list1) == len(num_list2)):
                    y_pred = 0
            f.write(str(y_pred) + "\n")
