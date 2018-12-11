# -*- coding: utf-8 -*-

# 最近傍法

import sys
import os
import numpy as np
from PIL import Image
from collections import Counter
from random import sample

#プロトタイプの数を入力
print("Please input K:")
K = int(input())

# 読み込む画像数
print("Please input N:")
N = int(input())
train_img = np.zeros((10,K,28,28), dtype=np.float32)

# プロトタイプの読み込みと作成
for i in range(10):
    for k in range(K):
        train_img_1 = []

        # 連番を作成し、その中からN個のランダムなインデックスのリストを作成
        hundred = list(np.arange(1, 101))
        train_list = sample(hundred, N)

        # 画像を読み込み
        for j in train_list:
            train_file = "mnist/train/" + str(i) + "/" + str(i) + "_" + str(j) + ".jpg"
            train_img_1.append(np.asarray(Image.open(train_file).convert('L')).astype(np.float32))

        # 平均画像を作成
        train_img[i][k] = np.mean(train_img_1, axis=0)
        print("{}, k = {}, class = {}".format(train_img[i][k].shape, k, i))


# 混合行列
result = np.zeros((10,10), dtype=np.int32)




#0-9まで検索
for i in range(10):
    for j in range(1,101):
        # 未知パターンの読み込み
        pat_file = "mnist/test/" + str(i) + "/" + str(i) + "_" + str(j) + ".jpg"
        pat_img = np.asarray(Image.open(pat_file).convert('L')).astype(np.float32)

        min_val = float('inf')
        ans = 0

        #候補をストックしておく配列
        cnd = []

        # 最近傍法
        for k in range(10):
            for l in range(1,K+1):
                # SSDの計算
                t = train_img[k][l-1].flatten()
                p = pat_img.flatten()
                dist = np.dot( (t-p).T , (t-p) )

                # 最小値の探索
                if dist < min_val:
                    min_val = dist
                    ans = k

        # 結果の出力
        result[i][ans] +=1
        print( i , j , "->" , ans )
        
print( "\n [ 混合行列 ]" )
print( result )
print( "\n 正解数 -> " , np.trace(result) )

