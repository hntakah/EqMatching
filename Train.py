
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import h5py
import os
import argparse
import numpy as np
import cv2
from EqMatchingLib import CreateTrainData,train

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--CubemapImg",type=str,default="TestCube.png")
    parser.add_argument("--template",type=str,default="BEM12.png")
    parser.add_argument("--UsingData",type=str,default="TrainData_15_n500.npy")
    parser.add_argument("--UsingLabel",type=str,default="TrainLabel_15_n500.npy")
    parser.add_argument("--SavingData",type=str,default="TrainData_16_n500.npy")
    parser.add_argument("--SavingLabel",type=str,default="TrainLabel_16_n500.npy")
    parser.add_argument("--UsingData",type=str,default="TrainData_15_n500.npy")
    parser.add_argument("--UsingLabel",type=str,default="TrainLabel_15_n500.npy")
    parser.add_argument("--weightsfile",type=str,default="weights_n500_15_ep60.h5")
    parser.add_argument("--weights",type=str,default=None)    
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--nb_epoch", type=int, default=20)
    parser.add_argument("--num", type=int, default=500)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    #CubemapImg,template,Savingxxx,num
    if args.mode == "create":
        #画像の読み込み
        CubemapImg=cv2.imread(args.CubemapImg)
        filename=args.CubemapImg
        template=cv2.imread(args.template)
        h,w,ch=CubemapImg.shape
        h_template,w_template,ch_template=template.shape
        #一面ごとの縦横幅
        FaceSize=h/3

        height=int(w/3)
        width=int(2*height)
        #Traindataを作成
        num=args.num
        TrainData,TrainLabel=CreateTrainData(FaceSize,w_template,h_template,filename,template,num)

        #TrainDataの保存
        np.save(args.SavingData,TrainData)
        np.save(args.SavingLabel,TrainLabel)

    #Usingxxx,weights(arb),batch_size,nb_epoch
    elif args.mode == "train":
        #template=cv2.imread(args.template)
        #既存のTrainDataを使用
        TrainData=np.load(args.UsingData)
        TrainLabel=np.load(args.UsingLabel)
        height=TrainData[0,:].shape[0]
        width=TrainData[0,:].shape[1]
        batch_size=args.batch_size
        nb_epoch=args.nb_epoch
        weights=args.weights
        weightsFilename=args.weightsfile
        train(TrainData,[np.array(TrainLabel[:,0]),np.array(TrainLabel[:,1]),np.array(TrainLabel[:,2]),np.array(TrainLabel[:,3])],width,height,weightsFilename,batch_size,nb_epoch,weights)
