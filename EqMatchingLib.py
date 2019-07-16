import cv2
import numpy as np
from math import pi,sin,cos,tan,atan2,hypot,floor
from random import random
import sys
from Cube2Eq import CubemapToEquirectangular,find_Cubemap_corresponding_pixel,find_corresponding_pixel
import matplotlib.pyplot as plt

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

def AddMarkerInCubemap(filename,template,position):
    img=cv2.imread(filename)
    r,c,*ch=img.shape
    Mat=np.float32([[1,0,position[0]],[0,1,position[1]]])
    dst=cv2.warpAffine(template,Mat,(c,r),img,cv2.INTER_LINEAR,cv2.BORDER_TRANSPARENT)        
    return img


def SetPosition(FaceSize,w_template,h_template):
    position=np.zeros(2)
    while(1):
        #left
        position[0]=random()*(4*FaceSize-w_template)
        #top
        position[1]=random()*(3*FaceSize-h_template)
    
        #position[0]が2*FaceSizeから3*FaceSize-w_templateの間ならposition[1]は0から3*FaceSize-h_templateの間
        if(2*FaceSize<position[0]<3*FaceSize-w_template):
            if(0<position[1]<3*FaceSize-h_template):
                break
    
        #それ以外ならposition[1]はFaceSizeから2*FaceSize-h_templateの間
        else:
            if(FaceSize<position[1]<2*FaceSize-h_template):
                break
    position[0]=int(position[0])
    position[1]=int(position[1])

    return position

def MakeLabel(FaceSize,position,w_template,h_template,width,height):
    position[0]=int(position[0])
    position[1]=int(position[1])
    p_x=int(position[0]+w_template)
    p_y=int(position[1]+h_template)

    left,top=find_Cubemap_corresponding_pixel(position[0],position[1],width,height,FaceSize)
    right,bottom=find_Cubemap_corresponding_pixel(p_x,p_y,width,height,FaceSize)
    return (left,top,right,bottom)


def CreateTrainData(FaceSize,w_template,h_template,CubemapImg,template,num=500):
    i=0
    TrainData=[]
    TrainLabel=[]
    while(i<num):
        #Markerの付与座標のleft-topを取得
        #サイズを可変にする
        #r=random()*0.7+0.3
        #サイズ固定
        r=1
        position=SetPosition(FaceSize,int(w_template*r),int(h_template*r))
        template2=cv2.resize(template,(int(w_template*r),int(h_template*r)))
        # Cubemap画像のどこかにMarker付与
        C_Testdata=AddMarkerInCubemap(CubemapImg,template2,position)
        # Eqirectangular展開
        Eq_Testdata=CubemapToEquirectangular(C_Testdata)
        #画像をグレースケール変換
        Eq_Testdata=cv2.cvtColor(Eq_Testdata,cv2.COLOR_BGR2GRAY)
        #画像を正規化
        Eq_Testdata = Eq_Testdata.astype('float32')
        Eq_Testdata=Eq_Testdata/255.0
        height,width=Eq_Testdata.shape
        #正解labelを計算
        (left,top,right,bottom)=MakeLabel(FaceSize,position,int(w_template*r),int(h_template*r),width,height)
        i=i+1
        TrainData.append(Eq_Testdata)
        TrainLabel.append((left,top,right,bottom))
        print('\r',i,end='')
    return TrainData,TrainLabel


# A-KAZE
def akazeMatching(Eq_Input,template):
    # A-KAZE検出器の生成
    akaze = cv2.AKAZE_create()                                

    img2=Eq_Input
    img2=img2*255
    img2=cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    img2=img2.astype('uint8')

    img1=template
    # 特徴量の検出と特徴量ベクトルの計算
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # Brute-Force Matcher生成
    bf = cv2.BFMatcher()

    # 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
    matches = bf.knnMatch(des1, des2, k=2)
    # Sort them in the order of their distance.
    mlist=sorted([(n.distance-m.distance ,m,n) for m,n in matches],key=lambda x:x[0])
          
    #五点の加算平均でマーカーの中心座標を推定
    EstMarkerCenter=np.zeros(2)
    for i in range(0,5):
        EstMarkerCenter=EstMarkerCenter+np.float32([kp2[mlist[i][1].trainIdx].pt])
    EstMarkerCenter=EstMarkerCenter/5
    
    #マーカー中心の推定座標
    posxeq=np.int32(EstMarkerCenter)[0][0]
    posyeq=np.int32(EstMarkerCenter)[0][1]
    #h=width/3,facesize=width/4
    w2,h2,*ch2=img2.shape
    """
    #left-top estimation
    #print(posxeq,posyeq)
    pts_eq=[]
    px,py = find_corresponding_pixel(int(posxeq),int(posyeq),h2,w2,int(w2*3/4))
    pts_eq.append([posxeq,posyeq])
    wmark,hmark,*ch1=template.shape
    coordx,coordy = find_Cubemap_corresponding_pixel(int(px+wmark),int(py),h2,w2,int(w2*3/4))
    pts_eq.append([coordx,coordy])
    coordx,coordy = find_Cubemap_corresponding_pixel(int(px+wmark),int(py+hmark),h2,w2,int(w2*3/4))
    pts_eq.append([coordx,coordy])
    coordx,coordy = find_Cubemap_corresponding_pixel(int(px),int(py+hmark),h2,w2,int(w2*3/4))
    pts_eq.append([coordx,coordy])
    pts_eq=np.array(pts_eq)
    """
        
    #center estimation
    pts_eq=[]
    px,py = find_corresponding_pixel(int(posxeq),int(posyeq),h2,w2,int(w2*3/4))
    wmark,hmark,*ch1=template.shape
    print(px,py)
    coordx,coordy = find_Cubemap_corresponding_pixel(int(px-wmark/2),int(py-hmark/2),h2,w2,int(w2*3/4))
    pts_eq.append([coordx,coordy])
    coordx,coordy = find_Cubemap_corresponding_pixel(int(px-wmark/2),int(py+hmark/2),h2,w2,int(w2*3/4))
    pts_eq.append([coordx,coordy])
    coordx,coordy = find_Cubemap_corresponding_pixel(int(px+wmark/2),int(py+hmark/2),h2,w2,int(w2*3/4))
    pts_eq.append([coordx,coordy])
    coordx,coordy = find_Cubemap_corresponding_pixel(int(px+wmark/2),int(py-hmark/2),h2,w2,int(w2*3/4))
    pts_eq.append([coordx,coordy])
    pts_eq=np.array(pts_eq)
        
    #img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)      
    tmp=img2
        
    cv2.polylines(tmp,[pts_eq],True,(0,0,255),thickness=10)
            
    #print('pts_eq',pts_eq)
    tmp=cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)
    plt.imshow(tmp)
    plt.show()
    

def TMatching(Eq_Input,template):
    #Template Matching
    tmp=0
    template=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    template=template.astype('float32')
    
    img_CV=Eq_Input
    #img_CV=img_CV*255
    #img_CV=img_CV.astype('uint8')
    res = cv2.matchTemplate(img_CV,template,eval('cv2.TM_CCOEFF'))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    pts_eq=[]
    h2,w2,*c2=img_CV.shape
    wmark,hmark,*ch1=template.shape
    """
    #図形推定
    posxeq,posyeq=top_left
    px,py = find_corresponding_pixel(int(posxeq),int(posyeq),h2,w2,int(w2*3/4))
    print(posxeq,posyeq)
    print(px,py)
    pts_eq.append([posxeq,posyeq])
    print(wmark,hmark)
    coordx,coordy = find_Cubemap_corresponding_pixel(int(px+wmark),int(py),h2,w2,int(w2*3/4))
    pts_eq.append([coordx,coordy])
    coordx,coordy = find_Cubemap_corresponding_pixel(int(px+wmark),int(py+hmark),h2,w2,int(w2*3/4))
    pts_eq.append([coordx,coordy])
    coordx,coordy = find_Cubemap_corresponding_pixel(int(px),int(py+hmark),h2,w2,int(w2*3/4))
    pts_eq.append([coordx,coordy])
    pts_eq=np.array(pts_eq)
    print(pts_eq)
    """
    img=img_CV
    bottom_right = (top_left[0] + wmark, top_left[1] + hmark)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    #cv2.polylines(img,[pts_eq],True,(0,0,1),thickness=10)
    cv2.rectangle(img,top_left, bottom_right, (1,0,0),20)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

def CNNModel(width,height):
    input_layer=Input(shape=(width,height,1))
    hidden_layer1=Conv2D(4,(8,8),activation='relu',padding='same')(input_layer)
    hidden_layer2=Conv2D(4,(8,8),activation='relu',padding='same')(hidden_layer1)
    hidden_layer3=Flatten()(hidden_layer2)
    hidden_layer3=Dropout(0.25)(hidden_layer3)
    #hidden_layer3=Dense(8,activation='relu')(flatten)
    output_layer1=Dense(1)(hidden_layer3)
    output_layer2=Dense(1)(hidden_layer3)
    output_layer3=Dense(1)(hidden_layer3)
    output_layer4=Dense(1)(hidden_layer3)

    adam = Adam(lr=1e-3)
    model=Model(inputs=input_layer,outputs=[output_layer1,output_layer2,output_layer3,output_layer4])
    #model.compile(optimizer=adam, loss='mae',metrics=['mae'])
    return model


def train(TrainData,TrainLabel,width,height,weightsFilename,batch_size=1,nb_epoch=20,weights=None):
    TrainData=TrainData.reshape(TrainData.shape[0],width,height,1)
    adam = Adam(lr=1e-3)
    model=CNNModel(width,height)
    model.compile(optimizer=adam, loss='mae',metrics=['mae'])
    if(weights!=None):
        model.load_weights(weights)
    
    #model.summary()
    history = model.fit(TrainData, TrainLabel, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.2)
    model.save_weights(weightsFilename)

def MatchingByCNN(Eq_Input,Eq_Label=np.zeros(4),weights=None):
    height,width=Eq_Input.shape
    model=CNNModel(width,height)
    expect=np.zeros((4,1,1))
    if(weights!=None):
        model.load_weights(weights)
        Eq_Input=Eq_Input.reshape(1,width,height,1)
        expect=model.predict(Eq_Input,batch_size=1)
    tl=[int(expect[0][0][0]),int(expect[1][0][0])]
    tr=[int(expect[0][0][0]),int(expect[3][0][0])]    
    br=[int(expect[2][0][0]),int(expect[3][0][0])]
    bl=[int(expect[2][0][0]),int(expect[1][0][0])]
    pts=[np.array([tl,tr,br,bl])]

    tl_l=[Eq_Label[0],Eq_Label[1]]
    tr_l=[Eq_Label[2],Eq_Label[1]]
    br_l=[Eq_Label[2],Eq_Label[3]]
    bl_l=[Eq_Label[0],Eq_Label[3]]
    pts_l=[np.array([tl_l,tr_l,br_l,bl_l])]

    
    Eq_Input=Eq_Input.reshape(height,width,1)
    Eq_Input=Eq_Input
    print(pts)
    res=cv2.cvtColor(Eq_Input,cv2.COLOR_GRAY2RGB)
    cv2.polylines(res, pts, True, (0,0,1), thickness=10, lineType=cv2.LINE_8, shift=0)    
    res=cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
    
    plt.imshow(res)
    plt.show()
