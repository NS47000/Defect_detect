from functools import lru_cache
import cv2
import numpy as np
import os,stat
import glob
import subprocess
import argparse
import time
import shutil
batch_file = 'A9Q_show_Mason.bat'
Show_batch_file='A9Q_GA2_show_pattern_YN_0614.bat'
L_RGB=[81,99,53]
R_RGB=[70,70,36]
DxDy=[0,0,0,0]
def offset_x(im_left):
    #im_left=cv2.imread(input_path,1)
    im_left_new=np.zeros_like(im_left)
    for i in range(0,np.shape(im_left_new)[0]):
        for j in range(0,np.shape(im_left_new)[1]):
            offset=0
            if j+3<np.shape(im_left_new)[1]:
                if (j)%3==0:
                    offset=3
                else:
                    offset=0
                im_left_new[i,j]=im_left[i,j+offset]
    return im_left_new
def pullA9Qdata():
    data_path="./A9Q_OE_DATA"
    if os.path.exists(data_path)!=True:
        os.mkdir(data_path)

    os.system("adb pull /sdcard/l_curr ./A9Q_OE_DATA/.")
    os.system("adb pull /sdcard/r_curr ./A9Q_OE_DATA/.")
    os.system("adb pull /sdcard/l_offset ./A9Q_OE_DATA/.")
    os.system("adb pull /sdcard/r_offset ./A9Q_OE_DATA/.")
    
    path = './A9Q_OE_DATA/l_curr'
    with open(path) as f:
        for line in f.readlines():
            s=line.split(" ")
            L_RGB[0]=s[0].strip('\n')
            L_RGB[1]=s[1].strip('\n')
            L_RGB[2]=s[2].strip('\n')
            print("set L_curr:"+str(L_RGB[0])+" "+str(L_RGB[1])+" "+str(L_RGB[2]))
    path = './A9Q_OE_DATA/r_curr'
    with open(path) as f:
        for line in f.readlines():
            s=line.split(" ")
            R_RGB[0]=s[0].strip('\n')
            R_RGB[1]=s[1].strip('\n')
            R_RGB[2]=s[2].strip('\n')
            print("set R_curr:"+str(R_RGB[0])+" "+str(R_RGB[1])+" "+str(R_RGB[2]))
    path = './A9Q_OE_DATA/l_offset'
    with open(path) as f:
        for line in f.readlines():
            s=line.split(" ")
            DxDy[0]=s[0].strip('\n')
            DxDy[1]=s[1].strip('\n')
            print("set l_offset:"+str(DxDy[0])+" "+str(DxDy[1]))
    path = './A9Q_OE_DATA/r_offset'
    with open(path) as f:
        for line in f.readlines():
            s=line.split(" ")
            DxDy[2]=s[0].strip('\n')
            DxDy[3]=s[1].strip('\n')
            print("set r_offset:"+str(DxDy[2])+" "+str(DxDy[3]))
    os.chmod(data_path,stat.S_IWRITE)
    shutil.rmtree(data_path)
    return L_RGB,R_RGB,DxDy
            





def Show_and_LED(im_id, L_RGB, R_RGB):
    
    os.system('kill_all.bat')
    #subprocess.call('run_all.bat {}'.format(im_id) )  #A9Q
    subprocess.call("run_all.bat "+im_id)
    #subprocess.call("adb root")
    #subprocess.call("adb shell /usr/bin/qc2displayfilterunittest --foldername "+im_id+" --format C8_LINEAR")
    time.sleep(7)
    #os.system('showImage.bat {}'.format('White') )  #A9Q
    
    DxDyout=[0,0,0,0]
    try:
        L_RGB,R_RGB,DxDy=pullA9Qdata()
        print("讀取OE參數成功")
        DxDyout=DxDy
    except:
        print("======讀取參數失敗=========")
    Lr=L_RGB[0]
    Lg=L_RGB[1]
    Lb=L_RGB[2]
    Rr=R_RGB[0]
    Rg=R_RGB[1]
    Rb=R_RGB[2]
    print('=======setBacklight.bat {} {} {} {} {} {} ======='.format(Lr,Lg,Lb,Rr,Rg,Rb))
    os.system('setBacklight.bat {} {} {} {} {} {}'.format(Lr,Lg,Lb,Rr,Rg,Rb) )  #A9Q
    time.sleep(2)
    print("====set-dxdy.bat {} {} {} {}=======".format(DxDyout[0],DxDyout[1],DxDyout[2],DxDyout[3]))
    os.system("set-dxdy.bat {} {} {} {}".format(DxDyout[0],DxDyout[1],DxDyout[2],DxDyout[3]))
    

        
    
def Png2RawData(ImagePathR,ImagePathL):
    LR = ['Left', 'Right']
    img_R=cv2.imread(ImagePathR)
    print("load image:"+ImagePathR)
    img_L=cv2.imread(ImagePathL)
    print("load image:"+ImagePathL)
    # split image to 3 channel
    #img_R_new=offset_x(img_R)
    #img_L_new=offset_x(img_L)
    (BR, GR, RR) = cv2.split(img_R)
    (BL, GL, RL) = cv2.split(img_L)
    img_name2_R = ImagePathR.split("\\")[-1].split('.')[-2]
    img_name2_L = ImagePathL.split("\\")[-1].split('.')[-2]
    # create folder
    
    outfolder_L = os.path.join(script_dir,"A9Q_RAW",img_name2_L)
    outfolder_R = os.path.join(script_dir,"A9Q_RAW",img_name2_R)
    if os.path.exists(outfolder_L):
        os.chmod(outfolder_L,stat.S_IWRITE)
        shutil.rmtree(outfolder_L)
    os.makedirs(outfolder_L, exist_ok = True)

    #os.makedirs(outfolder_L, exist_ok = True) 
    H, W = BL.shape
    print("H:"+str(H)+"W:"+str(W))
    print(str(outfolder_L)+ r'\{}_r_1440X1080.raw'.format(LR[0]))
    RL.tofile(outfolder_L + r'\{}_r_1440X1080.raw'.format(LR[0]))       
    GL.tofile(outfolder_L + r'\{}_g_1440X1080.raw'.format(LR[0]))        
    BL.tofile(outfolder_L + r'\{}_b_1440X1080.raw'.format(LR[0]))
    RR.tofile(outfolder_L + r'\{}_r_1440X1080.raw'.format(LR[1]))       
    GR.tofile(outfolder_L + r'\{}_g_1440X1080.raw'.format(LR[1]))        
    BR.tofile(outfolder_L + r'\{}_b_1440X1080.raw'.format(LR[1]))  
    return img_name2_L

def GetRawData():
    script_dir =  os.path.abspath(os.path.dirname(__file__))
    image_list = (glob.glob(r".\\input\\*.png"))   
    #image_list = ['Cross_1080_1440.bmp']
    LR = ['Left', 'Right']
    for img_name in image_list:
        # read image and cut
        img = cv2.imread(img_name)        
        print("load image:"+img_name)
        # split image to 3 channel
        img_new=offset_x(img)
        (B, G, R) = cv2.split(img_new)
        
        if ('_right'in img_name) or ('_left'in img_name):
            img_folder_name = img_name.split("\\")[-1].split('.')[-2].split('_')[-2]
            img_name2=img_name.split("\\")[-1].split('.')[-2]
        # create folder
        
        outfolder = os.path.join(script_dir,"A9Q_RAW",img_folder_name)
        os.makedirs(outfolder, exist_ok = True)   
    
        H, W = B.shape

        #for lr in LR:
        if '_right' in img_name2:
            lr=LR[1]   
            R.tofile(outfolder + r'\{}_r_1440X1080.raw'.format(lr))       
            G.tofile(outfolder + r'\{}_g_1440X1080.raw'.format(lr))        
            B.tofile(outfolder + r'\{}_b_1440X1080.raw'.format(lr))
        elif '_left' in img_name2:
            lr=LR[0]   
            R.tofile(outfolder + r'\{}_r_1440X1080.raw'.format(lr))       
            G.tofile(outfolder + r'\{}_g_1440X1080.raw'.format(lr))        
            B.tofile(outfolder + r'\{}_b_1440X1080.raw'.format(lr)) 
        else:
            print("error imagename:"+img_name2)

    return img_folder_name

def Close_Image():
    os.system('{} E'.format(Show_batch_file))
def process_command():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--ImagePathR','-R',default='none',required=False, type=str, help='input R image path')
    arg_parser.add_argument('--ImagePathL','-L',default='none',required=False, type=str, help='input L image path')
    return arg_parser.parse_args()

if __name__ == '__main__':
    args = process_command()
    script_dir =  os.path.abspath(os.path.dirname(__file__))
    if (args.ImagePathR=='none') or (args.ImagePathL=='none'):
        img_name2=GetRawData()
    else:
        img_name2=Png2RawData(args.ImagePathR,args.ImagePathL)     
    print(batch_file+" P "+os.path.join(script_dir,"A9Q_RAW")+"\\ "+img_name2)
    os.system(batch_file+" P "+os.path.join(script_dir,"A9Q_RAW")+"\\ "+img_name2)
    print("==============push end==================")
    print("image2:"+img_name2)
    time.sleep(1)
    start_time=time.time()
    print("start showing undistort checkboard")
    Show_and_LED(img_name2,L_RGB,R_RGB)
    print(img_name2)  
    time.sleep(1)
    
    end_time=time.time()
    print(end_time-start_time)
    print("end showing undistort checkboard")
    #os.system('{} E'.format(Show_batch_file))


