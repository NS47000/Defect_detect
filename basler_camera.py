

# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:50:13 2021
`
@author: 10806213
"""
#from pypylon import genicam
from pypylon import pylon
import sys
import getopt
import os 

import cv2
import numpy as np
import time
from datetime import datetime
import threading
import subprocess
from subprocess import run, PIPE
from queue import Queue

import csv

#for QARGC
#import screeninfo

emulation = False
Lumus_exp_set = True

ghost_switch = False
Distortion_switch = False
script_dir =  os.path.abspath(os.path.dirname(__file__))
PROJECT_NAME = 'A9Q' # A9Q or QAR7GA2 or QAR7GA2MQ or QAR7GC
TEST_MODE = False # Set true when debugging without jig camera
if PROJECT_NAME == 'A9Q':
    Show_batch_file = 'A9Q_GA2_show_pattern_YN_0614.bat' #justin: p = push, U = show
    batch_file = 'A9Q_GA2_IQT_batch_file.bat'
elif PROJECT_NAME == 'QAR7GA2':
    Show_batch_file = 'GA2_show_pattern_YN_0614.bat' #justin: p = push, U = show
    batch_file = 'GA2_IQT_batch_file.bat'#justin: push files by calling Show_batch_file several times
elif PROJECT_NAME == 'QAR7GA2MQ':
    Show_batch_file = 'GA2MQ_show_pattern_YN_0614.bat' #justin: U = push and show; p = push
    batch_file = 'GA2MQ_IQT_batch_file.bat'#justin: push files by calling Show_batch_file several times

class Capture_function:
    def __init__(self, SN, current_times, station, Mono):
        # image folder neme        
        self.contrast_cb_name = ['ContrastW','ContrastB']
        self.blemish_name = ['White_4px_30gap','White_4px_50gap', 'Black_4px_30gap', 'Black_4px_50gap']
        self.WB_name = ['White_blemish', 'Black_blemish']
        self.MTF_name = ['MTFH','MTFV']
        self.Ghost_name = ['Ghost']
        self.Distortion_name = ['Distortion']
        
        self.Straylight= [ 'straylight_test_255', 'straylight_test_50', 'straylight_test_30', 'straylight_test_10']

        self.Cross_name = ['Cross']
        self.defect=['img_ori_1.png','img_ori_2.png','img_ori_3.png','img_ori_4.png']
        self.defect_white=['img_ori_1_white.png','img_ori_2_white.png','img_ori_3_white.png','img_ori_4_white.png']
        
        # image_exp_set
        self.SN = SN
        self.current_times = current_times
        self.station = station
        self.items_exp_list = []       
        self.outfolder = []
        self.base = 16667    # exp. init
        
        if Lumus_exp_set:
            self.base = 8346    # exp. init 8280 or 8350 or 8346 or 8260 or 8333
        
        # camera set
        self.cam_width = 4024
        self.cam_height = 3036
        self.Mono = Mono
        self.devices_number = 2


        # camera SN
        # Basler 8mm
        if station == 'IQT':
            #L_cam = '23370800' 
            R_cam = '24261885' 
            
            # A9Q EVT1.2 turned off
            #L_cam = '23370803' 
            #R_cam = '23370800'

            # lent from Ted
            # L_cam = '24261885'
            # R_cam = '24261884'



        # Basler 16mm(MTF)
        elif station =='MTF':
            L_cam = '22657061'
            R_cam = '22657059'
            
            
        
        # camera connect
        #try:
        # Get the transport layer factory.
        tlFactory = pylon.TlFactory.GetInstance()
    
        # Get all attached devices and exit application if no device is found.
        devices = tlFactory.EnumerateDevices()
        self.devices_number = len(devices)
        
        if self.devices_number == 0:
            raise pylon.RuntimeException("No camera present.")
    
        # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
        self.cameras = pylon.InstantCameraArray(self.devices_number)
    
        #l = cameras.GetSize()    
        
        new_exp_list=[]
        # Create and attach all Pylon Devices.
        
        
        self.cameras[0].Attach(tlFactory.CreateDevice(devices[0]))
        #self.cameras[1].Attach(tlFactory.CreateDevice(devices[1]))
        
        cam0_sn = self.cameras[0].GetDeviceInfo().GetSerialNumber()
        #cam1_sn = self.cameras[1].GetDeviceInfo().GetSerialNumber()
        cam0_sn == R_cam
        self.cam0_side = 'camr'
        """
        if cam0_sn == L_cam:
            self.cam0_side = 'caml'
        elif cam0_sn == R_cam:
            self.cam0_side = 'camr'
        else:
            print("Please check camera 0 SN")
            print("connected sn {0}".format(cam0_sn))
        """

        print("Using device 0: ", cam0_sn)
        
        self.cameras[0].Open()            
        self.cameras[0].PixelFormat = self.Mono
        
        if(not emulation):
            self.cameras[0].Width.SetValue(self.cam_width)
            self.cameras[0].Height.SetValue(self.cam_height)
            self.cameras[0].Gain.SetValue(0.0)
            self.cameras[0].Gamma.SetValue(1.0)
        
        """
        if cam1_sn == L_cam:
            self.cam1_side = 'caml'
        elif cam1_sn == R_cam:
            self.cam1_side = 'camr'
        else:
            print("Please check camera 1 SN")
            print("connect sn {0}".format(cam1_sn))


        print("Using device 1: ", cam1_sn)
        
        self.cameras[1].Open()            
        self.cameras[1].PixelFormat = self.Mono
        
        if(not emulation):
            self.cameras[1].Width.SetValue(self.cam_width)
            self.cameras[1].Height.SetValue(self.cam_height)
            self.cameras[1].Gain.SetValue(0.0)
            self.cameras[1].Gamma.SetValue(1.0)
        """
        print("Basler complete initialization!")
        """          
        except Exception as e:
            try:
                self.cameras[0].Close()
                self.cameras[1].Close()
            except:
                pass
            print(e)
            print("請檢查或是重插拔治具相機.")
            os.system("pause")
            os.exit()
            """      
        



    def image_exp_set(self):
        # Create Capture folder    
        curr_path = os.getcwd()
        outfolder = os.path.join(curr_path, 'Capture' , self.SN, self.current_times)
        os.makedirs(outfolder, exist_ok = True)    
        

    # =============================================================================
    #     white_exp_times_list = [1,2,4,8,16,32]
    #     black_exp_times_list = [64,128,256,512]
    #     ghost_exp_times_list = [8] # 66.666ms
    #     distort_exp_times_list = [64]#[64] # 66.666ms
    #     contrast_exp_times_list = [1,2,4,8,16,32,64,128]
    #     #contrast_exp_times_list = [1,3,9,27,81,243]
    # =============================================================================
        
        if self.station == 'IQT':
            #Justin: should be the same with A9Q_IQT_run_All_items_211115_myPC_combine_straylight_fullContrast.py line 86

            white_exp_times_list = [1,3,9]#,27 ,81
            black_exp_times_list = [27,81,243]
            contrast_exp_times_list = [1,3,9,27]# ,81,243
            distort_exp_times_list = [27]
            ghost_exp_times_list = [9]  
            defect_exp_list=[4]
            defect_exp_list=[300]
            

            # Lumus setting
            if Lumus_exp_set:
                #A93, A9Q EVT1.2, A9Q DVT1.0 (self.base = 8333)
                white_exp_times_list = [1,4]#,27 ,81
                black_exp_times_list = [24,120,240]#
                contrast_exp_times_list = [1,6,48]# ,81,243
                distort_exp_times_list = [48]
                ghost_exp_times_list = [48]
                
                # A9Q DVT1.1 proposed
                white_exp_times_list = [2,4,6]
                black_exp_times_list = [300, 600, 1198]
                contrast_exp_times_list = [1,6,48]
                
            
            white_exp_list = []
            black_exp_list = []
            contrast_exp_list = []
            distortion_exp_list = []
            ghost_exp_list = []    
            defect_exp_list=[]
            defect_exp_white_list=[]
                            
            
            for i in white_exp_times_list:
                white_exp_list.append(self.base * i)
                
            for i in black_exp_times_list:
                black_exp_list.append(self.base * i)        
                
            for i in contrast_exp_times_list:
                contrast_exp_list.append(self.base * i)    
            for i in distort_exp_times_list:
                distortion_exp_list.append(self.base * i)
            
            for i in ghost_exp_times_list:
                ghost_exp_list.append(self.base * i)
            defect_exp_list.append(self.base*4) 
            defect_exp_white_list.append(self.base*300)                                 
    
            self.items_exp_list = {'White':white_exp_list, 'Black':black_exp_list,'Contrast_cb':contrast_exp_list, 'Distortion':distortion_exp_list, 'Ghost':ghost_exp_list}   
        
        elif self.station == 'MTF':
            mtf_exp_times_list = [1,2,4,8,16]
            
            # Lumus setting
            if Lumus_exp_set:
                mtf_exp_times_list = [1,4]
            

            mtf_exp_list = []

            for i in mtf_exp_times_list:
                mtf_exp_list.append(self.base * i)

            self.items_exp_list = {'MTF':mtf_exp_list}       

        self.outfolder = outfolder
            
        
        '''
        mtf_exp_times_list = [1,2,4,8,16,32]
        ghost_exp_times_list = [9]
        distort_exp_times_list = [81]
        
        mtf_exp_list = []
        ghost_exp_list = []
        distort_exp_list = []

        for i in mtf_exp_times_list:
            mtf_exp_list.append(base * i)
                        
        for i in ghost_exp_times_list:
            ghost_exp_list.append(base * i)
            
        for i in distort_exp_times_list:
            distort_exp_list.append(base * i)

        self.mtf_exp_list = mtf_exp_list
        self.ghost_exp_list = ghost_exp_list
        self.distort_exp_list = distort_exp_list
        '''

    

    def capture_item(self, item_name, deco_term = ''):    
        
        start_item = time.time()

        
        print('item:' + str(item_name))
        
        
# =============================================================================
#         contrast_cb_name = ['ContrastW','ContrastB']
#         blemish_name = ['White_4px_30gap','White_4px_50gap', 'Black_4px_30gap', 'Black_4px_50gap']
#         WB_name = ['White_blemish', 'Black_blemish']
#         MTF_name = ['MTFH','MTFV']
# =============================================================================
        
        # check contrast_cb
        if item_name in self.contrast_cb_name:
            Mono_set = 'Mono12' 
            muti = 1
            blemish_item = ''
            
            item_exp_info = self.items_exp_list.get('Contrast_cb')

        elif item_name in self.blemish_name:
            Mono_set = 'Mono12'
            muti = 1
            blemish_item = 'Blemish'
            
            if item_name.split('_')[0] == 'White':
                item_exp_info = self.items_exp_list.get('White')
            else:
                item_exp_info = self.items_exp_list.get('Black')
        
        elif item_name in self.WB_name:
            Mono_set = 'Mono12'
            muti = 1
            blemish_item = ''
            
            if item_name.split('_')[0] == 'White':
                item_exp_info = self.items_exp_list.get('White')
            else:
                item_exp_info = self.items_exp_list.get('Black')
                
        elif item_name in self.MTF_name:
            Mono_set = 'Mono8'
            muti = 1
            blemish_item = ''
            item_exp_info = self.items_exp_list.get('MTF')
            
        elif item_name in self.Ghost_name:
            Mono_set = 'Mono8'
            muti = 1
            blemish_item = ''
            item_exp_info = self.items_exp_list.get('Ghost')
        
        elif item_name in self.Distortion_name:
            Mono_set = 'Mono8'
            muti = 1
            blemish_item = ''
            item_exp_info = self.items_exp_list.get('Distortion').copy()
            
            if (deco_term.find('green') == -1) :
                if (deco_term.find('white') != -1) :
                    item_exp_info[0] = item_exp_info[0]//3
                else:
                    item_exp_info[0] = item_exp_info[0]*3
            
        elif item_name in self.Straylight:
            Mono_set = 'Mono8'
            muti = 1
            blemish_item = ''
            item_exp_info = self.items_exp_list.get('Black')
            
        elif item_name in self.Cross_name:
            Mono_set = 'Mono8'
            muti = 1
            blemish_item = ''
            item_exp_info = self.items_exp_list.get('Contrast_cb') # change from Black to Contrast_cb to save time
        elif item_name in self.defect:
            Mono_set = 'Mono8'
            muti = 1
            blemish_item = ''
            item_exp_info = self.base*4
        elif item_name in self.defect_white:
            Mono_set = 'Mono8'
            muti = 1
            blemish_item = ''
            item_exp_info = self.base*300
            
        else:
            print('Error: Non-define item name.')
            self.close_cameras()
            sys.exit()                                  
            
        self.sync_capture(item_exp_info, os.path.join(script_dir,item_name), deco_term =deco_term, muti = muti, Mono = Mono_set)
        


        endt = time.time()
        print('Time(s):' + str(endt-start_item))

        print('+Time(s):' + str(endt-start))


    def sync_capture(self, exp, Outfolder, deco_term = '', muti = 1, Mono = "Mono8"):
        os.makedirs(Outfolder, exist_ok = True)            
            
        new_exp_list=[]

        # ========================= 3 channel ====================================================
        # =============================================================================
        #         converter = pylon.ImageFormatConverter()
        #         
        #         # converting to opencv bgr format
        #         converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        #         converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        # 
        # =============================================================================
                
        # Check camera Mono
        if self.Mono != Mono:
            self.change_pixel_format(Mono)
        
        
        # e.x Outfolder = r'C:\Users\10806213\Desktop\112233'                
        img_save_folder = Outfolder
            
        #for j, exp in enumerate(exp_list):
        if not muti == 1:                     
            os.makedirs(Outfolder + '\\'+str(exp), exist_ok = True)            
        elif Outfolder.split('\\')[-1] in self.WB_name or Outfolder.split('\\')[-1] in self.blemish_name:
            os.makedirs(Outfolder + '\\'+str(exp), exist_ok = True)            
        
        '''
        if (not emulation):            
            last0_exp = self.cameras[0].ExposureTime.GetValue()# Get exp. value

            if(last0_exp < exp):
                last0_exp = exp
                
            if(last0_exp < 66666):
                last0_exp = 66666
            
            self.cameras[0].ExposureTime.SetValue(exp)
            self.cameras[1].ExposureTime.SetValue(exp)
            
            new_exp_list.append(self.cameras[0].ExposureTime.GetValue())
            new_exp_list.append(self.cameras[1].ExposureTime.GetValue())
            
            wait_time = 3*(last0_exp/1000000)
            
            time.sleep(wait_time)
        else:
            time.sleep(1)    
        '''

        self.change_exposure_time(exp)

        for n in range(0, muti):
            print('Capture image : {} _{}'.format(Outfolder, str(n)))
            
            grab0Result_successful = False
            grab1Result_successful = False
            grab0Result = None
            grab1Result = None
            try_times = 0
            while(not grab0Result_successful):
                try_times = try_times + 1
                print("Try to capture at " + str(try_times) + " times.")
                # Wait for an image and then retrieve it. A timeout of 2000 ms is used.
                grab0Result = self.cameras[0].RetrieveResult(15000, pylon.TimeoutHandling_ThrowException) # 2000 ms   
                #grab1Result = self.cameras[1].RetrieveResult(15000, pylon.TimeoutHandling_ThrowException) # 2000 ms   
                
                grab0Result_successful = grab0Result.GrabSucceeded()
                #grab1Result_successful = grab1Result.GrabSucceeded()  
            
            if grab0Result.GrabSucceeded():# and grab1Result.GrabSucceeded()
                    
    # ===============================3 channel ==============================================
    # =============================================================================
    #                 image = converter.Convert(grabResult)
    #                 img = image.GetArray()
    # =============================================================================
                    
                img0 = grab0Result.GetArray()
                #img1 = grab1Result.GetArray()
                
                
                
                file_deco = ''
                if deco_term == '' :
                    file_deco = file_deco
                else:
                    file_deco = file_deco + '_' + deco_term
                    
                    
                if Outfolder.split('\\')[-1] in self.WB_name or Outfolder.split('\\')[-1] in self.blemish_name:
                    file_deco = file_deco + '_n' + str(n)
                    img_save_folder = os.path.join(Outfolder,str(exp))
                elif muti == 1 :
                    file_deco = file_deco
                else:
                    file_deco = file_deco + '_n' + str(n)
                    img_save_folder = os.path.join(Outfolder,str(exp))
                
                
                if Mono == "Mono12":
                    
                    img0_hb = (img0 // 16).astype(np.uint8)
                    img0_lb = (img0 % 16).astype(np.uint8)
                    
                    img1_hb = (img1 // 16).astype(np.uint8)
                    img1_lb = (img1 % 16).astype(np.uint8)
                    
                    cv2.imwrite(img_save_folder + r"\{}_exp{}{}_hb.png".format(self.cam0_side, file_deco), img0_hb)
                    cv2.imwrite(img_save_folder + r"\{}_exp{}{}_lb.png".format(self.cam0_side, file_deco), img0_lb)
                    
                    #cv2.imwrite(img_save_folder + r"\{}_exp{}{}_hb.png".format(self.cam1_side, file_deco), img1_hb)
                    #cv2.imwrite(img_save_folder + r"\{}_exp{}{}_lb.png".format(self.cam1_side, file_deco), img1_lb)
                else:
                    cv2.imwrite(img_save_folder + r"\{}_exp{}{}.png".format(self.cam0_side, file_deco), img0)
                    #cv2.imwrite(img_save_folder + r"\{}_exp{}{}.png".format(self.cam1_side, file_deco), img1)


                time.sleep(1)    
        if (not emulation):    
            self.cameras[0].ExposureTime.SetValue(self.base)
            #self.cameras[1].ExposureTime.SetValue(self.base)

        
        return new_exp_list

    def change_exposure_time(self, exp):
        print("Change exposure time: {}".format(exp))
        try:
            # Stop the grabbing.
            self.cameras[0].StopGrabbing()
            #self.cameras[1].StopGrabbing()
    
            
            self.cameras[0].ExposureTime.SetValue(exp)
            #self.cameras[1].ExposureTime.SetValue(exp)
            
            self.start_grabbing()
            
        except Exception as e:
            self.cameras[0].Close()
            #self.cameras[1].Close()  
            print(e)


    # turn on cameras
    def start_grabbing(self):
        try:
            self.cameras[0].StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            #self.cameras[1].StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print("Basler start grabbing")
            
        except Exception as e:
            self.cameras[0].Close()
            #self.cameras[1].Close()   
            print(e)
        

    def change_pixel_format(self, Mono):
        print("Change pixel format")
        try:
            # Stop the grabbing.
            self.cameras[0].StopGrabbing()
            #self.cameras[1].StopGrabbing()
            
            self.Mono = Mono
            
            self.cameras[0].PixelFormat = self.Mono
            #self.cameras[1].PixelFormat = self.Mono
            
            self.start_grabbing()
            
        except Exception as e:
            self.cameras[0].Close()
            #self.cameras[1].Close()  
            print(e)




    def close_cameras(self):
        self.cameras[0].Close()
        #self.cameras[1].Close()  
        print("Close Basler cameras")



# not use
def oneCAM_capture_test(exp_list, Outfolder, deco_term = '', camLR = 'caml'):    
    os.makedirs(Outfolder, exist_ok = True)

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    camera.GetDeviceInfo().GetModelName()
    camera.GetDeviceInfo().GetSerialNumber() 


    camera.PixelFormat = "Mono8"
    # set camera size
    camera.Width.SetValue(4024)
    camera.Height.SetValue(3036)
    camera.Gain.SetValue(-0.00000)
    camera.Gamma.SetValue(1.0)



    # Grabing Continusely (video) with minimal delay
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
# =============================================================================
#     converter = pylon.ImageFormatConverter()
#     
#     # converting to opencv bgr format
#     converter.OutputPixelFormat = pylon.PixelType_BGR8packed
#     converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
# =============================================================================
    
    
    # Set the pixel data format.
    for i, exp in enumerate(exp_list):
        print("Capture image : "+ Outfolder + " " + str(i))
        camera.ExposureTime.SetValue(exp);
        time.sleep(1)
        
        grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException) # 1000 ms
        if grabResult.GrabSucceeded():
# =============================================================================
#             image = converter.Convert(grabResult)
#             img = image.GetArray()
# =============================================================================
            img = grabResult.GetArray()
            cv2.imwrite(Outfolder + r"\{}_exp{}.png".format(camLR, exp), img)
            #cv2.imwrite(Outfolder + r"\{}_exp{}.raw".format(camLR, exp), img)

    camera.Close()   
    
    
def Check_device_open( check=False):
    txt = 'device_trun_on.txt'

    # os.system('adb shell getprop dev.bootcomplete > ' + txt)  #A93
    # os.system('adb wait-for-device)  #A93
    f = open(txt, 'r')
    device_trun_on = f.read()
    f.close()

    
    txt2 = 'capture_finish_info.txt'
    f = open(txt2, 'w')
    f.write('device_connect')
    f.close()

    if check == True and device_trun_on == '':
        f = open(txt2, 'w')
        f.write('device_disconnect')
        f.close()
    else:
        f = open(txt2, 'w')
        f.write('device_connect')
        f.close()

    return device_trun_on

    
def print_help():
    return


def showim(window_name,img):
    '''
    For QAR7GC, windows extension
    Reference: https://gist.github.com/ronekko/dc3747211543165108b11073f929b85e
    '''

    subprocess.call('QCI_HMD_all_in_one.exe s 11 5f 0') # turn off auto-adjust-brightness function

    #cv2.destroyAllWindows()
    screen_id = 1
    # get the size of the screen
    screen = screeninfo.get_monitors()[screen_id]
    width, height = screen.width, screen.height

    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, img)
    cv2.waitKey(500)
    #cv2.destroyAllWindows()
    hello = 0

def Show(im):
    os.system('{} U {}'.format(Show_batch_file, im) )  #QAR7GA2
def dxdy():
    os.system('set_dxdy_default.bat')  #A9Q
    
def Show_and_LED(im_id, L_RGB, R_RGB):
    
    # os.system('adb reboot')
    # os.system('adb wait-for-device')
    os.system('kill_all.bat')
    subprocess.call('run_all.bat {}'.format(im_id) )  #A9Q
    time.sleep(7)
    #os.system('showImage.bat {}'.format('White') )  #A9Q
    
    Lr=L_RGB[0]
    Lg=L_RGB[1]
    Lb=L_RGB[2]
    Rr=R_RGB[0]
    Rg=R_RGB[1]
    Rb=R_RGB[2]
    
    os.system('setBacklight.bat {} {} {} {} {} {}'.format(Lr,Lg,Lb,Rr,Rg,Rb) )  #A9Q
        
    time.sleep(2)

def Check_device(time):

    if PROJECT_NAME == 'QAR7GC':
        return True #TODO

    q = Queue()      
    
    p=threading.Thread(target=check, args=(q,))
    p.start()
    p.join(10)
    
    result = q.get()
    return result
    
def check(q0):
    connect = False
    os.system('adb wait-for-device')
    q0.put(True)
    return connect

def Get_LedRGB(side):
    if side == 'L':
        LR = 'l_curr'
    elif side =='R':
        LR = 'r_curr'
    print(LR)
    try:
        result = run("adb shell cat sdcard/{}".format(LR), stdout=PIPE, stderr=PIPE, check=True, universal_newlines=True)
        print('1')
        RGB_list = result.stdout.split('\n')[0]
        print(RGB_list)
        RGB_list = RGB_list.split(' ')
        print(RGB_list)
    except Exception as e:
       
       print("An error occured:")
       print(e)
       print('2:fail')
       return [0, 0, 0]
   
    if not len(RGB_list) == 3:
        print('3:fail')
        return [0, 0, 0]
    
    
    return(RGB_list)
    #print(result.stdout)
    
def read_CCT_data(device_SN):
    with open('A9Q_CCT_Calibration_backup_202207112120.csv', newline='') as csvfile:
    # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
        rows = csv.DictReader(csvfile)
        for row in rows:
            if (row['Devise SN'] == device_SN):
                L_r = row['L_R_current']
                L_g = row['L_G_current']
                L_b = row['L_B_current']
                R_r = row['R_R_current']
                R_g = row['R_G_current']
                R_b = row['R_B_current']    
                print(L_r, L_g, L_b, R_r, R_g, R_b)
            
        try:            
            return [L_r, L_g, L_b], [R_r, R_g, R_b]
        except Exception as e:
            return [0, 0, 0], [0, 0, 0]


if __name__ == '__main__':



    #justin test for MQ
    # start = time.time()
    # SN = 'MQ_demo_kit'    
    # current_times = '20230217'
    # Cap_F = Capture_function(SN, current_times, 'IQT', Mono ="Mono8")
    # Cap_F.start_grabbing()        
    # Cap_F.image_exp_set()
    # #Cap_F.capture_item('White_blemish')
    # Cap_F.capture_item('Black_blemish')
    # hello = 0

   


    #time.sleep(2)
    #Show_and_LED('Black_1080_1440')  
    #Show_and_LED('CB3X3A_1080_1440')  
    #sys.exit(2)
    
    # push test pattern to device
    #os.system('{} P'.format(batch_file))
    #time.sleep(1) # s
    '''
    SN='QA9Q220729011'
    print(SN)
    # Get RGB current
    L_RGB, R_RGB =read_CCT_data(SN)
    print (L_RGB[1])
    if L_RGB[0]==0 and L_RGB[1]==0:  
        print('T')
        L_RGB = Get_LedRGB('L')
        R_RGB = Get_LedRGB('R')
    
    print (L_RGB)
    #print (L_RGB[0], L_RGB[1], L_RGB[2], R_RGB[0], R_RGB[1], R_RGB[2])
    #Show_and_LED('FSAT_dots', L_RGB, R_RGB) #A9Q CBCW

    sys.exit(2)
    '''

    #showim('2nd',cv2.imread('Resoure_data/GA2MQ/CB3X3A_1440_1440.png'))


    start = time.time()
    #     "python IQT_ADB_Basler_capture_210901_NEW.py -s 12321 -c 20210916093925"

# =============================================================================
#     SN = 'SN001_IQT_test_3'    
#     current_times = '20210608151109'
# =============================================================================
    

    '''
                "args": ["-s", "221252003_cap",
                    "-c", "202301042146"]
    '''
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], 's:c:', ["sn=", "capture_time="])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)


    
 
    
# =============================================================================
#     SN = 'SN014'    
#     current_times = '20211012_2'
# =============================================================================


    for opt, arg in opts:
        if opt == '-h':
            print_help()
            sys.exit()
        elif opt in ("-s", "--sn"):
            SN = arg
        elif opt in ("-c", "--capture_path"):
            current_times = arg

# =============================================================================
#     SN = '123456'    
#     current_times = '20210608151109'
# =============================================================================

    #print("device_sn: "+SN)
    #print("current_times: "+current_times)
  
       
# =============================================================================
#     Item = ['ContrastW', 'ContrastB',
#             'White_blemish', 'Black_blemish','white_4px_30gap', 'white_4px_50gap', 'black_4px_30gap', 'black_4px_50gap',
#             'MTF2H', 'MTF2V']
# =============================================================================
    
    ''' 
    #A93 RGB current 
    os.system('set-current.bat L 20 20 20')# 20 20 20
    time.sleep(1)
    os.system('set-current.bat R 20 20 20')
    time.sleep(1)

    if Lumus_exp_set:
        os.system('set-current.bat L 152 130 36')# 20 20 20
        time.sleep(1)
        os.system('set-current.bat R 152 130 36')
        time.sleep(1)
    '''
  
    if not TEST_MODE:
        # turn on camera and init capture image exp.
        print('Init capture setting')
        cam_set_start = time.time()
        
        Cap_F = Capture_function(SN, current_times, 'IQT', Mono ="Mono8")
        
        Cap_F.start_grabbing()        

        Cap_F.image_exp_set()

        cam_set_end = time.time()
        print('Init capture setting time(s):' + str(cam_set_end-cam_set_start))
    
    
# =============================================================================
#     os.system('{} p black_4px_30gap_96value_2048_2048_90.bmp IQT'.format(Show_batch_file))    
#     os.system('{} p black_4px_50gap_96value_2048_2048_90.bmp IQT'.format(Show_batch_file))    
# =============================================================================
    #time.sleep(2) # s
    

    # Wait device turn on
    '''
    device_open_start = time.time()
    device_trun_on = 1
    t=10
    txt = 'device_trun_on.txt'
    while device_trun_on == 1 or device_trun_on == '':
        device_trun_on = Check_device_open()
        if device_trun_on == '1\n': break

        device_open_end = time.time()
        if (device_open_end-device_open_start) > 40:
            print('time out')
            f = open('test.txt', 'a+')
            f.write('time out\n')
            f.close()
            sys.exit()
        else:
            time.sleep(t) # s
            device_open_end = time.time()
            if (device_open_end-device_open_start) > 25: t=1
    '''
    
    # A9Q closed
    if ghost_switch:
        ghost_charts = []
        ghost_charts.append('4K_A93_ghost_256px_A11.png')
        ghost_charts.append('4K_A93_ghost_256px_A12.png')
        ghost_charts.append('4K_A93_ghost_256px_A13.png')
        ghost_charts.append('4K_A93_ghost_256px_A14.png')
        ghost_charts.append('4K_A93_ghost_256px_A21.png')
        ghost_charts.append('4K_A93_ghost_256px_A22.png')
        ghost_charts.append('4K_A93_ghost_256px_A23.png')
        ghost_charts.append('4K_A93_ghost_256px_A24.png')
        ghost_charts.append('4K_A93_ghost_256px_A31.png')
        ghost_charts.append('4K_A93_ghost_256px_A32.png')
        ghost_charts.append('4K_A93_ghost_256px_A33.png')
        ghost_charts.append('4K_A93_ghost_256px_A34.png')
        ghost_charts.append('4K_A93_ghost_256px_A41.png')
        ghost_charts.append('4K_A93_ghost_256px_A42.png')
        ghost_charts.append('4K_A93_ghost_256px_A43.png')
        ghost_charts.append('4K_A93_ghost_256px_A44.png')
    # A9Q closed
    if Distortion_switch:
        distortion_charts = []
        distortion_charts.append('dist_blue_h_center.png')
        distortion_charts.append('dist_blue_h_lines.png')
        distortion_charts.append('dist_blue_v_center.png')
        distortion_charts.append('dist_blue_v_lines.png')
        distortion_charts.append('dist_green_h_center.png')
        distortion_charts.append('dist_green_h_lines.png')
        distortion_charts.append('dist_green_v_center.png')
        distortion_charts.append('dist_green_v_lines.png')
        distortion_charts.append('dist_red_h_center.png')
        distortion_charts.append('dist_red_h_lines.png')
        distortion_charts.append('dist_red_v_center.png')
        distortion_charts.append('dist_red_v_lines.png')
        distortion_charts.append('dist_white_3x3_dot.png')
        distortion_charts.append('dist_white_prewarp_grid.png')
                                                    
    # Put images - (210917 change to IQT_batch_file.bat)
# =============================================================================
#     os.system('{} p black_4px_30gap_96value_2048_2048_90.png IQT'.format(Show_batch_file))    
#     os.system('{} p black_4px_50gap_96value_2048_2048_90.png IQT'.format(Show_batch_file))
# =============================================================================
        

    #time.sleep(2) # s
    
    ''' A93
    # Check oe app trun on/off
    print('Check oe app turn on/off')
    txt = 'current_app.txt'
    os.system('adb shell dumpsys window | findstr mCurrentFocus >' + txt)
    f = open(txt, 'r')
    line = f.read()
    f.close()
    if 'com.example.oe' in line:
        print('show white image(change)')
        os.system('{} s White_2048_2048.bmp IQT'.format(Show_batch_file))    
        time.sleep(1) # s

    else:
        # White image
        image_open_start = time.time()
        print('show white image(first)')
        os.system('{} o White_2048_2048.bmp IQT'.format(Show_batch_file))    
        image_open_end = time.time()
        
        # wait image open
        t=13
        print('wait({}s) image open'.format(t))
        time.sleep(t) # s
# =============================================================================
#         t=3
#         while (image_open_end-image_open_start) < 15:
#             time.sleep(t) # s
#             image_open_end = time.time()
#             if (image_open_end-image_open_start) > 5: t=1
# =============================================================================
    '''
    
    #Justin: push all images here. We pushed images at MTF station. The following line is ignored on purpose.
    #os.system('{} P'.format(batch_file))
    
    Device_open_shitch = False
    Check_device_open_shitch = True
    
    t=1 # wait show pattern time
        
    check_device_bool = Check_device(40) 
    if not check_device_bool:
        print('device_not_open')
        sys.exit()
    
    if PROJECT_NAME == 'A9Q':
        # Get RGB current
        L_RGB, R_RGB =read_CCT_data(SN)
        if L_RGB[0]==0 and L_RGB[1]==0:    
            L_RGB = Get_LedRGB('L')
            R_RGB = Get_LedRGB('R')
        #print L_RGB[0], L_RGB[1], L_RGB[2]
        #sys.exit(2)
        
        if L_RGB[0]==0 and L_RGB[1]==0:    
            L_RGB=[123,110,46]
            R_RGB=[123,110,46]#[120,110,50]

            #QA9Q230101042
            # L_RGB=[95,110,54]
            # R_RGB=[61,71,37]
        
        #L_RGB=[180,161,67]
        #R_RGB=[121,103,39]#[120,110,50]
        print("Use left and right current", L_RGB, R_RGB)


    time.sleep(6)
    # ContrastW
    # os.system('{} s CB3X3B_1080_1440.bmp'.format(Show_batch_file))#A93 CBCW

    first_image_time = time.time()
    print('first_image_time(s):' + str(first_image_time - start))

    if PROJECT_NAME == 'A9Q':
        Show_and_LED('CB3X3B_1080_1440', L_RGB, R_RGB) #A9Q CBCW
    elif PROJECT_NAME == 'QAR7GC':
        showim('2nd',cv2.imread('Resoure_data/GA2MQ/CB3X3B_1440_1440.png'))
    else:
        Show('CBCW')
    time.sleep(2) # s
    Cap_F.capture_item('ContrastW')

    
    # re-capture first image. Justin: It's a workaround. We will consider to delete it.
    if PROJECT_NAME == 'A9Q':
        Show_and_LED('CB3X3B_1080_1440', L_RGB, R_RGB) #A9Q CBCW
    elif PROJECT_NAME == 'QAR7GC':
        showim('2nd',cv2.imread('Resoure_data/GA2MQ/CB3X3B_1440_1440.png'))
    else:
        Show('CBCW')
    time.sleep(2) # s
    Cap_F.capture_item('ContrastW')
    
            
    check_device_bool = Check_device(40) 
    if not check_device_bool:
        print('device_loss')
        sys.exit()

    # Cross
    #os.system('{} s black_point_image_4px_30gap_96value_1080_1440.bmp IQT'.format(Show_batch_file) )#A93 Black 30 dummy
    if PROJECT_NAME == 'A9Q':
        Show_and_LED('Cross_1080_1440', L_RGB, R_RGB)   #A9Q Black 30 dummy
    elif PROJECT_NAME == 'QAR7GC':
        showim('2nd',cv2.imread('Resoure_data/GA2MQ/Cross_1440_1440.png'))
    else:
        Show('Cross')
    time.sleep(2) # s
    Cap_F.capture_item('Cross')        


    check_device_bool = Check_device(40) 
    if not check_device_bool:
        print('device_loss')
        sys.exit()
    # ContrastB
    #os.system('{} s CB3X3A_1080_1440.bmp'.format(Show_batch_file))#A93 CBCB
    if PROJECT_NAME == 'A9Q':
        Show_and_LED('CB3X3A_1080_1440', L_RGB, R_RGB) #A9Q CBCB
    elif PROJECT_NAME == 'QAR7GC':
        showim('2nd',cv2.imread('Resoure_data/GA2MQ/CB3X3A_1440_1440.png'))
    else:
        Show('CBCB')

    time.sleep(2) # s
    Cap_F.capture_item('ContrastB')      

    check_device_bool = Check_device(40) 
    if not check_device_bool:
        print('device_loss')
        sys.exit()
    
    # capture first image - White 
    #os.system('{} s White_1080_1440.bmp'.format(Show_batch_file) ) #A93 white
    if PROJECT_NAME == 'A9Q':
        Show_and_LED('White_1080_1440', L_RGB, R_RGB)#A9Q white
    elif PROJECT_NAME == 'QAR7GC':
        showim('2nd',cv2.imread('Resoure_data/GA2MQ/White_1440_1440.png'))
    else:
        Show('White')
    time.sleep(2) # s
    white_capture_time = time.time()
    print('white_capture_time(s):' + str(white_capture_time - start))
    Cap_F.capture_item('White_blemish')
    

    check_device_bool = Check_device(40) 
    if not check_device_bool:
        print('device_loss')
        sys.exit()

    # Black image
    #os.system('{} s Black_1080_1440.bmp'.format(Show_batch_file)) #A93 black
    if PROJECT_NAME == 'A9Q':
        Show_and_LED('Black_1080_1440', L_RGB, R_RGB)#A9Q black
    elif PROJECT_NAME == 'QAR7GC':
        showim('2nd',cv2.imread('Resoure_data/GA2MQ/Black_1440_1440.png'))
    else:
        Show('Black')
    time.sleep(2) # s
    black_capture_time = time.time()
    print('black_capture_time(s):' + str(black_capture_time - start))
    Cap_F.capture_item('Black_blemish')  

    
       
    
    check_device_bool = Check_device(40) 
    if not check_device_bool:
        print('device_loss')
        sys.exit()

    # White 50 dummy
    # os.system('{} s white_point_image_4px_50gap_0value_1080_1440.bmp'.format(Show_batch_file)) #A93 White 50 dummy
    if PROJECT_NAME == 'A9Q':
        Show_and_LED('white_point_image_4px_50gap_0value_1080_1440', L_RGB, R_RGB)#A9Q White 50 dummy
    elif PROJECT_NAME == 'QAR7GC':
        showim('2nd',cv2.imread('Resoure_data/GA2MQ/white_point_image_4px_50gap_0value_1440_1440.png'))
    else:
        Show('W50D')
    time.sleep(2) # s
    Cap_F.capture_item('White_4px_50gap')
    

    check_device_bool = Check_device(40) 
    if not check_device_bool:
        print('device_loss')
        sys.exit()

    # Black 50 dummy
    # os.system('{} s black_point_image_4px_50gap_96value_1080_1440.bmp'.format(Show_batch_file)) #A93 Black 50 dummy
    if PROJECT_NAME == 'A9Q':
        Show_and_LED('black_point_image_4px_50gap_96value_1080_1440', L_RGB, R_RGB) #A9Q Black 50 dummy
    elif PROJECT_NAME == 'QAR7GC':
        showim('2nd',cv2.imread('Resoure_data/GA2MQ/black_point_image_4px_50gap_96value_1440_1440.png'))
    else:
        Show('B50D')
    time.sleep(2) # s
    Cap_F.capture_item('Black_4px_50gap')
    

    check_device_bool = Check_device(40) 
    if not check_device_bool:
        print('device_loss')
        sys.exit()

    # White 30 dummy
    #os.system('{} s white_point_image_4px_30gap_0value_1080_1440.bmp IQT'.format(Show_batch_file))#A93 White 30 dummy
    if PROJECT_NAME == 'A9Q':
        Show_and_LED('white_point_image_4px_30gap_0value_1080_1440', L_RGB, R_RGB)   #A9Q White 30 dummy
    elif PROJECT_NAME == 'QAR7GC':
        showim('2nd',cv2.imread('Resoure_data/GA2MQ/white_point_image_4px_30gap_0value_1440_1440.png'))
    else:
        Show('W30D')
    time.sleep(2) # s
    Cap_F.capture_item('White_4px_30gap')
    
    check_device_bool = Check_device(40) 
    if not check_device_bool:
        print('device_loss')
        sys.exit()

    # Black 30 dummy
    #os.system('{} s black_point_image_4px_30gap_96value_1080_1440.bmp IQT'.format(Show_batch_file) )#A93 Black 30 dummy
    if PROJECT_NAME == 'A9Q':
        Show_and_LED('black_point_image_4px_30gap_96value_1080_1440', L_RGB, R_RGB)   #A9Q Black 30 dummy
    elif PROJECT_NAME == 'QAR7GC':
        showim('2nd',cv2.imread('Resoure_data/GA2MQ/black_point_image_4px_30gap_96value_1440_1440.png'))
    else:
        Show('B30D')    
    time.sleep(3) # s
    Cap_F.capture_item('Black_4px_30gap')


    
    '''
    # A93 IQC part , A9Q closed
    if ghost_switch:
        # Ghost
        for chart in ghost_charts:
            deco = chart.split('_')[-1].split('.')[0]
            os.system('{} s {}'.format(Show_batch_file,chart))
            os.system('set-current_default.bat')
            time.sleep(t) # s
            Cap_F.capture_item('Ghost', deco_term = deco)
            
            if not Check_device_open( Check_device_open_shitch):
                print('device disconnect')
                Cap_F.close_cameras()

    ##'dist_white_prewarp_grid.png'

    # A93 IQC part , A9Q closed
    if Distortion_switch:
        # Distortion
        for chart in distortion_charts:
            deco = chart.split('dist_')[1].split('.')[0]
            os.system('{} s {}'.format(Show_batch_file,chart))
            os.system('set-current_default.bat')
            time.sleep(t) # s
            Cap_F.capture_item('Distortion', deco_term = deco)
            
            if not Check_device_open( Check_device_open_shitch):
                print('device disconnect')
                Cap_F.close_cameras()                               
    '''
    
    '''    
    # straylight
    os.system('{} t 4K_A93_straylight2_10.png IQT'.format(Show_batch_file))
    time.sleep(t) # s
    Cap_F.capture_item('straylight_test_10') 
       
   # straylight
    os.system('{} t 4K_A93_straylight2_30.png IQT'.format(Show_batch_file))
    time.sleep(t) # s
    Cap_F.capture_item('straylight_test_30') 

    # straylight
    os.system('{} t 4K_A93_straylight2_50.png IQT'.format(Show_batch_file))
    time.sleep(t) # s
    Cap_F.capture_item('straylight_test_50') 

    # straylight
    os.system('{} t 4K_A93_straylight2_255.png IQT'.format(Show_batch_file))
    time.sleep(t) # s
    Cap_F.capture_item('straylight_test_255') 

    os.system('{} s Black_2048_2048.bmp IQT'.format(Show_batch_file))
    time.sleep(t) # s
    Cap_F.capture_item('Black_blemish')    
        

    sys.exit(0)
    '''


    # Turn off camera
    if not TEST_MODE:
        Cap_F.close_cameras()
    
    # Check capture image
    if PROJECT_NAME != 'QAR7GC':
        os.system('{} E'.format(Show_batch_file) )

    
    end = time.time()    
    print('capture time(s):' + str(end - start))
    
    txt = 'capture_finish_info.txt'
    f = open(txt, 'w')
    f.write('finish')
    f.close()

    
