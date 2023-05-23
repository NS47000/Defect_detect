import cv2
import numpy as np
import os,subprocess,sys,glob
from GlobalVar import *
import Flir_camera
import joblib
from sklearn import svm
import traceback
from sklearn.model_selection import train_test_split
import basler

threshold=30
blur_size=40#綠波閥值
need_capture=False
filter_thr=190
filter_thr_white=45
script_dir =  os.path.abspath(os.path.dirname(__file__))
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )
def sortroi(element):
    return element[0]+element[2]+(element[1]+element[3])*2
    
def showimage(img):
    cv2.namedWindow('Show Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Show Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_roi(squares,ratio):
    
    for square in squares:
        max_x=0
        max_y=0
        min_x=99999
        min_y=99999
        for point in square:
            if point[0]>max_x:
                max_x=point[0]
            if point[0]<min_x:
                min_x=point[0]
            if point[1]>max_y:
                max_y=point[1]
            if point[1]<min_y:
                min_y=point[1]
        mid_x=(min_x+max_x)/2
        mid_y=(min_y+max_y)/2
        length_x=max_x-min_x
        length_y=max_y-min_y
        shift_x=length_x*(1-ratio)/2
        shift_y=length_y*(1-ratio)/2
        x_min,y_min,x_max,y_max=int(min_x+shift_x),int(min_y+shift_y),int(max_x-shift_x),int(max_y-shift_y)
    
    #print("x_min,y_min,x_max,y_max:{},{},{},{}".format(x_min,y_min,x_max,y_max))
    return x_min,y_min,x_max,y_max

def find_squares(img):
    squares = []
    img = cv2.GaussianBlur(img, (3, 3), 0)
    try:   
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        gray=img
    ret, bin = cv2.threshold(gray,threshold, 255, cv2.THRESH_BINARY)     
    #cv2.namedWindow('My Image2', cv2.WINDOW_NORMAL)
    #cv2.imshow('My Image2',bin)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    contours, _hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("轮廓数量：%d" % len(contours))
    index = 0
    # 轮廓遍历
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True) #计算轮廓周长
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True) #多边形逼近
        # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
        
        if  cv2.contourArea(cnt)>img.shape[0]*img.shape[1]//30:#and cv2.isContourConvex(cnt) and len(cnt) == 4 
            #print("Area:"+str(cv2.contourArea(cnt))+"cnt_len:"+str(len(cnt)))
            M = cv2.moments(cnt) #计算轮廓的矩
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])#轮廓重心
            
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            # 只检测矩形（cos90° = 0）
            #if max_cos < 0.3:
            # 检测四边形（不限定角度范围）
            if True:
                index = index + 1
                #cv2.putText(img,("#%d"%index),(cx,cy),font,0.7,(255,0,255),2)
                squares.append(cnt)
    return squares, img
def capture(pattern_name,save_img):
    dot_img=cv2.imread(pattern_name,1)
    A9Q_showpattern(dot_img)
    #Flir_camera.getpicture(save_img,66666,0)
    
    basler.capture_image(save_img,8346*4,0)
   
    
def capture_white(pattern_name,save_img):
    dot_img=cv2.imread(pattern_name,1)
    A9Q_showpattern(dot_img)
    #Flir_camera.getpicture(save_img,66666*4,0)
    
    basler.capture_image(save_img,8346*300,0)
    
   
    
def filter(base_folder,img_in):
    detect_img=cv2.imread(img_in,0)
    #white_img=cv2.imread(img_white,0)
    draw_img=detect_img.copy()
    squares, img = find_squares(draw_img)
    cv2.drawContours( draw_img, squares, -1, (0, 0, 255), 2 )
    cv2.imwrite(os.path.join(script_dir,"ROI_detect.png"),draw_img)
    x_min,y_min,x_max,y_max=detect_roi(squares,0.9)
    cv2.rectangle(draw_img, (x_min,y_min), (x_max,y_max), (255, 0, 255), 2) 
    cv2.imwrite(os.path.join(script_dir,"ROI_detect_2.png"),draw_img)
    roi_img=detect_img[y_min:y_max,x_min:x_max]

    max_gray=np.max(np.max(roi_img))
    
    roi_img=roi_img.astype(np.float32)
    detect_blur=cv2.blur(detect_img,(blur_size,blur_size))
    detect_blur=detect_blur[y_min:y_max,x_min:x_max]
    detect_blur=detect_blur.astype(np.float32)
    #img_max=np.max(np.max(roi_img-detect_blur))
    #img_min=np.min(np.min(roi_img-detect_blur))
    #print("max:{} min:{}".format(img_max,img_min))

    after_img=(roi_img-detect_blur)+128
    #after_img=((roi_img-detect_blur)-img_min)*255/(img_max-img_min)
    #after_img=after_img.astype(np.uint8)
    avg_after_img=np.mean(np.mean(after_img))
    after_img=after_img*200/avg_after_img
    cv2.imwrite(os.path.join(base_folder,"after_blur_{}.png".format(blur_size)),after_img)
    return "after_blur_{}.png".format(blur_size),roi_img,draw_img,x_min,y_min,x_max,y_max
def filter_white(base_folder,img_in,x_min,y_min,x_max,y_max):
    detect_img=cv2.imread(img_in,0)
    draw_img=detect_img.copy()
    
    #squares, img = find_squares(draw_img)
    #cv2.drawContours( draw_img, squares, -1, (0, 0, 255), 2 )
    #cv2.imwrite(os.path.join(script_dir,"ROI_detect.png"),draw_img)
    #x_min,y_min,x_max,y_max=detect_roi(squares,0.9)
    cv2.rectangle(draw_img, (x_min,y_min), (x_max,y_max), (255, 0, 255), 2) 
    cv2.imwrite(os.path.join(script_dir,"ROI_detect_2.png"),draw_img)
    roi_img=detect_img[y_min:y_max,x_min:x_max]

    max_gray=np.max(np.max(roi_img))
    
    roi_img=roi_img.astype(np.float32)
    detect_blur=cv2.blur(detect_img,(blur_size,blur_size))
    detect_blur=detect_blur[y_min:y_max,x_min:x_max]
    detect_blur=detect_blur.astype(np.float32)
    img_max=np.max(np.max(roi_img-detect_blur))
    img_min=np.min(np.min(roi_img-detect_blur))
    #print("max:{} min:{}".format(img_max,img_min))
    after_img=(roi_img-detect_blur)+128
    #after_img=((roi_img-detect_blur)-img_min)*255/(img_max-img_min)
    #after_img=after_img.astype(np.uint8)
    avg_after_img=np.mean(np.mean(after_img))
    after_img=after_img*40/avg_after_img
    cv2.imwrite(os.path.join(base_folder,"after_blur_{}_w.png".format(blur_size)),after_img)
    return "after_blur_{}_w.png".format(blur_size),roi_img,draw_img
def train_model(base_folder):
    contour_area=[]
    contour_graylevel=[]
    contour_var=[]
    result=[]
    for pixel_size in range(1,5):
        pattern_name="defect_pattern_{}.png".format(pixel_size)
        if need_capture!=False:
            capture(pattern_name,"img_ori_{}p.png".format(pixel_size))
            A9Q_close()
        after_filter,roi_img,_,_,_,_,_=filter(base_folder,os.path.join(base_folder,"img_ori_{}p.png".format(pixel_size)))
        after_img=cv2.imread(os.path.join(base_folder,after_filter),0)
        after_img=cv2.medianBlur(after_img,5)
        thr=filter_thr
        _,thresh = cv2.threshold(after_img,thr,255,cv2.THRESH_BINARY)  
        kernel=(5,5)
        thresh = cv2.erode(thresh, kernel)   # 侵蝕
        thresh= cv2.dilate(thresh, kernel)  # 擴張
        
        

        contours, _hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        image_rgb=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
        
        roi_size=10
        num=0
        for contour in contours:
            if cv2.contourArea(contour)>15 and cv2.contourArea(contour)<5000 :
                #print(cv2.contourArea(contour))
                cv2.drawContours(image_rgb,contour,-1,(0,255,0),1)
                M=cv2.moments(contour)
                cx=int(M['m10']/M['m00'])
                cy=int(M['m01']/M['m00'])
                contour_cut=roi_img[cy-roi_size:cy+roi_size,cx-roi_size:cx+roi_size]
                if np.mean(contour_cut.flatten(),dtype=float)>0 and np.var(contour_cut.flatten(),dtype=float)>0:
                    contour_area.append(cv2.contourArea(contour))
                    contour_graylevel.append(np.mean(contour_cut.flatten(),dtype=float))
                    contour_var.append(np.var(contour_cut.flatten(),dtype=float))
                    result.append(pixel_size)
                    num+=1
                    
        cv2.imwrite(os.path.join(base_folder,"contour_{}.png".format(pixel_size)),image_rgb)
        #print("pixel_{} detect {} point".format(pixel_size,num))
    data_x=np.zeros((len(result),3))
    data_y=np.zeros((len(result),1))
    contour_area_mean=np.nanmean(contour_area)
    contour_graylevel_mean=np.nanmean(contour_graylevel)
    contour_var_mean=np.nanmean(contour_var)
    for i in range(0,len(result)):
        data_x[i,0]=contour_area[i]
        data_x[i,1]=contour_graylevel[i]
        data_x[i,2]=contour_var[i]
        data_y[i]=int(result[i])
    data_x_mean=[contour_area_mean,contour_graylevel_mean,contour_var_mean]
    for i in range(0,len(result)):
        data_x[i,0]/=contour_area_mean
        data_x[i,1]/=contour_graylevel_mean
        data_x[i,2]/=contour_var_mean
    #print("data_x_mean:")
    #print(data_x_mean)
    np.save(os.path.join(base_folder,'normalize_parameter.npy'),data_x_mean)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=0, train_size=0.7)#train_size can decide training rate
    clf = svm.SVC(C=1, kernel='rbf', decision_function_shape='ovr',cache_size=1000)#建立svm模型
    clf.fit(x_train, y_train.ravel())#training
    print("=====black model:========")
    print("Traindata Accuracy:"+str(clf.score(x_train, y_train)))  # show Accuracy
    y_hat = clf.predict(x_train)# try classify

    print("Testdata Accuracy:"+str(clf.score(x_test, y_test)))
    y_hat2 = clf.predict(x_test)# try classify
    #print(y_hat2)#show output
    joblib.dump(clf,os.path.join(base_folder,"svm_model.m"))#save model
    return clf.score(x_test, y_test) 
    
def train_white_model(base_folder,x_min,y_min,x_max,y_max):
    contour_area=[]
    contour_graylevel=[]
    contour_var=[]
    result=[]
    for pixel_size in range(1,5):
        pattern_name="defect_pattern_{}_white.png".format(pixel_size)
        if need_capture!=False:
            capture_white(pattern_name,"img_ori_{}_white.png".format(pixel_size))
            A9Q_close()
        after_filter,roi_img,_=filter_white(base_folder,os.path.join(base_folder,"img_ori_{}_white.png".format(pixel_size)),x_min,y_min,x_max,y_max)

        after_img=cv2.imread(os.path.join(base_folder,after_filter),0)
        after_img=cv2.medianBlur(after_img,5)
        thr=filter_thr_white
        _,thresh = cv2.threshold(after_img,thr,255,cv2.THRESH_BINARY)  
        kernel=(5,5)
        thresh= cv2.dilate(thresh, kernel)  # 擴張
        thresh = cv2.erode(thresh, kernel)   # 侵蝕
        

        contours, _hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        image_rgb=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
        
        roi_size=10
        num=0
        for contour in contours:
            if cv2.contourArea(contour)>5 and cv2.contourArea(contour)<5000 :
                #print(cv2.contourArea(contour))
                cv2.drawContours(image_rgb,contour,-1,(0,255,0),1)
                M=cv2.moments(contour)
                cx=int(M['m10']/M['m00'])
                cy=int(M['m01']/M['m00'])
                contour_cut=roi_img[cy-roi_size:cy+roi_size,cx-roi_size:cx+roi_size]
                if np.mean(contour_cut.flatten(),dtype=float)>0 and np.var(contour_cut.flatten(),dtype=float)>0:
                    contour_area.append(cv2.contourArea(contour))
                    contour_graylevel.append(np.mean(contour_cut.flatten(),dtype=float))
                    contour_var.append(np.var(contour_cut.flatten(),dtype=float))
                    result.append(pixel_size)
                    num+=1
                    
        cv2.imwrite("contour_{}.png".format(pixel_size),image_rgb)
        #print("white_pixel_{} detect {} point".format(pixel_size,num))
    data_x=np.zeros((len(result),3))
    data_y=np.zeros((len(result),1))
    contour_area_mean=np.nanmean(contour_area)
    contour_graylevel_mean=np.nanmean(contour_graylevel)
    contour_var_mean=np.nanmean(contour_var)
    for i in range(0,len(result)):
        data_x[i,0]=contour_area[i]
        data_x[i,1]=contour_graylevel[i]
        data_x[i,2]=contour_var[i]
        data_y[i]=int(result[i])
    data_x_mean=[contour_area_mean,contour_graylevel_mean,contour_var_mean]
    for i in range(0,len(result)):
        data_x[i,0]/=contour_area_mean
        data_x[i,1]/=contour_graylevel_mean
        data_x[i,2]/=contour_var_mean
    #print("data_x_mean:")
    #print(data_x_mean)
    np.save(os.path.join(base_folder,'normalize_parameter_white.npy'),data_x_mean)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=0, train_size=0.7)#train_size can decide training rate
    clf = svm.SVC(C=1, kernel='rbf', decision_function_shape='ovr',cache_size=1000)#建立svm模型
    clf.fit(x_train, y_train.ravel())#training
    print("=====white model:========")
    print("Traindata Accuracy_white:"+str(clf.score(x_train, y_train)))  # show Accuracy
    y_hat = clf.predict(x_train)# try classify

    print("Testdata Accuracy_white:"+str(clf.score(x_test, y_test)))
    y_hat2 = clf.predict(x_test)# try classify
    #print(y_hat2)#show output
    joblib.dump(clf,os.path.join(base_folder,"svm_model_white.m"))#save model
    return clf.score(x_test, y_test)
    


if __name__ == '__main__':
    folder_list=glob.glob(os.path.join('E:\IQT\defect_data','QA9Q230520064','L','white.png'))
    for folder_path in folder_list:
        try:
            main_folder=os.path.dirname(folder_path)
            Accurate=train_model(main_folder)
            pattern_name="test.png"
            if need_capture!=False:
                capture(pattern_name,"detect_1and2pixel.png")
                A9Q_close()
            after_filter,roi_img,draw_img,x_min,y_min,x_max,y_max=filter(main_folder,os.path.join(main_folder,"white.png"))
            draw_img_rgb=cv2.cvtColor(draw_img,cv2.COLOR_GRAY2BGR)
            after_img=cv2.imread(os.path.join(main_folder,after_filter),0)
            after_img=cv2.medianBlur(after_img,5)
            thr=filter_thr
            _,thresh = cv2.threshold(after_img,thr,255,cv2.THRESH_BINARY)  
            kernel=(5,5)
            thresh = cv2.erode(thresh, kernel,iterations=1)   # 侵蝕
            thresh= cv2.dilate(thresh, kernel,iterations=1)  # 擴張
            #showimage(thresh)
            contour_area=[]
            contour_graylevel=[]
            contour_var=[]
            center_pos=[]
            
            contours, _hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            image_rgb=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
            roi_size=10
            num=0
            for contour in contours:
                if cv2.contourArea(contour)>15 and cv2.contourArea(contour)<5000 :
                    #print(cv2.contourArea(contour))
                    
                    
                    M=cv2.moments(contour)
                    cx=int(M['m10']/M['m00'])
                    cy=int(M['m01']/M['m00'])
                    contour_cut=roi_img[cy-roi_size:cy+roi_size,cx-roi_size:cx+roi_size]
                    if np.mean(contour_cut.flatten(),dtype=float)>0 and np.var(contour_cut.flatten(),dtype=float)>0:
                        contour_area.append(cv2.contourArea(contour))
                        contour_graylevel.append(np.mean(contour_cut.flatten(),dtype=float))
                        contour_var.append(np.var(contour_cut.flatten(),dtype=float))
                        center_pos.append([cy,cx])
                        num+=1
                        cv2.drawContours(image_rgb,contour,-1,(0,255,0),2)
            cv2.imwrite(os.path.join(main_folder,"check_contour.png"),image_rgb)
            #print("Detect {} points!!".format(num))
            data_x=np.zeros((len(contour_area),3))
            #data_y=np.zeros((len(contour_area),1))
            for i in range(0,len(contour_area)):
                data_x[i,0]=contour_area[i]
                data_x[i,1]=contour_graylevel[i]
                data_x[i,2]=contour_var[i]
                
            data_x_mean=np.load('normalize_parameter.npy')
            for i in range(0,len(contour_area)):
                data_x[i,0]/=data_x_mean[0]
                data_x[i,1]/=data_x_mean[1]
                data_x[i,2]/=data_x_mean[2]
            #print("data_x_mean:")
            #print(data_x_mean)
            clf=joblib.load(os.path.join(main_folder,"svm_model.m"))
            cat_number = 0
            one_pixel_number=0
            two_pixel_number=0
            three_pixel_number=0
            four_pixel_number=0
            if len(data_x)!=0:
                index_svm=clf.predict(data_x)
                
                for defect in range(0,len(index_svm)):
                    if int(index_svm[defect])==1:#將分類結果畫圖秀出來
                        #if len(c) < 15: # noise
                        one_pixel_number += 1
                        cv2.rectangle(draw_img_rgb, (center_pos[defect][1]-roi_size+x_min,center_pos[defect][0]-roi_size+y_min),(center_pos[defect][1]+roi_size+x_min,center_pos[defect][0]+roi_size+y_min), (255, 0, 0), 2) 
                    elif int(index_svm[defect])==2: 
                        two_pixel_number += 1
                        cv2.rectangle(draw_img_rgb, (center_pos[defect][1]-roi_size+x_min,center_pos[defect][0]-roi_size+y_min),(center_pos[defect][1]+roi_size+x_min,center_pos[defect][0]+roi_size+y_min), (0, 255, 0), 2) 
                    elif int(index_svm[defect])==3:
                        three_pixel_number += 1
                        cv2.rectangle(draw_img_rgb, (center_pos[defect][1]-roi_size+x_min,center_pos[defect][0]-roi_size+y_min),(center_pos[defect][1]+roi_size+x_min,center_pos[defect][0]+roi_size+y_min), (0, 0, 255), 2) 
                    else:
                        four_pixel_number += 1
                        cv2.rectangle(draw_img_rgb, (center_pos[defect][1]-roi_size+x_min,center_pos[defect][0]-roi_size+y_min),(center_pos[defect][1]+roi_size+x_min,center_pos[defect][0]+roi_size+y_min), (255, 0, 255), 2) 
                print("one_pixel detect: {} points".format(one_pixel_number))
                print("two_pixel detect: {} points".format(two_pixel_number))
                print("three_pixel detect: {} points".format(three_pixel_number))
                print("four_pixel detect: {} points".format(four_pixel_number))
            else:
                print("No any black defect")
            cv2.imwrite(os.path.join(main_folder,"result.png"),draw_img_rgb)
            
            #sys.exit(1)
            #==============white===========================================
            Accurate_w=train_white_model(main_folder,x_min,y_min,x_max,y_max)
            pattern_name="test_white.png"
            if need_capture!=False:
                capture_white(pattern_name,"detect_1and2pixel_white.png")
                A9Q_close()
            after_filter,roi_img,draw_img=filter_white(main_folder,os.path.join(main_folder,"black.png"),x_min,y_min,x_max,y_max)
            draw_img_rgb=cv2.cvtColor(draw_img,cv2.COLOR_GRAY2BGR)
            after_img=cv2.imread(os.path.join(main_folder,after_filter),0)
            #showimage(after_img)
            after_img=cv2.medianBlur(after_img,5)
            #showimage(after_img)
            thr=filter_thr_white
            _,thresh = cv2.threshold(after_img,thr,255,cv2.THRESH_BINARY)  
            kernel=(5,5)
            #showimage(thresh)
            thresh= cv2.dilate(thresh, kernel)  # 擴張
            thresh = cv2.erode(thresh, kernel)   # 侵蝕
            #showimage(thresh)
            contour_area=[]
            contour_graylevel=[]
            contour_var=[]
            center_pos=[]
            
            contours, _hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            image_rgb=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
            roi_size=10
            num=0
            for contour in contours:
                if cv2.contourArea(contour)>5 and cv2.contourArea(contour)<5000 :
                    #print(cv2.contourArea(contour))
                    
                    
                    M=cv2.moments(contour)
                    cx=int(M['m10']/M['m00'])
                    cy=int(M['m01']/M['m00'])
                    contour_cut=roi_img[cy-roi_size:cy+roi_size,cx-roi_size:cx+roi_size]
                    if np.mean(contour_cut.flatten(),dtype=float)>0 and np.var(contour_cut.flatten(),dtype=float)>0:
                        contour_area.append(cv2.contourArea(contour))
                        contour_graylevel.append(np.mean(contour_cut.flatten(),dtype=float))
                        contour_var.append(np.var(contour_cut.flatten(),dtype=float))
                        center_pos.append([cy,cx])
                        num+=1
                        cv2.drawContours(image_rgb,contour,-1,(0,255,0),2)
            cv2.imwrite(os.path.join(main_folder,"check_contour_white.png"),image_rgb)
            print("Detect {} White points!!".format(num))
            data_x=np.zeros((len(contour_area),3))
            #data_y=np.zeros((len(contour_area),1))
            for i in range(0,len(contour_area)):
                data_x[i,0]=contour_area[i]
                data_x[i,1]=contour_graylevel[i]
                data_x[i,2]=contour_var[i]
                
            data_x_mean=np.load(os.path.join(main_folder,'normalize_parameter_white.npy'))
            for i in range(0,len(contour_area)):
                data_x[i,0]/=data_x_mean[0]
                data_x[i,1]/=data_x_mean[1]
                data_x[i,2]/=data_x_mean[2]
            #print("data_x_mean(area,gray,var):")
            #print(data_x_mean)
            clf=joblib.load(os.path.join(main_folder,"svm_model_white.m"))
            cat_number = 0
            one_pixel_number_w=0
            two_pixel_number_w=0
            three_pixel_number_w=0
            four_pixel_number_w=0
            if len(data_x)!=0:
                index_svm=clf.predict(data_x)
                
                
                for defect in range(0,len(index_svm)):
                    if int(index_svm[defect])==1:#將分類結果畫圖秀出來
                        #if len(c) < 15: # noise
                        one_pixel_number_w += 1
                        cv2.rectangle(draw_img_rgb, (center_pos[defect][1]-roi_size+x_min,center_pos[defect][0]-roi_size+y_min),(center_pos[defect][1]+roi_size+x_min,center_pos[defect][0]+roi_size+y_min), (255, 0, 0), 2) 
                    elif int(index_svm[defect])==2: 
                        two_pixel_number_w += 1
                        cv2.rectangle(draw_img_rgb, (center_pos[defect][1]-roi_size+x_min,center_pos[defect][0]-roi_size+y_min),(center_pos[defect][1]+roi_size+x_min,center_pos[defect][0]+roi_size+y_min), (0, 255, 0), 2) 
                    elif int(index_svm[defect])==3:
                        three_pixel_number_w += 1
                        cv2.rectangle(draw_img_rgb, (center_pos[defect][1]-roi_size+x_min,center_pos[defect][0]-roi_size+y_min),(center_pos[defect][1]+roi_size+x_min,center_pos[defect][0]+roi_size+y_min), (0, 0, 255), 2) 
                    else:
                        four_pixel_number_w += 1
                        cv2.rectangle(draw_img_rgb, (center_pos[defect][1]-roi_size+x_min,center_pos[defect][0]-roi_size+y_min),(center_pos[defect][1]+roi_size+x_min,center_pos[defect][0]+roi_size+y_min), (255, 0, 255), 2) 
                #print("one_Whitepixel detect: {} points".format(one_pixel_number))
                #print("two_Whitepixel detect: {} points".format(two_pixel_number))
                #print("three_Whitepixel detect: {} points".format(three_pixel_number))
                #print("four_Whitepixel detect: {} points".format(four_pixel_number))
                #print("Accuracy:{}".format(max([one_pixel_number,two_pixel_number,three_pixel_number,four_pixel_number])/(one_pixel_number+two_pixel_number+three_pixel_number+four_pixel_number)))
                print("one_Whitepixel_w detect: {} points".format(one_pixel_number_w))
                print("two_Whitepixel_w detect: {} points".format(two_pixel_number_w))
                print("three_Whitepixel_w detect: {} points".format(three_pixel_number_w))
                print("four_Whitepixel_w detect: {} points".format(four_pixel_number_w))
                print("Accuracy_w:{}".format(max([one_pixel_number_w,two_pixel_number_w,three_pixel_number_w,four_pixel_number_w])/(one_pixel_number_w+two_pixel_number_w+three_pixel_number_w+four_pixel_number_w)))
            else:
                print("No any white defect")
            cv2.imwrite(os.path.join(main_folder,"result_white.png"),draw_img_rgb)
            if os.path.exists("result.csv")==0:
                with open("result.csv",'w') as f:
                    f.write("SN,Side,one_pixel_number,two_pixel_number,three_pixel_number,four_pixel_number,one_pixel_number_w,two_pixel_number_w,three_pixel_number_w,four_pixel_number_w,Accurate,Accurate_w \n")
                
            with open("result.csv",'a') as f:
                sn=os.path.dirname(main_folder).split('\\')[-1] 
                side=main_folder.split('\\')[-1] 
                print("SN:{} Side:{} done!!!".format(sn,side))
                f.write("{},{},{},{},{},{},{},{},{},{},{},{} \n".format(sn,side,one_pixel_number,two_pixel_number,three_pixel_number,four_pixel_number,one_pixel_number_w,two_pixel_number_w,three_pixel_number_w,four_pixel_number_w,Accurate,Accurate_w))
        except Exception as e:
            
            print(e)
            error_class = e.__class__.__name__ #取得錯誤類型
            detail = e.args[0] #取得詳細內容
            cl, exc, tb = sys.exc_info() #取得Call Stack
            lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
            fileName = lastCallStack[0] #取得發生的檔案名稱
            lineNum = lastCallStack[1] #取得發生的行號
            funcName = lastCallStack[2] #取得發生的函數名稱
            errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
            print(errMsg)
            with open("error.csv",'a') as f:
                sn=os.path.dirname(main_folder).split('\\')[-1] 
                side=main_folder.split('\\')[-1] 
                print("SN:{} Side:{} error!!!".format(sn,side))
                f.write("{},{},{} \n".format(sn,side,errMsg))
            
    
                
                

    
        
    



