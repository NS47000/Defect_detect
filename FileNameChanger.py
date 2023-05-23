import os
import cv2
import glob,stat
import shutil

mason_folder="defect_data"
if os.path.exists(mason_folder)==0:
    os.mkdir(mason_folder)
    os.chmod(mason_folder,stat.S_IWRITE)


folder_list=glob.glob(os.path.join('*','*','Black_blemish','2503800'))
for folder in folder_list:
    folder=os.path.dirname(os.path.dirname(os.path.dirname(folder)))
    print(folder)
    if os.path.exists(os.path.join(mason_folder,folder))==0:
        os.mkdir(os.path.join(mason_folder,folder))
        os.mkdir(os.path.join(mason_folder,folder,'R'))
        os.mkdir(os.path.join(mason_folder,folder,'L'))
    filepath_L=glob.glob(os.path.join(folder,"*",'Black_blemish','2503800','caml_exp0_n0_hb.png'))
    shutil.copyfile(filepath_L[-1], os.path.join(mason_folder,folder,'L','black.png'))
    filepath_L=glob.glob(os.path.join(folder,"*",'White_blemish','33384','caml_exp1_n0_hb.png'))
    shutil.copyfile(filepath_L[-1], os.path.join(mason_folder,folder,'L','white.png'))
    filepath_R=glob.glob(os.path.join(folder,"*",'Black_blemish','2503800','camr_exp0_n0_hb.png'))
    shutil.copyfile(filepath_R[-1], os.path.join(mason_folder,folder,'R','black.png'))
    filepath_R=glob.glob(os.path.join(folder,"*",'White_blemish','33384','camr_exp1_n0_hb.png'))
    shutil.copyfile(filepath_R[-1], os.path.join(mason_folder,folder,'R','white.png'))
    
    for pixel in range(1,5):
        
        filepath_L=glob.glob(os.path.join(folder,"*",'defect_pattern_{}'.format(pixel),'caml_exp0_hb.png'))
        shutil.copyfile(filepath_L[-1], os.path.join(mason_folder,folder,'L',"img_ori_{}p.png".format(pixel)))
        filepath_L=glob.glob(os.path.join(folder,"*",'defect_pattern_{}_white'.format(pixel),'caml_exp0_hb.png'))
        shutil.copyfile(filepath_L[-1], os.path.join(mason_folder,folder,'L',"img_ori_{}_white.png".format(pixel)))
        filepath_R=glob.glob(os.path.join(folder,"*",'defect_pattern_{}'.format(pixel),'camr_exp0_hb.png'))
        shutil.copyfile(filepath_R[-1], os.path.join(mason_folder,folder,'R',"img_ori_{}p.png".format(pixel)))
        filepath_R=glob.glob(os.path.join(folder,"*",'defect_pattern_{}_white'.format(pixel),'camr_exp0_hb.png'))
        shutil.copyfile(filepath_R[-1], os.path.join(mason_folder,folder,'R',"img_ori_{}_white.png".format(pixel)))
        