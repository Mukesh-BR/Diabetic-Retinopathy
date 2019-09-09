import cv2
import os
import glob 

input_path="/home/mukesh/Documents/Datasets/Train_Kaggle_Split/3"
test_files=glob.glob(os.path.join(input_path, "*.jpeg"))

print(len(test_files))
i=0
for path in test_files:
    org=cv2.imread(path)
    
    hf=org.copy()
    vf=org.copy()
    bf=org.copy()

    hf=cv2.flip(org,0)
    vf=cv2.flip(org,1)
    bf=cv2.flip(org,-1)

    list_path=path.split('/')
    file_name=list_path[-1].split('.')[0]

    output_path_hf=input_path+"/"+file_name +"_hf.jpeg"
    output_path_vf=input_path+"/"+file_name +"_vf.jpeg"
    output_path_bf=input_path+"/"+file_name +"_bf.jpeg"

    status_1=cv2.imwrite(output_path_hf,hf)
    status_2=cv2.imwrite(output_path_bf,bf)
    status_3=cv2.imwrite(output_path_vf,vf)
    print(i,status_1,status_2,status_3)
    i+=1