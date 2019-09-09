import cv2, glob, os
import numpy as np
import pandas as pd

def estimate_radius(img):
    mx = img[img.shape[0] // 2,:,:].sum(1)
    rx = (mx > mx.mean() / 10).sum() / 2

    my = img[:,img.shape[1] // 2,:].sum(1)
    ry = (my > my.mean() / 10).sum() / 2

    return (ry, rx)

def subtract_gaussian_blur(img):
    # http://docs.opencv.org/trunk/d0/d86/tutorial_py_image_arithmetics.html
    # http://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
    gb_img = cv2.GaussianBlur(img, (0, 0), 5)

    return cv2.addWeighted(img, 4, gb_img, -4, 128)

def remove_outer_circle(a, p, r):
    b = np.zeros(a.shape, dtype=np.uint8)
    cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(r * p), (1, 1, 1), -1, 8, 0)

    return a * b + 128 * (1 - b)


def crop_img(img, h, w):
        h_margin = (img.shape[0] - h) // 2 if img.shape[0] > h else 0
        w_margin = (img.shape[1] - w) // 2 if img.shape[1] > w else 0

        crop_img = img[h_margin:h + h_margin,w_margin:w + w_margin,:]

        return crop_img

def place_in_square(img, r, h, w):
    new_img = np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)
    new_img[r - h // 2:r - h // 2 + img.shape[0], r - w // 2:r - w // 2 + img.shape[1]] = img

    return new_img

def preprocess1(file):
    image=cv2.imread(file)
    b,g,r=cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(g)
    return cl1

def preprocess(f, r, debug_plot=False):
    try:
        img = cv2.imread(f)
        ry, rx = estimate_radius(img)

        b,g,red=cv2.split(img)
        img2=preprocess1(f)

        image=cv2.imread(f)
        imagelab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        #image2=preprocess2(f)
        l,a,b=cv2.split(imagelab)
        img=cv2.merge((img2,g,img2))
        # if debug_plot:
        #     plt.figure()
        #     plt.imshow(img)

        resize_scale = r / max(rx, ry)
        #print(r,rx,ry)
        w = min(int(rx * resize_scale * 2), r * 2)
        h = min(int(ry * resize_scale * 2), r * 2)

        #print(rx,ry)
        #print(img.shape,resize_scale,r,rx,ry)
        w1=int(img.shape[1]*resize_scale)
        h1=int(img.shape[0]*resize_scale)
        #print(w1,h1)
        img = cv2.resize(img,(w1,h1))

        img = crop_img(img, h, w)
        #print("crop_img", np.mean(img), np.std(img))
        #print(img.shape)
        # if debug_plot:
        #     plt.figure()
        #     plt.imshow(img)

        #img = subtract_gaussian_blur(img)
        #img = remove_outer_circle(img, 0.9, r)
        img = place_in_square(img, int(r), int(h), int(w))

        # if debug_plot:
        #     plt.figure()
        #     plt.imshow(img)
        #print(img.shape)
        return img

    except Exception as e:
        print("file {} exception {}".format(f, e))
        pass

    return None


input_path_0 = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\test\\0"
input_path_1 = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\test\\1"
input_path_2 = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\test\\2"
input_path_3 = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\test\\3"
input_path_4 = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\test\\4"

test_files_0=glob.glob(os.path.join(input_path_0, "*.jpg"))
print(len(test_files_0))
test_files_1=glob.glob(os.path.join(input_path_1, "*.jpg"))
print(len(test_files_1))
test_files_2=glob.glob(os.path.join(input_path_2, "*.jpg"))
print(len(test_files_2))
test_files_3=glob.glob(os.path.join(input_path_3, "*.jpg"))
print(len(test_files_3))
test_files_4=glob.glob(os.path.join(input_path_4, "*.jpg"))
print(len(test_files_4))

out_directory_0 = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\preprocessed\\test\\0"
out_directory_1 = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\preprocessed\\test\\1"
out_directory_2 = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\preprocessed\\test\\2"
out_directory_3 = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\preprocessed\\test\\3"
out_directory_4 = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\preprocessed\\test\\4"

i_0=0
i_1=0
i_2=0
i_3=0
i_4=0
# for i in test_files_0:
#     img=preprocess(i,448.0,False)
#     list_name=i.split('\\')
#     file_name=out_directory_0+'\\'+list_name[-1]
#     print(file_name)
#     status=cv2.imwrite(file_name,img)
#     print(i_0,status,img.shape)
#     i_0+=1


# for i in test_files_1:
#     img=preprocess(i,448.0,False)
#     list_name=i.split('\\')
#     file_name=out_directory_1+'\\'+list_name[-1]
#     status=cv2.imwrite(file_name,img)
#     print(i_1,status,img.shape)
#     i_1+=1

# for i in test_files_2:
#     img=preprocess(i,448.0,False)
#     list_name=i.split('\\')
#     file_name=out_directory_2+'\\'+list_name[-1]
#     status=cv2.imwrite(file_name,img)
#     print(i_2,status,img.shape)
#     i_2+=1

for i in test_files_3:
    img=preprocess(i,448.0,False)
    list_name=i.split('\\')
    file_name=out_directory_3+'\\'+list_name[-1]
    status=cv2.imwrite(file_name,img)
    print(i_3,status,img.shape)
    i_3+=1


for i in test_files_4:
    img=preprocess(i,448.0,False)
    list_name=i.split('\\')
    file_name=out_directory_4+'\\'+list_name[-1]
    status=cv2.imwrite(file_name,img)
    print(i_4,status,img.shape)
    i_4+=1

 
 