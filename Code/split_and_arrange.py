import csv 
import cv2

# csv file name
# files=[13,14,21,22,23,24,31,32,33,34]
# files=["a. Training Set","b. Testing Set"]
# for i in files:
filename = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\2. Groundtruths\\train.csv"
input_folder = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\1. Original Images\\a. Training Set"
output_folder = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\train"

rows = [] 
with open(filename, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    for row in csvreader: 
        rows.append(row) 

for row in rows[1:]:
    image_last = row[0]
    label = row[1]
    input_path = input_folder + "\\" + image_last + ".jpg"
    print(input_path)
    img = cv2.imread(input_path)
    output_path = output_folder + "\\" + label + "\\" + image_last + ".jpg"
    print(output_path)
    status=cv2.imwrite(output_path,img)
    print(status)


filename = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\2. Groundtruths\\test.csv"
input_folder = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\1. Original Images\\b. Testing Set"
output_folder = "F:\\IISc Stuff\\Idrid\\B. Disease Grading\\test"

rows = [] 
with open(filename, 'r') as csvfile: 
    csvreader = csv.reader(csvfile) 
    for row in csvreader: 
        rows.append(row) 

for row in rows[1:]:
    image_last = row[0]
    label = row[1]
    input_path = input_folder + "\\" + image_last + ".jpg"
    print(input_path)
    img = cv2.imread(input_path)
    output_path = output_folder + "\\" + label + "\\" + image_last + ".jpg"
    print(output_path)
    status=cv2.imwrite(output_path,img)
    print(status)