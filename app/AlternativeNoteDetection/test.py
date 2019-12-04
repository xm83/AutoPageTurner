import cv2

staff_file = "resources/template/staff2.png"
im = cv2.imread(staff_file, 0) 
print(im)