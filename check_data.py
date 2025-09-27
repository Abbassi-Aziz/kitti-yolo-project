import cv2
import matplotlib.pyplot as plt
import random

x = random.randint(0,7481)
file_base = str(x)
while len(file_base) < 6:
    file_base = '0' + file_base
filetxt = 'datasets/data_object_label_2/train/label_2/' + file_base + '.txt'
fileimg = 'datasets/data_object_image_2/train/image_2/' + file_base + '.png'
img = cv2.imread(fileimg)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with open (filetxt, 'r') as f:
    lines = f.readlines()
    for line in lines:
        info = line.split(' ')
        label = info[0]
        left,top,right,bottom = int(float(info[4])), int(float(info[5])),int(float(info[6])), int(float(info[7]))
        cv2.rectangle(img_rgb,(left,top),(right,bottom),(0,0,255),2)

        cv2.putText(
            img_rgb,
            label,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

plt.imshow(img_rgb)
plt.axis('off')
plt.show()