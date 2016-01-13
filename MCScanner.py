import cv2
import numpy as np


def cropsheet(image,org_image):
    template = cv2.imread('find.jpg',0)
    w = np.size(template,0)
    h = np.size(template,1)
    height = np.size(img_rgb,0)
    length = np.size(img_rgb,1)

    res = cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.80
    loc = np.where( res >= threshold)

    loc = list(loc)

    loc_maxy = max(loc[0])
    loc_miny = min(loc[0])
    loc_maxx = max(loc[1])
    loc_minx = min(loc[1])

    loc_1x = loc_minx + w/2
    loc_2x = loc_maxx + w/2
    loc_y = (loc_maxy + loc_miny)/2 + h/2

    loc_1 = tuple([loc_1x,loc_y])
    loc_2 = tuple([loc_2x,loc_y])

    org_image = org_image[(height - loc_y):loc_y, loc_1x:loc_2x]
    cropped = image[(height - loc_y):loc_y, loc_1x:loc_2x]

    first_crop_h = np.size(cropped,0)
    first_crop_w = np.size(cropped,1)

    cropped2 = cropped[(first_crop_h*0.22):(first_crop_h*0.975), (first_crop_w*0.385):(first_crop_w*.995)]
    
    return cropped2

img_rgb = cv2.imread('scan.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
cropped_image = cropsheet(img_gray,img_rgb)
cropped_image = cv2.adaptiveThreshold(cropped_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,2)
cropped_image = cv2.bitwise_not(cropped_image)
answer_area_h = np.size(cropped_image,0)
answer_area_w = np.size(cropped_image,1)

column_const = [[0,0.095],[0.19,0.275],[0.36,0.45],[0.545,0.63],[0.73,0.815],[0.915,1]]
row_const = [[0,0.145],[0.17,0.315],[0.34,0.4825],[0.5125,0.6552],[0.685,0.83],[0.855,1]]
block = []

for j in column_const:
    for i in row_const:
        block.append(cropped_image[(i[0])*answer_area_h:(i[1])*answer_area_h,(j[0])*answer_area_w:(j[1])*answer_area_w])

new_height = np.size(cropped_image,0)
new_width = np.size(cropped_image,1)

answers = []
responses = []
options = []
answers.append("")
responses.append("")



for blk in block:

    row = []

    new_height = np.size(blk,0)
    new_width = np.size(blk,1)
    new_area = (new_height/5)*new_width
    row.append("")

    for i in range(1,6):
        row.append(blk[((i-1)*new_height/5):(i*new_height/5),0:new_width])

        options = []
        left_pos_pixels = cv2.countNonZero(row[i][0:new_height/5,0:(new_width/2)])
        right_pos_pixels = cv2.countNonZero(row[i][0:new_height/5,(new_width/2):(new_width)])
        left_density = (left_pos_pixels*1000/new_area)
        right_density = (right_pos_pixels*1000/new_area)

        options.append(left_density)
        options.append(right_density)
        answers.append(options)

for k in range(1,len(answers)):
    if answers[k][0] - answers[k][1] > 30:
        responses.append("True")
    elif answers[k][0] - answers[k][1] < -30:
        responses.append("False")
    else:
        responses.append("Null")

count = 1
for ans in responses[1:]:
    print(str(count) + ", " + ans)
    if count % 5 == 0:
        print(" ")
    count += 1
####cv2.waitKey(0)
####cv2.destroyAllWindows()
