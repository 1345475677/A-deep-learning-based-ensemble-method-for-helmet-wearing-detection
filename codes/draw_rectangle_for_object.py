import os
import xml.dom.minidom
import cv2 as cv
import numpy as np


ImgPath = 'data/train\JPEGImages/'
AnnoPath = 'data/train\Annotations/'
save_path = 'data/train/JPEGImages_object'


def cv_imread(filePath):
    cv_img=cv.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img
def cv_imwrite(save_Path,img):
    cv.imencode('.jpg', img)[1].tofile(save_Path)

def draw_anchor(ImgPath, AnnoPath, save_path):
    imagelist = os.listdir(ImgPath)
    for image in imagelist:
        image_pre, ext = os.path.splitext(image)
        imgfile = ImgPath + image
        xmlfile = AnnoPath + image_pre + '.xml'

        DOMTree = xml.dom.minidom.parse(xmlfile)

        collection = DOMTree.documentElement

        img = cv_imread(imgfile)

        filenamelist = collection.getElementsByTagName("filename")
        filename = filenamelist[0].childNodes[0].data
        print(filename)

        objectlist = collection.getElementsByTagName("object")

        for objects in objectlist:

            namelist = objects.getElementsByTagName('name')

            objectname = namelist[0].childNodes[0].data

            bndbox = objects.getElementsByTagName('bndbox')
            # print(bndbox)
            for box in bndbox:
                x1_list = box.getElementsByTagName('xmin')
                x1 = int(x1_list[0].childNodes[0].data)
                y1_list = box.getElementsByTagName('ymin')
                y1 = int(y1_list[0].childNodes[0].data)
                x2_list = box.getElementsByTagName('xmax')
                x2 = int(x2_list[0].childNodes[0].data)
                y2_list = box.getElementsByTagName('ymax')
                y2 = int(y2_list[0].childNodes[0].data)
                cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
                cv.putText(img, objectname, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),
                           thickness=2)
                # cv.imshow('head', img)
                cv_imwrite(save_path + '/' + filename, img)  # save picture


draw_anchor(ImgPath, AnnoPath, save_path)

