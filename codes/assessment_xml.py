import xml.dom.minidom
import numpy as np
import glob
import os

def read_xml(file_path,object_name='hat'):
    dom=xml.dom.minidom.parse(file_path)
    root=dom.documentElement
    all_objectt=root.getElementsByTagName('object')
    path=root.getElementsByTagName('path')[0].childNodes[0].data
    all_box=[]

    for objectt in all_objectt:
        name=objectt.getElementsByTagName('name')[0]
        #print(name.childNodes[0].data)
        box=[]
        if object_name==name.childNodes[0].data:
            bndbox=objectt.getElementsByTagName('bndbox')[0]
            xmin=bndbox.getElementsByTagName('xmin')[0].childNodes[0].data
            ymin=bndbox.getElementsByTagName('ymin')[0].childNodes[0].data
            xmax=bndbox.getElementsByTagName('xmax')[0].childNodes[0].data
            ymax=bndbox.getElementsByTagName('ymax')[0].childNodes[0].data
            box.append(xmin)
            box.append(ymin)
            box.append(xmax)
            box.append(ymax)
            all_box.append(box)
    all_box=np.array(all_box)
    return all_box,path


def assessment(right_box,test_box):
    right_box_len=len(right_box)
    test_box_len=len(test_box)



    def calc_overarea(right,test):
        t_xmin,t_ymin,t_xmax,t_ymax=test[0],test[1],test[2],test[3]
        r_xmin,r_ymin,r_xmax,r_ymax=right[0],right[1],right[2],right[3]
        xmin=max(float(t_xmin),float(r_xmin))
        ymin=max(float(t_ymin),float(r_ymin))
        xmax=min(float(t_xmax),float(r_xmax))
        ymax=min(float(t_ymax),float(r_ymax))
        width=xmax-xmin
        height=ymax-ymin
        if width<=0 or height<=0:
            return 0
        else:
            return width*height


    def calc_area(a):
        xmin,ymin,xmax,ymax=a[0],a[1],a[2],a[3]
        width=float(xmax)-float(xmin)
        height=float(ymax)-float(ymin)
        return width*height


    all_result=[]

    for test in test_box:
        for right in right_box:
            over_area=calc_overarea(right,test)
            right_area=calc_area(right)
            test_area=calc_area(test)
            if over_area/right_area<0.3:
                continue
            else:
                result=[]
                result.append(over_area/right_area)
                result.append((test_area-over_area)/test_area)
                all_result.append(result)
                print("正确占比",over_area/right_area)
                print("多余占比",(test_area-over_area)/test_area)
                print()

    lack=right_box_len-len(all_result)
    wrong=test_box_len-len(all_result)
    print('少标了',lack,'个')
    print('错标了',wrong,'个')

    return all_result,wrong,lack
            




def arrange_xml(image_dir,xml_dir):
    image_names = []
    image_paths=[]
    xml_paths=[]



    image_names=os.listdir(image_dir)
    for ext in ('*.png', '*.gif', '*.jpg', '*.jpeg'):
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    xml_paths.extend(glob.glob(os.path.join(xml_dir, '*.xml')))
    
    for i in range(len(image_paths)):
        image_paths[i]=image_paths[i].replace('\\','/')

    

    def change_xml(image_path,image_name,xml_path):

        dom=xml.dom.minidom.parse(xml_path)
        root=dom.documentElement
        root.getElementsByTagName('filename')[0].childNodes[0].nodeValue=image_name
        root.getElementsByTagName('path')[0].childNodes[0].nodeValue=image_path
        with open(xml_path,'w',encoding='UTF-8') as fh:
            dom.writexml(fh)
            
    

    if len(image_names)==len(xml_paths):
        for i in range(len(image_names)):
            change_xml(image_paths[i],image_names[i],xml_paths[i])
    else:
        print('error')

if __name__=="__main__":
    arrange_xml('D:\学习\helmet_5_test_data\data5/val\JPEGImages','D:\学习\helmet_5_test_data\data5/val\Annotations')


'''
a=read_xml('D:/service2019/测试数据/Annotations/000002.xml')
c=np.array([[162.53139033328858, 100.5994480406164, 208.40066173877523, 127.01952603835335, 11.291155815124512], [409.8603102584695, 77.03811132388674, 457.17480464356544, 103.6677541198777, 11.010414123535156], [345.66010933757786, 108.05855331181364, 388.6716935059144, 132.93440323318478, 10.47685432434082], [336.9657804533182, 61.50911698572206, 376.7007853971642, 83.28908099675024, 9.811666488647461], [179.03810076714709, 68.93443523245934, 215.94453267231475, 90.21147761611076, 8.430644035339355], [247.97244894479212, 63.897766514025406, 284.29753708838734, 84.67341284900708, 7.590930938720703], [12.573500325831734, 71.67753405870366, 53.828179289825336, 96.27161204481386, 6.808356761932373], [40.14720599399676, 40.81461142018881, 75.43458664432788, 61.14612089506134, 5.991604328155518], [366.97190379068354, 61.499156128116056, 402.7801192361978, 82.81635174333647, 5.822369575500488], [222.30200441112996, 47.63926485079481, 254.67470030088364, 65.20422766124112, 5.448849201202393], [94.05988038914012, 48.957104696090774, 120.7841986090277, 64.26057167179287, 2.7467586994171143]])
assessment(a,c)
'''
'''
image_dir='D:/service2019/测试数据/JPEGImages'
xml_dir='D:/service2019/测试数据/Annotations'
arrange_xml(image_dir,xml_dir)
'''
'''
a,b=read_xml('D:/service2019/测试数据/Annotations/000002.xml')
print(a)
print(b)
'''