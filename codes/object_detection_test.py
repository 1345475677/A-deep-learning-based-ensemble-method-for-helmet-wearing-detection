from __future__ import division
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 03 13:57:45 2018

@author: weimin
"""

import numpy as np
import pandas as pd
import sys, os
import tensorflow as tf
import cv2, csv
import shutil

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



#标签的路径
PATH_TO_LABELS = 'D:\\anaconda\envs\helmet_detection\Lib\site-packages\models/research\object_detection/training\helmet_label_map.pbtxt'
#模型的路径
PATH_TO_CKPT = 'D:\\anaconda\envs\helmet_detection\Lib\site-packages\models/research\object_detection/training/4/frozen_inference_graph.pb'


NUM_CLASSES = 2
detection_graph = tf.Graph()

def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img
def cv_imwrite(save_Path,img):
    cv2.imencode('.jpg', img)[1].tofile(save_Path)




with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

category_index = label_map_util.create_category_index(categories)




def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):

    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    if 'detection_masks' in tensor_dict:

        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)

        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


    output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})


    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def detect_singe_img(image_path,write_path,filename):
    img = cv2.imread(image_path)
    print(img.shape)
    size = img.shape
    output_dict = run_inference_for_single_image(img, detection_graph)
    img,bndbox_helmet= vis_util.visualize_boxes_and_labels_on_image_array(#this functuin was modified
      img,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=4,
      min_score_thresh=0.74
      )
    cv2.imwrite(os.path.join(write_path,filename),img)
    return bndbox_helmet, size


def detect_img_for_video(img):
    output_dict = run_inference_for_single_image(img, detection_graph)
    img, bndbox_helmet = vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=.50
    )
    return img


def detect_file_imgs(PATH_TO_TEST_IMAGES_DIR,PATH_TO_WRITE):
  imagelist = os.listdir(PATH_TO_TEST_IMAGES_DIR)
  if os.path.exists(PATH_TO_WRITE):
    shutil.rmtree(PATH_TO_WRITE)
    os.makedirs(PATH_TO_WRITE)
  else:
    os.makedirs(PATH_TO_WRITE)
  bndbox_all = []
  for filename in imagelist:
      image_path=os.path.join(PATH_TO_TEST_IMAGES_DIR,filename)
      bndbox_helmet, size = detect_singe_img(image_path,PATH_TO_WRITE,filename)



def object_detection_test(video_path,save_path):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # 标签的路径
    PATH_TO_LABELS = 'D:\\anaconda\models/research\object_detection/training1\helmet_label_map.pbtxt'
    # 模型的路径
    PATH_TO_CKPT = 'D:\\anaconda\models/research\object_detection/training1/6/frozen_inference_graph.pb'

    #读入模型
    NUM_CLASSES = 2
    detection_graph = tf.Graph()

    # 读入模型
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    # {1: {'id': 1, 'name': 'person'}, 2: {'id': 2, 'name': 'hat'}}
    category_index = label_map_util.create_category_index(categories)


    def run_inference_for_single_image(image, graph):

        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        if 'detection_masks' in tensor_dict:

            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict


    def detect_img_for_video(img):
        output_dict = run_inference_for_single_image(img, detection_graph)
        img, bndbox_helmet = vis_util.visualize_boxes_and_labels_on_image_array(
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=4,
            min_score_thresh=.60
        )
        return img

    video = cv2.VideoCapture(video_path)
    success, img = video.read()
    success = 1
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sum = 1
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(save_path, fourcc, 20, size)
    with detection_graph.as_default():
        with tf.Session() as sess:
            while (success and video.isOpened()):
                print(sum)
                sum += 1
                if(sum==50):
                    break
                img = detect_img_for_video(img)
                out.write(img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                success, img = video.read()

    out.release()
    video.release()

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #test 文件的读取路径
    print(os.getcwd())
    PATH_TO_TEST_IMAGES_DIR='../../../helmet_5_test_data\data1/test/JPEGNEW/'

    PATH_TO_WRITE='../data/test_results/'
    imagelist = os.listdir(PATH_TO_TEST_IMAGES_DIR)
    if os.path.exists(PATH_TO_WRITE):
        shutil.rmtree(PATH_TO_WRITE)
        os.makedirs(PATH_TO_WRITE)
    else:
        os.makedirs(PATH_TO_WRITE)
    bndbox_dic_all = {}

  # save to csv
    columns_name = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax','score']
    dflist=[]
    with detection_graph.as_default():
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            for filename in imagelist:
                image_path=os.path.join(PATH_TO_TEST_IMAGES_DIR,filename)
                bndbox_helmet, size = detect_singe_img(image_path,PATH_TO_WRITE,filename)
                bndbox_dic_all[filename] = bndbox_helmet
            # write to csv
                if len(bndbox_helmet)==0:
                    row = [filename,""]
                    dflist.append(row)
                else:
                    for i in range(len(bndbox_helmet)):
                        row = [filename,bndbox_helmet[i][0],int(bndbox_helmet[i][1]*size[1]),int(bndbox_helmet[i][2]*size[0]),int(bndbox_helmet[i][3]*size[1]),int(bndbox_helmet[i][4]*size[0]),bndbox_helmet[i][5]]
                        print(row)
                        dflist.append(row)
    df=pd.DataFrame(dflist, columns=columns_name)
    df.to_csv('D:\学习\helmet_5_test_data\data1/test/Faster_RCNN008.4.csv')
    print('saved!')

