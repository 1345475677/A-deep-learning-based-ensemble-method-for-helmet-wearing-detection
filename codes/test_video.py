from __future__ import division
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import cv2
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from codes.Enlarge_picture_in_wanted_place import class_judge_video
from video_test import video_test_Tiny_face
from codes.csv_merge import merge_for_video
from sklearn.externals import joblib

def object_detection_test(video_path,save_path=None):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    PATH_TO_LABELS = 'helmet_label_map.pbtxt'

    PATH_TO_CKPT = 'frozen_inference_graph.pb'


    NUM_CLASSES = 2
    detection_graph = tf.Graph()


    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    # [{'id': 1, 'name': 'person'}, {'id': 2, 'name': 'hat'}]
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    # {1: {'id': 1, 'name': 'person'}, 2: {'id': 2, 'name': 'hat'}}
    category_index = label_map_util.create_category_index(categories)


    def run_inference_for_single_image(image, graph):

        ops = tf.compat.v1.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
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
        image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

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
        for i in range(len(bndbox_helmet)):
            bndbox_helmet[i][1]=bndbox_helmet[i][1]*img.shape[1]
            bndbox_helmet[i][2]=bndbox_helmet[i][2]*img.shape[0]
            bndbox_helmet[i][3]=bndbox_helmet[i][3]*img.shape[1]
            bndbox_helmet[i][4]=bndbox_helmet[i][4]*img.shape[0]
        return img,bndbox_helmet

    video = cv2.VideoCapture(video_path)
    success, img = video.read()
    success = 1
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sum = 0
    k=0
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))

    df1=pd.DataFrame(columns=['frame','class','xmin','ymin','xmax','ymax','score'])
    with detection_graph.as_default():
        with tf.compat.v1.Session() as sess:
            while (success and video.isOpened()):
                img,list = detect_img_for_video(img)
                for i in range(len(list)):
                    df1.loc[k,'frame']=sum
                    df1.loc[k,'class']=list[i][0]
                    df1.loc[k,'xmin']=list[i][1]
                    df1.loc[k,'ymin']=list[i][2]
                    df1.loc[k,'xmax']=list[i][3]
                    df1.loc[k,'ymax']=list[i][4]
                    df1.loc[k,'score']=list[i][5]
                    k+=1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                success, img = video.read()
                sum += 1

    video.release()
    return df1


def Tiny_face_test(video_path,save_path=None):
    df=video_test_Tiny_face(video_path)
    print(df)
    df=class_judge_video(video_path,df)
    return df



def draw_for_video(video_path,df,save_path):
    video = cv2.VideoCapture(video_path)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(save_path, fourcc, 5, size)
    success, img = video.read()
    i=0
    while (success and video.isOpened() and i <len(df)):
        xmin = int(df.loc[i, 'xmin'])
        xmax = int(df.loc[i, 'xmax'])
        ymin = int(df.loc[i, 'ymin'])
        ymax = int(df.loc[i, 'ymax'])
        objectname = df.loc[i, 'class']
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), thickness=2)
        cv2.putText(img, objectname, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),
                   thickness=2)

        if (i < len(df) - 1):
            if (df.loc[i + 1, 'frame'] > df.loc[i, 'frame']):
                out.write(img)
                success, img = video.read()
        i += 1
    out.write(img)
    out.release()
    video.release()


def boost_video(df):
    adaboost_path='adaboost_model/model.m'
    adaboost=joblib.load(adaboost_path)
    df.insert(loc=len(df.columns)-2, column='area', value=0)
    for i in range(len(df)):
        df.loc[i,'area']=(df.loc[i,'xmax']-df.loc[i,'xmin'])*(df.loc[i,'ymax']-df.loc[i,'ymin'])
    X = np.array(df[['area', 'score', 'score2']])
    label=adaboost.predict(X)
    for i in range(len(label)):
        if label[i] == 0:
            df = df.drop(i)
    return df




def main():#视频部分结束，还差集成
    video_path="D:\学习\服务外包竞赛\Object-Detection_HelmetDetection-master\data/test/安全帽2.avi"
    save_path="D:\学习\服务外包竞赛\Object-Detection_HelmetDetection-master\data/test/安全帽3.avi"
    df1=object_detection_test(video_path)
    df2=Tiny_face_test(video_path)
    df=merge_for_video(df1,df2)
    df=df.fillna(int(0))
    df=boost_video(df)
    draw_for_video(video_path,df,save_path)

if __name__ == '__main__':
    main()