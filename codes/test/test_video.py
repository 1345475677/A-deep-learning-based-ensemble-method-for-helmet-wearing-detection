rom __future__ import division
import time
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import cv2
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from Enlarge_picture_in_wanted_place import class_judge_video
#from models.Tiny_Face.Tiny_Face.video_test import video_test_Tiny_face
from models.Tiny_Face.Tiny_Face import video_test_1
from csv_merge import merge_for_video
from sklearn.externals import joblib


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def object_detection_test(video_path,save_path=None):

    PATH_TO_LABELS = 'models\object_detection_API_fast_rcnn\helmet_label_map.pbtxt'

    PATH_TO_CKPT = 'models\object_detection_API_fast_rcnn\saved_model/frozen_inference_graph.pb'


    NUM_CLASSES = 2
    detection_graph = tf.Graph()


    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    # [{'id': 1, 'name': 'person'}, {'id': 2, 'name': 'hat'}]
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    # {1: {'id': 1, 'name': 'person'}, 2: {'id': 2, 'name': 'hat'}}
    category_index = label_map_util.create_category_index(categories)

    #
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
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.

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
            img,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=4,
            min_score_thresh=.60
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
   # out = cv2.VideoWriter(save_path, fourcc, 20, size)
    df1=pd.DataFrame(columns=['frame','class','xmin','ymin','xmax','ymax','score'])
    with detection_graph.as_default():
        with tf.Session() as sess:
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
                #out.write(img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                success, img = video.read()
                sum += 1
    #out.release()
    video.release()
    return df1

#from Tiny_Face import video_test
def Tiny_face_test(video_path,save_path=None):
    # df=video_test_Tiny_face(video_path)
    # df=class_judge_video(video_path,df)
    # return df
    df=video_test_1.main(video_path)
    return df
#def boost_video(df):

def boost_video(df):
    adaboost_path='models/adaboost/model.m'
    adaboost=joblib.load(adaboost_path)
    df.insert(loc=len(df.columns)-2, column='area', value=0)
    for i in range(len(df)):
        df.loc[i,'area']=(df.loc[i,'xmax']-df.loc[i,'xmin'])*(df.loc[i,'ymax']-df.loc[i,'ymin'])
    X = np.array(df[['area', 'score', 'score2']])
    label=adaboost.predict(X)
    for i in range(len(label)):
        if label[i] == 0:
            df = df.drop(i)
    df=df.reindex()
    return df

def draw_for_video(video_path,df,save_path):
    video = cv2.VideoCapture(video_path)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps=video.get(cv2.CAP_PROP_FPS)
    print("fps=",fps)
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(save_path, fourcc, fps, size)
    success, img = video.read()
    i=0
    sum=0
    while (success and video.isOpened() and i <len(df)):
        xmin = int(df.loc[i, 'xmin'])
        xmax = int(df.loc[i, 'xmax'])
        ymin = int(df.loc[i, 'ymin'])
        ymax = int(df.loc[i, 'ymax'])
        objectname = df.loc[i, 'class']
        '''
        if(objectname=='hat'):
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 0), thickness=5)
            cv2.putText(img, objectname, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),
                       thickness=2)
        '''
        if(objectname=='person'):
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=5)
            cv2.putText(img, objectname, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),
                        thickness=2)


        if (i < len(df) - 1):
            while (df.loc[i + 1, 'frame'] > sum):
                out.write(img)
                success, img = video.read()
                sum+=1
        i += 1
    out.write(img)
    out.release()
    video.release()




def main():

    video_path="test/1.avi"
    save_path=video_path.split('.')[0]+"_test.avi"
    df1=object_detection_test(video_path)

    df2=Tiny_face_test(video_path)

    df=merge_for_video(df1,df2)

    df=df.fillna(int(0))
    df=boost_video(df)

    df.to_csv(save_path.split('.')[0]+".csv",index=False)
    file=pd.read_csv(save_path.split('.')[0]+".csv",encoding='gbk')
    df=pd.DataFrame(file)
    draw_for_video(video_path,df,save_path)


def test_the_video(video_path):

    save_path = video_path.split('.')[0] + "_test.avi"
    df1 = object_detection_test(video_path)

    df2 = Tiny_face_test(video_path)

    df = merge_for_video(df1, df2)

    df = df.fillna(int(0))
    df = boost_video(df)

    df.to_csv(save_path.split('.')[0] + ".csv", index=False)
    file = pd.read_csv(save_path.split('.')[0] + ".csv", encoding='gbk')
    df = pd.DataFrame(file)
    draw_for_video(video_path, df, save_path)



def test_the_video(video_path,t):

    def get_mid_video(video_path, mid_path):
        video = cv2.VideoCapture(video_path)
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = 20
        fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
        success, img = video.read()
        out = cv2.VideoWriter(mid_path, fourcc, fps, size)
        k = 0
        while (success and video.isOpened()):
            if (k % t == 0):
                out.write(img)
            success, img = video.read()
            k += 1
        out.release()
        video.release()

    def draw_all_video(video_path, save_path, df):
        video = cv2.VideoCapture(video_path)
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = video.get(cv2.CAP_PROP_FPS)
        print("fps=", fps)
        fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
        out = cv2.VideoWriter(save_path, fourcc, fps, size)
        success, img = video.read()
        sum = 0
        i = 0
        flag = 1
        frame = 0
        j1 = 0
        i1 = 0
        while (success and video.isOpened() and i < len(df)):
            if (flag == 1):
                frame = df.loc[i, 'frame']
                j1 = i
                i1 = i
                while (df.loc[j1, 'frame'] == frame and j1 < len(df) - 1):
                    j1 += 1
                if (j1 == len(df) - 1):
                    j1 += 1
                flag = 0
                print(j1)

            if ((1 + frame) * t <= sum):
                if (j1 < len(df)):
                    i = j1
                    flag = 1
                    while(frame*t>sum):
                        out.write(img)
                        success, img = video.read()
                        sum += 1
                    continue
                else:
                    out.release()
                    video.release()
                    break

            for i in range(i1, j1):
                if(df.loc[i,'class']!="person" and df.loc[i,'class']!="hat"):
                    flag=1
                    continue
                xmin = int(df.loc[i, 'xmin'])
                xmax = int(df.loc[i, 'xmax'])
                ymin = int(df.loc[i, 'ymin'])
                ymax = int(df.loc[i, 'ymax'])
                objectname = df.loc[i, 'class']
                if (objectname == 'person'):
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=5)
                    cv2.putText(img, "not_wear", (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255),
                                thickness=2)
                elif(objectname=='hat'):
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=5)
                    cv2.putText(img,"wear_helmet", (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),
                                thickness=2)
            out.write(img)
            success, img = video.read()
            sum += 1



    save_path = video_path.split('.')[0] + "_test.avi"
    mid_path = video_path.split('.')[0] + "_mid.avi"
    print(mid_path)
    get_mid_video(video_path, mid_path)
    df1 = object_detection_test(mid_path)


    df2 = Tiny_face_test(mid_path)

    df = merge_for_video(df1, df2)
    df = df.fillna(int(0))
    df = boost_video(df)
    df.to_csv(save_path.split('.')[0] + ".csv", index=False)
    file = pd.read_csv(save_path.split('.')[0] + ".csv", encoding='gbk')
    df = pd.DataFrame(file)



    draw_all_video(video_path, save_path, df)
    os.remove(mid_path)

    os.remove(save_path.split('.')[0] + ".csv")


'''
if __name__ == '__main__':
    main()
'''