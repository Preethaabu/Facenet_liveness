from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from PIL import Image
import tensorflow.compat.v1 as tf
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
video= 0
modeldir = r'E:\Projects\liveness_integration\model\20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy=r'E:\Projects\liveness_integration\npy'
train_img=r"E:\Projects\liveness_integration\train_img"

# Path to your liveness anti-spoof model
anti_spoof_model_dir = r"E:\Projects\liveness_integration\FaceSpoof\resources\anti_spoof_models"

# Initialize the anti-spoof model
anti_spoof_model = AntiSpoofPredict(0)  # Set the device ID according to your system

# Initialize the image cropper
image_cropper = CropImage()

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        minsize = 30  # minimum size of face
        threshold = [0.7,0.8,0.8] # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        batch_size =100 #1000
        image_size = 182
        input_image_size = 160
        HumanNames = os.listdir(train_img)
        HumanNames.sort()
        print('Loading Model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile,encoding='latin1')

        video_capture = cv2.VideoCapture(0)
        print('Start Recognition')
        while True:
            ret, frame = video_capture.read()
            #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
            timer =time.time()
            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)
            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            faceNum = bounding_boxes.shape[0]
            if faceNum > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(frame.shape)[0:2]
                cropped = []
                scaled = []
                scaled_reshape = []
                for i in range(faceNum):
                    emb_array = np.zeros((1, embedding_size))
                    xmin = int(det[i][0])
                    ymin = int(det[i][1])
                    xmax = int(det[i][2])
                    ymax = int(det[i][3])
                    try:
                        # inner exception
                        if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                            print('Face is very close!')
                            continue
                        cropped.append(frame[ymin:ymax, xmin:xmax,:])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        if best_class_probabilities>0.90:
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)    #boxing face
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]
                                    print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                    cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                    cv2.putText(frame, result_names, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 0), thickness=1, lineType=1)
                                    # Perform liveness detection on the recognized face
                                    img = frame[ymin:ymax, xmin:xmax, :]
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format

                                    # Perform liveness detection using the anti-spoof model
                                    image_bbox = anti_spoof_model.get_bbox(img)
                                    prediction = np.zeros((1, 3))

                                    # Sum the prediction from single model's result
                                    for model_name in os.listdir(anti_spoof_model_dir):
                                        h_input, w_input, model_type, scale = parse_model_name(model_name)
                                        param = {
                                            "org_img": img,
                                            "bbox": image_bbox,
                                            "scale": scale,
                                            "out_w": w_input,
                                            "out_h": h_input,
                                            "crop": True,
                                        }
                                        if scale is None:
                                            param["crop"] = False
                                        img_cropped = image_cropper.crop(**param)
                                        prediction += anti_spoof_model.predict(img_cropped,
                                                                               os.path.join(anti_spoof_model_dir,
                                                                                            model_name))

                                    # Draw the result of liveness detection
                                    label = np.argmax(prediction)
                                    value = prediction[0][label] / 2
                                    if label == 1:
                                        liveness_label = "fake"
                                        color = (0, 0, 255)  # Green color for real face
                                    else:
                                        liveness_label = "real"
                                        color = (0, 255, 0)  # Red color for spoof face

                                    cv2.putText(frame, liveness_label, (xmin, ymin - 40),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, color, thickness=1, lineType=1)
                        else :
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                            cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                            cv2.putText(frame, "?", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 0), thickness=1, lineType=1)
                    except:

                        print("error")

            endtimer = time.time()
            fps = 1/(endtimer-timer)
            cv2.rectangle(frame,(15,30),(135,60),(0,255,255),-1)
            cv2.putText(frame, "fps: {:.2f}".format(fps), (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imshow('Face Recognition', frame)
            key= cv2.waitKey(1)
            if key== 113: # "q"
                break
        video_capture.release()
        cv2.destroyAllWindows()