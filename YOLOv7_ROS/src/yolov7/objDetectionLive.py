#!/usr/bin/env python

#from ast import main
from configparser import Interpolation
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import ros_numpy
import cv2

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.datasets import letterbox

weights="/home/avalocal/catkin_ws/src/YOLOv7_ROS/src/yolov7/yolov7.pt"
source=0  #'inference/images'
img_size=640
conf_thres=0.7
iou_thres=0.5
device=torch.device('cuda:0') #torch.cuda.current_device() #cuda device, i.e. 0 or 0,1,2,3 or cpu'
view_img=True
save_conf=True
nosave=True
classes=None #'filter by class: --class 0, or --class 0 2 3'
agnostic_nms=False
augment=True
no_trace=True #'don`t trace model'
trace=False

class detect():

    def __init__(self):
        self.weights=weights
        self.device=device
        self.img_size=img_size
        self.iou_thres=iou_thres
        self.augment=augment
        self.view_img=view_img
        self.classes=classes
        self.agnostic_nms=agnostic_nms
        self.w=0
        self.h=0
        self.conf_thres=conf_thres
        self.classify=False
        self.half=self.device !='cpu' #half precision
    
        # Load model 
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        print("model is here", self.model)
        self.stride = int(self.model.stride.max())  # model stride
        print("stride", self.stride)
        self.img_size = check_img_size(self.img_size, s=self.stride)  # check img_size
        print("image size is", self.img_size)

        if trace:
            self.model = TracedModel(self.model, device, self.img_size)
            print("model is traced")

        if self.half:
            self.model.half()  # to FP16
            print("model is halfed")
       
        # Second-stage classifier
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()
            print("classifyed model is used as well")
        cudnn.benchmark = True  # set True to speed up constant image size inference

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        print("names and colors are set")

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once
            print("interference is running")
        self.image_pub = rospy.Publisher("~published_image", Image, queue_size=1)  
        self.image_sub = rospy.Subscriber("/kitti/camera_color_left/image_raw", Image, self.camera_callback, queue_size=1, buff_size=2**24) #"/camera_fr/image_raw"
        rospy.spin()
        


    def camera_callback(self, data):
        try:
            self.img = ros_numpy.numpify(data)
            #print("image is read from subscriber and shape is :", np.shape(self.img)) #(1544, 2048) numpy
            #self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:
           print("something is wrong with camera_callback function")
        
        img0=self.img
        #img0=cv2.resize(img0, (640,640))
    
        img=letterbox(img0, self.img_size, stride=self.stride)[0]
        #print(np.shape(img)) 
        img = img[:, :, ::-1].transpose(2, 0, 1)  #BGR to RGB
        img = np.ascontiguousarray(img)
        img=self.preProccess(img)
        
        with torch.no_grad():

            pred=self.model(img, augment=self.augment)[0]
            pred=non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

            if self.classify:
                pred=apply_classifier(pred, self.modelc, img, img0)

            for i,det in enumerate(pred):
                if len(det):
                    #Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    # Print results
                    #for c in det[:, -1].unique():
                        #n = (det[:, -1] == c).sum()  # detections per class
                        #s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    for *xyxy, conf, cls in reversed(det):
                        if view_img:
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=3)

            if self.view_img:
                image_out=img0
                #cv2.imshow('result', img0)
                #cv2.waitKey(1)
                image_out=image_out[...,::-1]
                img_out = im.fromarray(image_out,'RGB')
                msg=Image()
                msg.header.stamp=rospy.Time.now()
                msg.height=img_out.height
                msg.width=img_out.width
                msg.encoding="rgb8"
                msg.is_bigendian=False
                msg.step=3*img_out.width
                msg.data=np.array(img_out).tobytes()
                
                self.image_pub.publish(msg)

                rospy.Rate(50).sleep()

    def preProccess(self, img):
        imgs=torch.from_numpy(img).to(self.device)
        img=imgs.half() if self.half else img.float()  # uint8 to fp16/32
        img=img/255.0
        if img.ndimension()==3:
            img=img.unsqueeze(0)
        return img
        

    

if __name__ == '__main__':

    rospy.init_node('yoloLiveNode')
    detect()
    
