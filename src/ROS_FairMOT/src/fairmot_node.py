import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq_single

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import numpy as np
import cv2
from cv_bridge import CvBridge

logger.setLevel(logging.INFO)

ROS_Img = None
results = []

tracker = None

def preprocessing(ROS_Img):
    bridge = CvBridge()
    img0 = bridge.imgmsg_to_cv2(ROS_Img, desired_encoding="passthrough")

    width = 1088
    height = 608
    w, h = 1920, 1080
    print("IMG0 Orig Shape", img0.shape)

    img0 = cv2.resize(img0, (w, h))

    # Padded resize
    img, _, _, _ = letterbox(img0, height, width)

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0

    
    return img, img0


    # res, img0 = self.cap.read()  # BGR
    # assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
    # img0 = cv2.resize(img0, (self.w, self.h))

    # # Padded resize
    # img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

    # # Normalize RGB
    # img = img[:, :, ::-1].transpose(2, 0, 1)
    # img = np.ascontiguousarray(img, dtype=np.float32)
    # img /= 255.
    
    # res, img0 = self.cap.read()  # BGR
    # assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
    # img0 = cv2.resize(img0, (self.w, self.h))

    # # Padded resize
    # img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

    # # Normalize RGB
    # img = img[:, :, ::-1].transpose(2, 0, 1)
    # img = np.ascontiguousarray(img, dtype=np.float32)
    # img /= 255.0


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def demo(opt, ROS_Img, frame_id):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    # dataloader = datasets.LoadVideo(input_file, opt.img_size)
    # print(init)
    img, img0 = preprocessing(ROS_Img)

    result_filename = os.path.join(result_root, 'results.txt')
    #frame_rate = dataloader.frame_rate
    frame_rate = 10
    

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    #frame_dir = "../demos/img_results"
    print(opt.output_root)
    eval_seq_single(opt, img, img0, tracker, results, frame_id, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate, )
    
    # if opt.output_format == 'video':
    #     output_video_path = osp.join(result_root, 'MOT16-03-results.mp4')
    #     cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
    #     os.system(cmd_str)

class FairMOTNode(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.frame_id = 0
        self.ROS_Img = None
        self.subscription = self.create_subscription(
            Image,
            '/kitti/camera_2_image',
            self.listener_callback,
            100)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.ROS_Img = msg
        #print(ROS_Img)
        self.get_logger().info('I heard: "%s"' % msg.header)
        demo(opt, msg, self.frame_id)
        self.frame_id = self.frame_id + 1
        #rospy.Timer(rospy.Duration(10000), stop_callback)
    
    def get_ROS_Img(self):
        return self.ROS_Img

def main(args=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("============================HEREEEE1=============================")
    opt = opts().init()
    rclpy.init() 
    fairmot_node = FairMOTNode()

    print("============================HEREEEE2.25=============================")
    try:
        rclpy.spin(fairmot_node)
        # for i in range(150):
        #     rclpy.spin_once(minimal_subscriber)
        #preprocessing(init)
    except Exception as e:
        print("Exception:", e)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    # minimal_subscriber.destroy_node()
    print("============================HEREEEE2.5=============================")
    #rclpy.shutdown()


    print("============================HEREEEE3=============================")
if __name__ = '__main__':
    main()
