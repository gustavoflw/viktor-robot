#!/usr/bin/env python

# Mediapipe imports
import cv2
import mediapipe as mp

# Image processing
from cv_bridge import CvBridge

# ROS
import rospy
## Message definitions
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Point, TransformStamped
from sensor_msgs.msg import Image
## Transformation tree
from tf.msg import tfMessage
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler

# Math 
import numpy as np
from math import pow, sqrt, tan, radians

class LockPose():
    def __init__(self):

        # Messages
        self.msg_tfStamped              = TransformStamped()
        self.msg_targetStatus           = "?" # String
        self.msg_targetPoint            = PointStamped()   # Point
        self.msg_targetPoint.header.frame_id = "target"
        self.msg_targetTorsoRgbImg      = Image()
        self.msg_targetFaceRgbImg       = Image()
        self.msg_targetSkeletonImg      = Image()
        self.msg_rgbImg                 = None      # Image

        # To tell if there's a new msg
        self.newRgbImg = False

        # Publishers
        self.pub_tf = rospy.Publisher(
            "/tf", tfMessage, queue_size=1)
        self.pub_targetStatus = rospy.Publisher(
            "vision/target/status", String, queue_size=10)
        self.pub_targetTorsoRgbImg = rospy.Publisher(
            "vision/target/torso", Image, queue_size=10)
        self.pub_targetFaceRgbImg = rospy.Publisher(
            "vision/target/face", Image, queue_size=10)
        self.pub_targetSkeletonImg = rospy.Publisher(
            "vision/target/with_skeleton", Image, queue_size=10)
        
        # Subscribers
        self.sub_rgbImg = rospy.Subscriber(
            "/camera/rgb/image_raw", Image, self.callback_rgbImg)

        # ROS node
        rospy.init_node('locker_human', anonymous=True)

        # Time
        self.loopRate = rospy.Rate(30)
        self.t_last = 0.0  # sec
        self.t_timeout = 0.250  # sec

        # Cv
        self.cvBridge = CvBridge()

        # Mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=True)

        # Calls main loop
        self.mainLoop()

# Callbacks
    def callback_rgbImg(self, msg):
        self.msg_rgbImg = msg
        self.newRgbImg = True
        # print("- RGB: new msg")

# Basic MediaPipe Pose methods
    def ProcessImg(self, msg_img):
        # Conversion to cv image
        cvImg = self.cvBridge.imgmsg_to_cv2(self.msg_rgbImg, "bgr8")

        # Not writeable passes by reference (better performance)
        cvImg.flags.writeable = False

        # Converts BGR to RGB
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)

        # Image processing
        results = self.holistic.process(cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB))

        # To draw the hand annotations on the image
        cvImg.flags.writeable = True

        # Back to BGR
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR)

        # Returns
        return cvImg, results

    def DrawLandmarks(self, cv_rgbImg, results):
        self.mp_drawing.draw_landmarks(
            cv_rgbImg,
            results.face_landmarks,
            self.mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_contours_style())

        self.mp_drawing.draw_landmarks(
            cv_rgbImg,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles
            .get_default_pose_landmarks_style())


    ''' Gets points for torso (shoulders and hips) '''
    def GetTorsoPoints(self, landmark):
        rightShoulder = Point(
            landmark[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER].x, 
            landmark[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER].y, 
            landmark[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER].z)
        leftShoulder = Point(
            landmark[self.mp_holistic.PoseLandmark.LEFT_SHOULDER].x, 
            landmark[self.mp_holistic.PoseLandmark.LEFT_SHOULDER].y, 
            landmark[self.mp_holistic.PoseLandmark.LEFT_SHOULDER].z)
        rightHip = Point(
            landmark[self.mp_holistic.PoseLandmark.RIGHT_HIP].x, 
            landmark[self.mp_holistic.PoseLandmark.RIGHT_HIP].y, 
            landmark[self.mp_holistic.PoseLandmark.RIGHT_HIP].z)
        leftHip = Point(
            landmark[self.mp_holistic.PoseLandmark.LEFT_HIP].x, 
            landmark[self.mp_holistic.PoseLandmark.LEFT_HIP].y, 
            landmark[self.mp_holistic.PoseLandmark.LEFT_HIP].z)
        return [rightShoulder, leftShoulder, rightHip, leftHip]
    
    def GetFacePoints(self, landmark):
        # Defines which points belong to face outer part
        landmark_points_68 = np.array([
            162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
            296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
            380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87])
        
        points_x = np.zeros(68)
        points_y = np.zeros(68)
        i = 0
        for index in landmark_points_68:
            points_x[i] = landmark[index].x
            points_y[i] = landmark[index].y
            i = i + 1
        return points_x, points_y


    def CropTorsoImg(self, img, imgEncoding, torsoPoints, torsoCenter):
        if imgEncoding == "32FC1":
            imageHeight, imageWidth = img.shape
        else:
            imageHeight, imageWidth, a = img.shape
        torsoWidth = max(abs(torsoPoints[0].x - torsoPoints[1].x) * imageWidth, 1)
        torsoHeight = max(abs(torsoPoints[0].y - torsoPoints[2].y) * imageHeight, 1)

        x0 = max(int(torsoCenter.x * imageWidth - torsoWidth/2), 0)
        y0 = max(int(torsoCenter.y * imageHeight - torsoHeight/2), 0)
        xf = min(int(torsoCenter.x * imageWidth + torsoWidth/2), imageWidth)
        yf = min(int(torsoCenter.y * imageHeight + torsoHeight/2), imageHeight)

        cropped_image = img[y0:yf, x0:xf]
        return cropped_image
    
    def CropFaceImg(cv_rgbImg, img, imgEncoding, facePoints_x, facePoints_y):
        if imgEncoding == "32FC1":
            imageHeight, imageWidth = img.shape
        else:
            imageHeight, imageWidth, a = img.shape

        facePixels_x = np.array(facePoints_x) * imageWidth
        facePixels_y = np.array(facePoints_y) * imageHeight

        # Minimum pixels
        x0 = max(int(facePixels_x.min()), 0)
        y0 = max(int(facePixels_y.min()), 0)

        # Maximum pixels
        xf = min(int(facePixels_x.max()), imageWidth)
        yf = min(int(facePixels_y.max()), imageHeight)

        cropped_image = img[y0:yf, x0:xf]
        return cropped_image

# Points calculations
    def GetPointsMean(self, points):
        sum_x = 0
        sum_y = 0
        sum_z = 0
        counter = 0
        for point in points:
            sum_x = sum_x + point.x
            sum_y = sum_y + point.y
            sum_z = sum_z + point.z
            counter = counter + 1
        return Point(sum_x/counter, sum_y/counter, sum_z/counter)

    ''' Transforms the mpipe coordinate format to tf tree coordinate format'''
    def XyzToZxy(self, point):
        return Point(point.z, point.x, point.y)

# Nodes Publish
    def PublishEverything(self):
        self.pub_targetTorsoRgbImg.publish(self.msg_targetTorsoRgbImg)
        self.pub_targetFaceRgbImg.publish(self.msg_targetFaceRgbImg)
        self.pub_targetStatus.publish(self.msg_targetStatus)
        self.pub_targetSkeletonImg.publish(self.msg_targetSkeletonImg)

# Main
    def mainLoop(self):
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()
            self.PublishEverything()
                
            # Else -> new RGB image is true...
            if self.newRgbImg == True:
                self.newRgbImg = False

                cv_rgbImg, holisticResults = self.ProcessImg(self.msg_rgbImg)
                self.DrawLandmarks(cv_rgbImg, holisticResults)
                self.msg_targetSkeletonImg = self.cvBridge.cv2_to_imgmsg(cv_rgbImg)
                # cv2.imshow('MediaPipe Pose', cv_rgbImg)
                # if cv2.waitKey(5) & 0xFF == 27: break

                # If found pose landmarks...
                if holisticResults.pose_landmarks:
                    torsoPoints = self.GetTorsoPoints(holisticResults.pose_landmarks.landmark)
                    torsoCenter = self.GetPointsMean(torsoPoints)

                    # Tries to crop torso
                    try:
                        croppedRgbImg = self.CropTorsoImg(cv_rgbImg, "passthrough", torsoPoints, torsoCenter)
                        self.msg_targetTorsoRgbImg = self.cvBridge.cv2_to_imgmsg(croppedRgbImg)
                        # cv2.imshow("Cropped RGB torso", croppedRgbImg)
                        # if cv2.waitKey(5) & 0xFF == 27: break
                    except:
                        pass
                
                # If found face landmarks...
                if holisticResults.face_landmarks:
                    facePoints_x, facePoints_y = self.GetFacePoints(holisticResults.face_landmarks.landmark)

                    # Tries to crop face
                    try:
                        croppedRgbImg = self.CropFaceImg(cv_rgbImg, "passthrough", facePoints_x, facePoints_y)
                        self.msg_targetFaceRgbImg = self.cvBridge.cv2_to_imgmsg(croppedRgbImg)
                        # cv2.imshow("Cropped RGB face", croppedRgbImg)
                        # if cv2.waitKey(5) & 0xFF == 27: break
                    except Exception as e: 
                        print(e)


if __name__ == "__main__":
    lockHand = LockPose()