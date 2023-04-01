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
    def __init__(self, topic_rgbImg, camFov_vertical, camFov_horizontal):
        # Image FOV for trig calculations
        self.camFov_vertical = camFov_vertical
        self.camFov_horizontal = camFov_horizontal

        # Messages
        self.msg_tfStamped              = TransformStamped()
        self.msg_targetStatus           = "?" # String
        self.msg_targetPoint            = PointStamped()   # Point
        self.msg_targetPoint.header.frame_id = "target"
        self.msg_targetCroppedRgbImg    = Image()
        self.msg_targetSkeletonImg      = Image()
        self.msg_rgbImg                 = None      # Image

        # To tell if there's a new msg
        self.newRgbImg = False

        # Publishers and Subscribers
        self.pub_tf = rospy.Publisher(
            "/tf", tfMessage, queue_size=1)
        self.pub_targetStatus = rospy.Publisher(
            "vision/target/status", String, queue_size=10)
        self.pub_targetCroppedRgbImg = rospy.Publisher(
            "vision/target/torso", Image, queue_size=10)
        self.pub_targetSkeletonImg = rospy.Publisher(
            "vision/target/with_skeleton", Image, queue_size=10)
        self.sub_rgbImg = rospy.Subscriber(
            topic_rgbImg, Image, self.callback_rgbImg)

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
        self.mp_pose = mp.solutions.pose

        # Calls main loop
        self.pose = self.mp_pose.Pose(
            # Pose Configurations
            min_detection_confidence=0.75,
            min_tracking_confidence=0.9)
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
        poseResults = self.pose.process(cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB))

        # To draw the hand annotations on the image
        cvImg.flags.writeable = True

        # Back to BGR
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_RGB2BGR)

        # Returns
        return cvImg, poseResults

    def DrawLandmarks(self, cv_rgbImg, poseResults):
        self.mp_drawing.draw_landmarks(
            cv_rgbImg,
            poseResults.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())


# Torso data processing
    ''' Gets points for torso (shoulders and hips) '''
    def GetTorsoPoints(self, landmark):
        rightShoulder = Point(
            landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x, 
            landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y, 
            landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z)
        leftShoulder = Point(
            landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
            landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y, 
            landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z)
        rightHip = Point(
            landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x, 
            landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y, 
            landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].z)
        leftHip = Point(
            landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x, 
            landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y, 
            landmark[self.mp_pose.PoseLandmark.LEFT_HIP].z)
        return [rightShoulder, leftShoulder, rightHip, leftHip]


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

# Transformation tree methods
    def SetupTfMsg(self, x, y, z):
        self.msg_tfStamped.header.frame_id = "camera_link"
        self.msg_tfStamped.header.stamp = rospy.Time.now()
        self.msg_tfStamped.child_frame_id = "target"
        self.msg_tfStamped.transform.translation.x = 0
        self.msg_tfStamped.transform.translation.y = 0
        self.msg_tfStamped.transform.translation.z = 0
        self.msg_tfStamped.transform.rotation.x = 0.0
        self.msg_tfStamped.transform.rotation.y = 0.0
        self.msg_tfStamped.transform.rotation.z = 0.0
        self.msg_tfStamped.transform.rotation.w = 1.0

        msg_tf = tfMessage([self.msg_tfStamped])
        self.pub_tf.publish(msg_tf)

# Nodes Publish
    def PublishEverything(self):
        self.pub_targetCroppedRgbImg.publish(self.msg_targetCroppedRgbImg)
        self.pub_targetStatus.publish(self.msg_targetStatus)
        self.pub_targetSkeletonImg.publish(self.msg_targetSkeletonImg)
        # self.SetupTfMsg(self.msg_targetPoint.point.x, self.msg_targetPoint.point.y, self.msg_targetPoint.point.z)

# Main
    def mainLoop(self):
        while rospy.is_shutdown() == False:
            self.loopRate.sleep()
            self.PublishEverything()
                
            # Else -> new RGB image is true...
            if self.newRgbImg == True:
                self.newRgbImg = False

                cv_rgbImg, poseResults = self.ProcessImg(self.msg_rgbImg)
                self.DrawLandmarks(cv_rgbImg, poseResults)
                self.msg_targetSkeletonImg = self.cvBridge.cv2_to_imgmsg(cv_rgbImg)
                cv2.imshow('MediaPipe Pose', cv_rgbImg)
                if cv2.waitKey(5) & 0xFF == 27: break

                # If found landmarks...
                if poseResults.pose_landmarks:
                    torsoPoints = self.GetTorsoPoints(poseResults.pose_landmarks.landmark)
                    torsoCenter = self.GetPointsMean(torsoPoints)

                    # Tries to crop torso
                    try:
                        croppedRgbImg = self.CropTorsoImg(cv_rgbImg, "passthrough", torsoPoints, torsoCenter)
                        self.msg_targetCroppedRgbImg = self.cvBridge.cv2_to_imgmsg(croppedRgbImg)
                        cv2.imshow("Cropped RGB", croppedRgbImg)
                        if cv2.waitKey(5) & 0xFF == 27: break
                    except:
                        continue

                # If nothing was detected...
                else:
                    t_now = rospy.get_time()
                    if (t_now - self.t_last > self.t_timeout and self.msg_targetStatus != "?"):
                        self.t_last = t_now
                        self.msg_targetPoint.point = Point(0, 0, 0)
                        self.msg_targetStatus = "?"

if __name__ == "__main__":
    lockHand = LockPose(
        "/camera/rgb/image_raw",
        43,
        57)