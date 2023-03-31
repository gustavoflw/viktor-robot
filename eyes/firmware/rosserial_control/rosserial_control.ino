#include <Servo.h>
#include <ros.h>
#include <std_msgs/Int32.h>

Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
Servo servo5;
Servo servo6;

ros::NodeHandle nh;

void servo1_cb(const std_msgs::Int32& msg) {
  servo1.write(msg.data);
}

void servo2_cb(const std_msgs::Int32& msg) {
  servo2.write(msg.data);
}

void servo3_cb(const std_msgs::Int32& msg) {
  servo3.write(msg.data);
}

void servo4_cb(const std_msgs::Int32& msg) {
  servo4.write(msg.data);
}

void servo5_cb(const std_msgs::Int32& msg) {
  servo5.write(msg.data);
}

void servo6_cb(const std_msgs::Int32& msg) {
  servo6.write(msg.data);
}

ros::Subscriber<std_msgs::Int32> servo1_sub("servo1", &servo1_cb);
ros::Subscriber<std_msgs::Int32> servo2_sub("servo2", &servo2_cb);
ros::Subscriber<std_msgs::Int32> servo3_sub("servo3", &servo3_cb);
ros::Subscriber<std_msgs::Int32> servo4_sub("servo4", &servo4_cb);
ros::Subscriber<std_msgs::Int32> servo5_sub("servo5", &servo5_cb);
ros::Subscriber<std_msgs::Int32> servo6_sub("servo6", &servo6_cb);

void setup() {
  nh.initNode();
  nh.subscribe(servo1_sub);
  nh.subscribe(servo2_sub);
  nh.subscribe(servo3_sub);
  nh.subscribe(servo4_sub);
  nh.subscribe(servo5_sub);
  nh.subscribe(servo6_sub);
  
  servo1.attach(9);
  servo2.attach(10);
  servo3.attach(11);
  servo4.attach(12);
  servo5.attach(13);
  servo6.attach(14);
}

void loop() {
  nh.spinOnce();
  delay(1);
}