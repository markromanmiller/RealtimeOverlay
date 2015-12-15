LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_SHARED_LIBRARIES += -lopencv_java3

include /home/mark/miniprojects/OpenCV-android-sdk/sdk/native/jni/OpenCV.mk

LOCAL_MODULE    := helloworld
LOCAL_SRC_FILES := com_mrmallironmaker_realtimeoverlay_MainActivity.cpp
LOCAL_LDLIBS +=  -llog -ldl

include $(BUILD_SHARED_LIBRARY)