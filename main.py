# Limiting assumptions:
#   - assumes that there is only one object per ROI
#   - assumes that respiration rate is between 3 and 60 breaths/minute

from pathlib import Path
import cv2
import depthai as dai
import pandas as pd
import numpy as np
import time
import argparse
from time import monotonic
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from collections import deque
import datetime
import PySimpleGUI as sg
from scipy.signal import lombscargle

breath_frame_count = 0

# for demonstration purposes, these are set very low
delta_time_short = 1  # seconds
delta_time_medium = 3
delta_time_long = 7

nnPath    = str((Path(__file__).parent / Path('./models/mobilenet-ssd_openvino_2021.2_6shave.blob')).resolve().absolute())
videoPath = str((Path(__file__).parent / Path('./input_media/construction_vest.mp4')).resolve().absolute())

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
display_categories = ["bird", "cat", "dog", "person"]

parser = argparse.ArgumentParser()
parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)", default=videoPath)
parser.add_argument('-s', '--short', type=str, help="delta time for short activity metric in seconds", default=delta_time_short)
parser.add_argument('-m', '--medium', type=str, help="delta time for medium activity metric in seconds", default=delta_time_medium)
parser.add_argument('-l', '--long', type=str, help="delta time for long activity metric in seconds", default=delta_time_long)
parser.add_argument('-enc', '--encode', type=str, help="Path to where encoded video is saved")
parser.add_argument('-nofps', '--nofps', action="store_true", help="removes fps text in video")
parser.add_argument('-debug', '--debug', action="store_true", help="Adds debug output")
args = parser.parse_args()

delta_time_short = args.short
delta_time_medium = args.medium
delta_time_long = args.long

video = not args.camera  #this means that video mode is the default

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a neural network that will make predictions based on the source frames
# DetectionNetwork class produces ImgDetections message that carries parsed
# detection results.
nn = pipeline.createMobileNetDetectionNetwork()
nn.setBlobPath(nnPath)

nn.setConfidenceThreshold(0.5)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Define a source for the neural network input
if video:
    # Create XLinkIn object as conduit for sending input video file frames
    # to the neural network
    xinFrame = pipeline.createXLinkIn()
    xinFrame.setStreamName("inFrame")
    # Connect (link) the video stream from the input queue to the
    # neural network input
    xinFrame.out.link(nn.input)
    maxfps = 30
else:
    # Create color camera node.
    cam = pipeline.createColorCamera()
    cam.setPreviewKeepAspectRatio(False)
    cam.setPreviewSize(300, 300)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    maxfps = 5
    cam.setFps(maxfps) # cap camera fps
    cam.setInterleaved(False)
    # Connect (link) the camera preview output to the neural network input
    cam.preview.link(nn.input)

    # Create XLinkOut object as conduit for passing camera frames to the host
    xoutFrame = pipeline.createXLinkOut()
    xoutFrame.setStreamName("outFrame")
    cam.video.link(xoutFrame.input)   # switched cam.preview to cam.video

# Create neural network output (inference) stream
nnOut = pipeline.createXLinkOut()
nnOut.setStreamName("nn")
nn.out.link(nnOut.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("video", 1280, 720)
    cv2.moveWindow("video",0,0)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # Define queues for image frames
    if video:
        # Input queue for sending video frames to device
        qIn_Frame = device.getInputQueue(name="inFrame", maxSize=4, blocking=False)
    else:
        # Output queue for retrieving camera frames from device
        qOut_Frame = device.getOutputQueue(name="outFrame", maxSize=1, blocking=False)

    qDet = device.getOutputQueue(name="nn", maxSize=1, blocking=False)

    if video:
        cap = cv2.VideoCapture(args.video)
    def should_run():
        return cap.isOpened() if video else True

    def get_frame():
        if video:
            return cap.read()
        else:
            in_Frame= qOut_Frame.get()
            frame = in_Frame.getCvFrame()
            return True, frame

    startTime = time.monotonic()
    counter = 0
    detections = []
    frame = None
    rois = None
    tracks = None
    activity_history = None
    long_records = None
    long_avg = None
    wake_alert = None
    send_alerts = False
    monitor_breathing = False

    # rois =[[  68  ,133 ,1094,  558]]
    # tracks = [deque()] * len(rois)
    # activity_history = [deque()] * len(rois)
    # track_names = ['debugging']

    # nn data (bounding box locations) are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
            return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    if type(args.encode) == str:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        filename = str(Path(args.encode) / 'output.avi')
        fps = 25
        fw = 1280 #int(cap.get(3))
        fh = 720 #int(cap.get(4))
        result = cv2.VideoWriter(filename, fourcc, fps, (fw,fh),True)

    overall_start=time.time()
    frame_counter = 0

    while should_run():
        # Get image frames from camera or video file
        read_correctly, frame = get_frame()
        if not read_correctly:
            break

        frame_counter += 1

        if video:
            framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
            # Prepare image frame from video for sending to device
            img = dai.ImgFrame()
            img.setData(to_planar(frame, (300, 300)))
            img.setTimestamp(monotonic())
            img.setWidth(300)
            img.setHeight(300)
            # Use input queue to send video frame to device
            qIn_Frame.send(img)
        else:
            in_Frame = qOut_Frame.tryGet()

            if in_Frame is not None:
                frame = in_Frame.getCvFrame()
                if not args.nofps:
                    cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                                    (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_DUPLEX, 0.4, color=(255, 255, 255))

        inDet = qDet.tryGet()
        if inDet is not None:
            detections = inDet.detections
            counter += 1

        # color = colors[t.id % len(colors)]
        # color = [i * 255 for i in color]
        color = [255, 0, 0]

        # if the frame is available, render detection data on frame and display.
        if frame is not None:
            font_scale = frame.shape[0] / 720 * 0.6
            draw_thickness = int(frame.shape[0] / 720.0 * 3)
            text_offset_x = int(frame.shape[0] / 720.0 *10)
            text_offset_y = int(frame.shape[0] / 720.0 *15)
            if rois is not None:
                for roi in rois:
                    cv2.rectangle(frame, (roi[0],roi[1]), (roi[0]+roi[2],roi[1]+roi[3]), (0,0,255), draw_thickness//2)
                for detection in detections:
                    label = labelMap[detection.label]
                    if label not in display_categories: continue
                    bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    x1 = bbox[0]
                    y1 = bbox[1]
                    x2 = bbox[2]
                    y2 = bbox[3]
                    center = [(x1+x2)//2, (y1+y2)//2]

                    for t, roi in enumerate(rois):   # loop through tracks
                        if center[0] >= roi[0] and center[0] <= roi[0]+roi[2]:
                            if center[1] >= roi[1] and center[1] <= roi[1]+roi[3]:

                                # record position time history for a track
                                track_length = int(1.2 * delta_time_long * maxfps)
                                if len(tracks[t]) == track_length: tracks[t].popleft()
                                if video:
                                    timenow = datetime.datetime.fromtimestamp(frame_counter/framespersecond)
                                else:
                                    timenow = datetime.datetime.now()
                                tracks[t].append((timenow, center[0], center[1]))   # current time, center x, center y

                                # calculate activity levels and record activity time history for a track
                                activity_length = int(10 * delta_time_long * maxfps)
                                activity_short = None
                                activity_medium = None
                                activity_long = None
                                track = np.array(tracks[t])
                                for i in range(0, len(track)):
                                    if i == 0:
                                        continue
                                    if abs(track[i-1][0] - timenow).total_seconds() >= delta_time_short >= abs(track[i][0] - timenow).total_seconds():
                                        if len(activity_history[t]) == activity_length: activity_history[t].popleft()
                                        dist = 0
                                        for j in range(i,len(track)):
                                            dist += np.linalg.norm(track[j-1, 1:] - track[j, 1:])
                                        activity_history[t].append([timenow, dist / delta_time_short])
                                        activity_short = activity_history[t][-1][-1]
                                        if long_records[t] > 3 and activity_short > 3 * long_avg[t]: wake_alert[t] = True
                                        break
                                    if abs(track[i-1][0] - timenow).total_seconds() >= delta_time_medium >= abs(track[i][0] - timenow).total_seconds():
                                        dist = 0
                                        for j in range(i,len(track)):
                                            dist += np.linalg.norm(track[j-1, 1:] - track[j, 1:])
                                        activity_medium = dist / delta_time_medium
                                    if abs(track[i-1][0] - timenow).total_seconds() >= delta_time_long >= abs(track[i][0] - timenow).total_seconds():
                                        dist = 0
                                        for j in range(i,len(track)):
                                            dist += np.linalg.norm(track[j-1, 1:] - track[j, 1:])
                                        activity_long = dist / delta_time_long
                                        long_records[t] += 1
                                        long_avg[t] = (long_avg[t]*(long_records[t]-1) + activity_long) / long_records[t]

                                # draw object overlay
                                if track_names[t] == 0:
                                    track_label = label + '-' + str(t)
                                else:
                                    track_label = track_names[t]
                                cv2.putText(frame, f"{track_label}", (bbox[0] + text_offset_x, bbox[1] + text_offset_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color)
                                length = int( min(x2-x1, y2-y1) / 8 )
                                cv2.line(frame, (x1,y1), (int(x1+length), y1), color, draw_thickness)
                                cv2.line(frame, (x1,y1), (x1, int(y1+length)), color, draw_thickness)
                                cv2.line(frame, (x2,y2), (int(x2-length), y2), color, draw_thickness)
                                cv2.line(frame, (x2,y2), (x2, int(y2-length)), color, draw_thickness)
                                cv2.line(frame, (x1,y2), (int(x1+length), y2), color, draw_thickness)
                                cv2.line(frame, (x1,y2), (x1, int(y2-length)), color, draw_thickness)
                                cv2.line(frame, (x2,y1), (int(x2-length), y1), color, draw_thickness)
                                cv2.line(frame, (x2,y1), (x2, int(y1+length)), color, draw_thickness)

                                activity_saturation = frame.shape[1] / delta_time_short / 2  # moving all the way across the frame in 2 seconds
                                # print(activity_saturation, activity_short)
                                w = float(x2 - x1)
                                h = float(y2 - y1)
                                if activity_short is not None:
                                    cv2.rectangle(frame,(int(x2-w/10.0), y2),(x2, y2 - int(min(h, activity_short / activity_saturation * h))), color, draw_thickness)
                                if activity_medium is not None:
                                    cv2.rectangle(frame,(int(x2-w/2-w/20), y2),(int(x2-w/2+w/20), y2 - int(min(h, activity_medium / activity_saturation * h))), color, draw_thickness)
                                if activity_long is not None:
                                    cv2.rectangle(frame,(x1, y2),(int(x1+w/10), y2- int(min(h, activity_long / activity_saturation * h))),color, draw_thickness)
            else:
                # cv2.putText(frame, "Press 's' to select regions of interest", (frame.shape[1]//3, frame.shape[0]-text_offset_y*2), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255,255,255))
                pass

            # cv2.putText(frame, "elapsed time (wall clock): {:.4f} seconds".format(time.time()-overall_start), (text_offset_x,text_offset_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255,255,255))
            if video: 
                cv2.putText(frame, "time: {:.4f} seconds".format(frame_counter / framespersecond), (text_offset_x,2*text_offset_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255,255,255))
            else:
                cv2.putText(frame, "time: {:.4f} seconds".format(time.time()-overall_start), (text_offset_x,text_offset_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255,255,255))
            
            # cv2.putText(frame, "Press 'r' to show activity report", (frame.shape[1]*2//3, text_offset_y*2), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255,255,255))
            
            fps = counter / (time.monotonic() - startTime)
            if not args.nofps:
                if video:
                    cv2.putText(frame, "NN fps: {:.2f} ({:.1f}X speed)".format(fps, fps/framespersecond),
                                (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_DUPLEX, font_scale, color=(255, 255, 255))
                else:
                    cv2.putText(frame, "NN fps: {:.2f}".format(fps),
                                (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_DUPLEX, font_scale, color=(255, 255, 255))
        
            # Monitor breathing
            if monitor_breathing:
                breath_frame_count += 1
                min_detect_rate = 3  # breaths/minute
                max_detect_rate = 60
                breath_history_length = 4000

                crop = frame[breath_roi[1]:breath_roi[1]+breath_roi[3], breath_roi[0]:breath_roi[0]+breath_roi[2]]
                crop = frame[breath_center[1]:breath_center[1]+1, breath_center[0]:breath_center[0]+1]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if video:
                    timenow = datetime.datetime.fromtimestamp(frame_counter/framespersecond)
                else:
                    timenow = datetime.datetime.now()
                respiration_history.appendleft((timenow, gray[0,0]))

                cv2.line(frame, (breath_center[0],breath_center[1]-int(frame.shape[0]/50)), (breath_center[0],breath_center[1]+int(frame.shape[0]/50)), (255,0,255), 2)
                cv2.line(frame, (breath_center[0]-int(frame.shape[0]/50),breath_center[1]), (breath_center[0]+int(frame.shape[0]/50),breath_center[1]), (255,0,255), 2)

                temp_signal = None
                for i in range(0, len(respiration_history)):
                    if (timenow - respiration_history[i][0]).total_seconds() >= 5:
                        temp_signal = np.array(respiration_history)[:i]
                        break

                if temp_signal is not None:
                    temp_signal = pd.DataFrame(temp_signal, columns=['time', 'intensity'])
                    temp_signal['smoothed'] = temp_signal['intensity'].rolling(10).mean()
                    x=temp_signal['time']
                    y=temp_signal['intensity']

                    freqs = np.linspace(min_detect_rate/60*2*3.14, max_detect_rate/60*2*3.14, 6000)#int(max_detect_rate/60*2*3.14)*5)
                    periodogram = lombscargle(x, y, freqs)

                    kmax = periodogram.argmax()
                    resp_rate = freqs[kmax] / (2 * 3.14159) * 60
                    cv2.putText(frame, "{:.1f}".format(resp_rate), (breath_center[0]+frame.shape[0]//50,breath_center[1]-frame.shape[0]//50), cv2.FONT_HERSHEY_DUPLEX, font_scale, color=(255,0,255))

                    if args.debug:
                        # verify that the signal is similar to what is desired
                        print(temp_signal.head())
                        plt.figure(1)
                        sns.lineplot(x='time', y='intensity', data=temp_signal, ci=None)
                        sns.lineplot(x='time', y='smoothed', data=temp_signal, ci=None)

                        # first, try Lomb-Scargle periodogram, https://stackoverflow.com/questions/34428886/discrete-fourier-transformation-from-a-list-of-x-y-points/34432195#34432195
                        print("{:.3f} rad/s, {:.3f} breaths/min".format(freqs[kmax], freqs[kmax]/(2*3.14)*60))

                        plt.figure(2)
                        plt.plot(freqs, np.sqrt(4*periodogram/(400)))
                        plt.xlabel('Frequency (rad/s)')
                        plt.grid()
                        plt.axvline(freqs[kmax], color='r', alpha=0.25)
                        plt.show()
                        plt.close()

            if type(args.encode) == str:
                output = cv2.resize(frame,(1280, 720))
                result.write(output)
            
            cv2.imshow("video", frame)


        # Issue alerts
        if send_alerts:
            for i, alert in enumerate(wake_alert):
                if alert:
                    sg.popup(str(track_names[i]) + ' is awake! Sending message to Dr. Cristobal...')
                    wake_alert[i] = False
                    send_alerts = False


        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            rois = cv2.selectROIs("video", frame, fromCenter=False, showCrosshair=True)
            print('Regions of Interest identified:\n', rois)
            tracks = [deque() for roi in rois]
            activity_history = [deque() for roi in rois]
            long_records = [0]*len(rois)
            long_avg = [0]*len(rois)
            wake_alert = [False]*len(rois)
            track_names = [0]*len(rois)
            for p in range(0,len(rois)):
                event, values = sg.Window('Window Title', [[sg.Text('Please name patient-' + str(p))],[sg.Input()],[sg.Button('Ok')]]).read(close=True)
                track_names[p] = values[0]

        if key == ord('r'):
            print('opening report')
            data = []
            for t, activity_track in enumerate(activity_history):
                if len(activity_track) > 2:
                    for i in activity_track:
                        temp = [x for x in i]
                        temp.append(track_names[t])
                        data.append(temp)
                    df = pd.DataFrame(data, columns=['time', 'activity level', 'patient'])
                    # df['smoothed'] = df['activity level'].rolling(10).max()
            sns.lineplot(x='time', y='activity level', data=df, ci=None, hue='patient').get_figure().savefig('activity_history.png')
            plt.close()

            cv2.namedWindow('Activity History',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Activity History', 500, 500)
            cv2.moveWindow('Activity History', 1300, 300)

            report = cv2.imread('activity_history.png')
            cv2.imshow('Activity History',report)
        
        if key == ord('b'):
            breath_roi = list(cv2.selectROI("video", frame, fromCenter=True, showCrosshair=True))
            print('Respiration Region of Interest identified:\n', breath_roi)
            # num_pixels = breath_roi[2] * breath_roi[3]
            # max_query_width = 5
            # if num_pixels < max_query_width**2:
            #     respiration_histories = [deque()]*num_pixels
            # else:
            breath_center = [breath_roi[0]+breath_roi[2]//2, breath_roi[1]+breath_roi[3]//2]
                # breath_roi[0] = breath_center[0]-max_query_width//2
                # breath_roi[1] = breath_center[1]-max_query_width//2
                # breath_roi[2] = max_query_width
                # breath_roi[3] = max_query_width
                # respiration_histories = [deque()]*max_query_width**2
            respiration_history = deque()
            if monitor_breathing:
                monitor_breathing = False
                print("Respiration rate will NOT be monitored")
            else:
                monitor_breathing = True
                print("Respiration rate will be monitored")

        if key == ord('a'):
            if send_alerts:
                send_alerts = False
                print("Wake alerts will NOT be sent")
            else:
                send_alerts = True
                print("Wake alerts will be sent")

        elif key == ord('q') or key == ord('x'):
            break
