from ctypes import *
import math, random, os, cv2, time	  # For Math Calculations, Random Numbers, Read/Write Files(System), OpenCV(Vision Algorithms) and Time Modules(Calculating FPS) Respectively
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import numpy as np			  # Array and Matrix Operations for OpenCV
from mpl_toolkits.mplot3d import Axes3D	  # For Graph Plots
# import voice				  # Optional Module for Voice
# import sys, rospy, rospy_tutorials, message_filters

# DARKNET PYTHON IMPLEMENTATION	- START
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))] 

lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def detect(net, meta, image, thresh=.4, hier_thresh=.5, nms=.6):
    im, image = array_to_image(image)
    rgbgr_image(im)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], 
                           (b.x, b.y, b.w, b.h)))

    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes): free_image(im)
    free_detections(dets, num)
    print(res)
    return res

# DARKNET PYTHON IMPLEMENTATION	- END

# CONFIG, WEIGHTS AND META FILE PATH; CALIBRATION FILE VALUES
net = load_net("./yolov3-tiny.cfg", "./yolov3-tiny.weights", 0)
meta = load_meta("./coco.data")
# Q-Matrix: [ 1., 0., 0., -3.8261030960083008e+002, 0., 1., 0., -2.2310979270935059e+002, 0., 0., 0., 7.4444708366093369e+002, 0., 0., 1.6655093206481758e-002, 0. ]	Read From Calibration File
Q_Mat = [744.447, 382.610, 744.447, 223.1097]	# fx, cx, fy, cy of Q_Matrix from Calibration File - Refer Previous Q - Matrix Values

# OBJECT DETECTION MODULE : 
def object_detect(image, disp_map):
	point = [0, 0, 0] 
	time_now = time.time()
	identified_objects = detect(net, meta, image)
	fps = 0
	for i in identified_objects:  
		x = int(i[2][0])	# X, Y is the Co-ordinates of Center of The Bounding Box  
		y = int(i[2][1])
		w = int(i[2][2])	# W and H are the Dimensions of the Box
		h = int(i[2][3])
		Z = disp_map[y,x]
		X = (Z*(x - Q_Mat[1])) / (Q_Mat[0])	# Converting Pixel Co-ordinates to RealWorld 3D Co-ordinates
		Y = (Z*(y - Q_Mat[3])) / (Q_Mat[2])

		display = "(" + str(int(X)) + "," + str(int(Y)) + "," + str(int(Z)) + ") cm"
		time_t = str(1/(time.time() - time_now))	 
		cv2.putText(image, i[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
		cv2.putText(image, display, (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
		cv2.rectangle(image, (x-w/2, y-h/2), (x+w/2, y+h/2), (0, 255, 0), 3)
		print(display)

	fps = int(1/(time.time() - time_now))
	cv2.putText(image, "FPS: " + str(fps), (600, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)
	cv2.imshow("IMAGE", image)
	cv2.waitKey(1)		
	return fps

# Main Function
if __name__ == "__main__":
	# Read Frame from Real Sense
	pipe = rs.pipeline()
	pipeline.start()

	try:
	    while True:
		# Create a pipeline object. This object configures the streaming camera and owns it's handle
		color_frame = pipeline.wait_for_frames()
		depth_frame = frames.get_depth_frame()
		if not depth: continue

		fps = object_detect(color_frame, depth_frame)
		f = open(".\output.csv", "a")
		for i in r:  
			x = int(i[2][0]) 
			y = int(i[2][1])
			w = int(i[2][2])
			h = int(i[2][3])
			cv2.rectangle(im, (x-w/2, y-h/2), (x+w/2, y+h/2), (0,255,0), 3)	
		print("Streaming FPS :", fps)	
		
	finally:
	    pipeline.stop()

'''
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

'''
