from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np
import os
from tqdm import tqdm

detection_graph, sess = detector_utils.load_inference_graph()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.5,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()
    return args
    

def isHand(frame):
	blur = cv2.blur(frame,(3,3))

	hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
	mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
 
	kernel_square = np.ones((11,11),np.uint8)
	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

	dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
	erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
	dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
	filtered = cv2.medianBlur(dilation2,5)
	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
	dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
	kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	median = cv2.medianBlur(dilation2,5)
	ret,thresh = cv2.threshold(median,127,255,0)

	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
	
	if len(contours) == 0:
		return False

	areas = [cv2.contourArea(x) for x in contours]

	if areas[0] < 4000:
		return False
	return True

def video(score_thresh, filepath, savepath):
    """
        filepath: original file path
        savepath: should be in .mp4 format
    """
    cap = cv2.VideoCapture(filepath)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    #start_time = datetime.datetime.now()
    #num_frames = 0
    # IMPORTANT! W, H MUST BE IN INTEGER TYPE
    im_width, im_height = (int(cap.get(3)), int(cap.get(4)))
    im_size = (im_width, im_height)
    print(im_height, im_width)
    # max number of hands we want to detect/track
    num_hands_detect = 1
    fps = 20

    #cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(savepath, fourcc, fps, im_size)
    
    while (cap.isOpened()):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, frame = cap.read()
        # image_np = cv2.flip(image_np, 1)
        if ret: 
            try:
                image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

            boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)

            # draw bounding boxes on frame
            detector_utils.draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np)

            write_frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            out.write(write_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("EOF")
            break
    
    cap.release()
    out.release()


def image(score_thresh, IMAGE_PATH, TXT_PATH):
    """
        filepath: original file path
        savepath: should be in .mp4 format
    """
    no_object = open("no_object.txt", "a+")
    for filename in tqdm(os.listdir(IMAGE_PATH)):
        filepath = os.path.join(IMAGE_PATH, filename)
        txtname = filename.replace("jpeg", "txt")
        txtpath = os.path.join(TXT_PATH, txtname)

        image = cv2.imread(filepath)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_height, im_width, c = image_np.shape
        im_size = (im_width, im_height)
        #print(im_height, im_width)
        # max number of hands we want to detect/track
        num_hands_detect = 2
        
        # inference
        # scores에는 유력한 bbox들의 confidence가 내림차순으로 저장되어 있음
        boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
        
        # Calculate coordinates
        hand_coordinates = []
        for i in range(num_hands_detect):
            if (scores[i] > score_thresh):
                #normalized coordinates
                (left, right, top, bottom) = (boxes[i][1] , boxes[i][3] ,
                                            boxes[i][0] , boxes[i][2] )
                #print(left, right, top, bottom)
                
                hand_coordinates.append((left, right, top, bottom))
        
        # merge two boxes
        if len(hand_coordinates)==2:
            left = min(hand_coordinates[0][0], hand_coordinates[1][0])
            right = max(hand_coordinates[0][1], hand_coordinates[1][1])
            top = min(hand_coordinates[0][2], hand_coordinates[1][2])
            bottom = max(hand_coordinates[0][3], hand_coordinates[1][3])
            hand_coordinates[0] = (left, right, top, bottom)


        if len(hand_coordinates)==0:
            # no hand detected -> Delete file
            f = open(txtpath, 'r')
            label = str(f.readline().rstrip())
            no_object.write(f"{filename}\n")
            pass
        else:
            # Calculate Normalized cx, cy, w, h
            final_bbox = hand_coordinates[0]
            left, right, top, bottom = final_bbox
            cx = (left + right) / 2
            cy = (top + bottom) / 2
            width = right - left
            height = bottom - top
            #print(cx, cy, width, height)
            # Append in .txt file
            """ 
            FIXME
            Not working properly
            """
            f = open(txtpath, 'w+')
            label = str(f.readline()).split("\t")[0]
            f.write(f"{label}\t{cx}\t{cy}\t{width}\t{height}")
            f.close()



if __name__ == '__main__':
    args = parse_args()
    IMAGE_PATH = "/opt/ml/input/full_image/train"
    TXT_PATH = "/opt/ml/input/full_label/train"
    #IMAGE_PATH = "/opt/ml/example/image"
    #TXT_PATH = "/opt/ml/example/label"
    #print(os.listdir(IMAGE_PATH))

    image(args.score_thresh, IMAGE_PATH, TXT_PATH)