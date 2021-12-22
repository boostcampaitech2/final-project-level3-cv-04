from genericpath import exists
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
        default=0.2,
        help='Score threshold for displaying bounding boxes')
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
    parser.add_argument(
        '--imgdir',
        type=str,
        help='Dataset directory')
    parser.add_argument(
        '--savedir',
        type=str,
        help='Directory path to save rendered image')
    args = parser.parse_args()
    return args

def render_bbox(score_thresh, IMAGE_PATH, SAVE_IMG_PATH, SAVE_TXT_PATH):
    # Save no_object.txt per one directory(class)
    no_object = open(os.path.join(SAVE_IMG_PATH, "no_object.txt"), "w+")
    for filename in tqdm(os.listdir(IMAGE_PATH)):
        filepath = os.path.join(IMAGE_PATH, filename)

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
            no_object.write(f"{filename}\n")
            pass
        else:
            # Render bbox on image and save
            p1 = (int(left * im_width), int(top * im_height))
            p2 = (int(right * im_width), int(bottom * im_height))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(SAVE_IMG_PATH, filename), image_np)
            # Save .txt file
            final_bbox = hand_coordinates[0]
            left, right, top, bottom = final_bbox
            cx = (left + right) / 2
            cy = (top + bottom) / 2
            width = right - left
            height = bottom - top
            label = IMAGE_PATH.split("/")[-2]
            f = open(os.path.join(SAVE_TXT_PATH, filename.replace("jpg", "txt")), "x")
            f.write(f"{label}\t{cx}\t{cy}\t{width}\t{height}")
            f.close()
    no_object.close()

if __name__ == '__main__':
    args = parse_args()

    IMAGE_ROOT = args.imgdir
    SAVE_ROOT = args.savedir
    for i in range(0, 6):
        impath = os.path.join(IMAGE_ROOT, str(i))
        save_impath = os.path.join(SAVE_ROOT, f"{i}/image")
        save_txtpath = os.path.join(SAVE_ROOT, f"{i}/label")
        os.makedirs(save_impath, exist_ok=True)
        os.makedirs(save_txtpath, exist_ok=True)
        render_bbox(args.score_thresh, impath, save_impath, save_txtpath)