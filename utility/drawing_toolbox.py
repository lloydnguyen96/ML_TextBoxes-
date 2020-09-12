import cv2
import numpy as np
import random

TEXT_COLOR=(255, 255, 255)
FONT=cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE=0.4
TEXT_THICKNESS=2
LINE_TYPE=8

def get_color():
    return int(random.random() * 256),\
           int(random.random() * 256),\
           int(random.random() * 256)

def draw_bboxes_on_image(image,
                         classes,
                         scores,
                         bboxes,
                         thickness=2):
    shape=image.shape
    num_bboxes=bboxes.shape[0]
    for i in range(num_bboxes):
        if classes[i] < 1: continue
        bbox=bboxes[i]
        color=get_color()

        # Draw bounding boxes
        ymin=int(round(bbox[0]))
        xmin=int(round(bbox[1]))
        ymax=int(round(bbox[2]))
        xmax=int(round(bbox[3]))
        if (xmax - xmin < 1) or (ymax - ymin < 1):
            continue

        # Draw bouding boxes.
        cv2.rectangle(image,
                      (xmin, ymin),
                      (xmax, ymax),
                      color,
                      thickness)

        # Draw text.
        text='%.1f%%' % (scores[i]*100)
        text_size, baseline=\
            cv2.getTextSize(text,
                            fontFace=FONT,
                            fontScale=FONT_SCALE,
                            thickness=TEXT_THICKNESS)

        cv2.rectangle(image,
                      (xmin - thickness//2,
                       ymin - text_size[1] - baseline - thickness),
                      (xmin + text_size[0] + thickness//2,
                       ymin - thickness//2),
                      color=color,
                      thickness=-1)

        cv2.putText(image,
                    text,
                    (xmin, ymin - baseline - thickness//2),
                    fontFace=FONT,
                    fontScale=FONT_SCALE,
                    color=TEXT_COLOR,
                    thickness=TEXT_THICKNESS,
                    lineType=LINE_TYPE)
    return image

def draw_quadrilaterals_on_image(image,
                                 classes,
                                 scores,
                                 quadrilaterals,
                                 thickness=2):
    num_quadrilaterals=quadrilaterals.shape[0]
    for i in range(num_quadrilaterals):
        if classes[i] < 1:
            continue
        quadrilateral=quadrilaterals[i]
        color=get_color()
        y1, x1, y2, x2,\
        y3, x3, y4, x4=quadrilateral
        if x2 - x1 < 1 or y3 - y2 < 1 or x3 - x4 < 1 or y4 - y1 < 1:
            continue

        # Draw quadrilateral.
        polygon=np.array([[x1, y1],
                          [x2, y2],
                          [x3, y3],
                          [x4, y4]], np.int32)
        polygon=polygon.reshape((-1, 1, 2))
        cv2.polylines(image,
                      [polygon],
                      True,
                      color,
                      thickness=thickness)

        # Draw text indicating the confidence score.
        text='%.1f%%' % (scores[i]*100)
        text_size, baseline=\
            cv2.getTextSize(text,
                            fontFace=FONT,
                            fontScale=FONT_SCALE,
                            thickness=TEXT_THICKNESS)
        cv2.rectangle(image,
                      (int(x1 - thickness//2),
                       int(y1 - text_size[1] - baseline - thickness)),
                      (int(x1 + text_size[0] + thickness//2),
                       int(y1 - thickness//2)),
                      color=color,
                      thickness=-1)
        cv2.putText(image,
                    text,
                    (x1, int(y1 - thickness//2 - baseline)),
                    fontFace=FONT,
                    fontScale=FONT_SCALE,
                    color=TEXT_COLOR,
                    thickness=TEXT_THICKNESS,
                    lineType=LINE_TYPE)
    return image