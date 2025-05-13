import cv2
import numpy as np

def track_point(video_path, start_point, color_bgr, output_path="output.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    #lucas-kanade method is chosen as the gradient around the end of the barbell is likely to be different from its surroundings
    #changes in gradient are sufficient in both x and y directions for the lk method to be effective
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY) #first frame
    pos0 = np.array([[start_point]], dtype=np.float32) #last known position of the point

    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #curr frame

        #pos1 is the estimated new positon of the point
        pos1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, pos0, None, **lk_params)

        if st[0][0] == 1:
            x1, y1 = pos0[0][0]
            x2, y2 = pos1[0][0]
            mask = cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, 3)
            frame = cv2.circle(frame, (int(x2), int(y2)), 5, (0, 255, 0), -1)
            pos0 = pos1
        output = cv2.add(frame, mask) #to combine all frames and masks together
        out.write(output)

        old_gray = frame_gray.copy()

    cap.release()
    out.release()
    return output_path
