from detect_face import detect_face, create_mtcnn

import os
import cv2

import tensorflow as tf
import time

import threading

def initiate_graph():
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        with sess.as_default():
            return create_mtcnn(sess, None)

def img_resizing(img, size=300):
    row, column, ch = img.shape
    new_pct = row/size
    new_row, new_column = (row/new_pct, column/new_pct)
    return cv2.resize(img, (int(new_column), int(new_row))), new_pct


def face_det(coordinates, img, pnet, rnet, onet, resize=True):
    # coordinates = []
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    if resize:
        resized_img, pct = img_resizing(img, 224)

        start = time.time()
        bbx, _ = detect_face(resized_img, minsize, pnet, rnet, onet, threshold, factor)
        end = time.time()
        # print(bbx)
        print('Took with resize: ', end - start)
        for i in range(len(bbx)):
            coordinates.append([])
            for j in range(5):
                coordinates[i].append([0, 0, 0, 0, 0])
                coordinates[i][j] = bbx[i][j]*pct
            coordinates[i][-1] = bbx[i][-1]
        # print(coordinates)
        return
    else:
        start = time.time()
        bbx, _ = detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        end = time.time()
        # print(bbx)
        print('Took no_resize: ', end - start)
        for i in range(len(bbx)):
            coordinates.append([])
            for j in range(5):
                coordinates[i].append([0, 0, 0, 0, 0])
                coordinates[i][j] = bbx[i][j]
            coordinates[i][-1] = bbx[i][-1]
        # print(coordinates)
        return

def draw_faces(img, bounding_boxes):
    for face in bounding_boxes:
        try:
            if face[-1] > 0.9:
                # print(face)
                cv2.rectangle(img, (int(face[0]), int(face[1])), (int(face[2]), int(face[3])), (0, 255, 0), 1)
        except Exception as e:
            print(e)
            continue


def video_detection(path, pnet, rnet, onet):
    cap = cv2.VideoCapture(path)
    nf = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
    tmp = cap.get(cv2.CAP_PROP_POS_MSEC)
    fps = round(nf/(tmp/1000))
    print(nf, tmp, fps)

    cap = cv2.VideoCapture(path)
    # first frame
    ret, frame = cap.read()
    if ret:
        bbx = []
        boxes = bbx.copy()
        cvt_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        thread = threading.Thread(target=face_det, args=(bbx, cvt_img, pnet, rnet, onet))
        thread.start()

    while (ret):
        # Capture frame-by-frame
        if not thread.is_alive():
            # print('finished: ', bbx)
            cvt_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = bbx.copy()
            bbx = []
            thread = threading.Thread(target=face_det, args=(bbx, cvt_img, pnet, rnet, onet))
            thread.start()
        # print(boxes)

        ret, frame = cap.read()

        # Display the resulting frame
        draw_faces(frame, boxes)
        cv2.imshow('frame', frame)

        if cv2.waitKey(fps) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    pnet, rnet, onet = initiate_graph()

    video_detection(path='/home/lucas/Vídeos/test.mp4', pnet=pnet, rnet=rnet, onet=onet)

    # img = cv2.imread('/home/lucas/Imagens/people/alot/people-to-people.jpg', cv2.IMREAD_COLOR)
    # img = cv2.imread('/home/lucas/Imagens/people/c/pe.jpg', cv2.IMREAD_COLOR)

    # cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # bbx = []
    # print(bbx)
    # start = time.time()
    # face_det(bbx, cvt_img, pnet, rnet, onet, resize=True)
    # thread = threading.Thread(target=face_det, args=(bbx, cvt_img, pnet, rnet, onet))
    # while thread.isAlive():
    #     if not thread.isAlive():
    #         break
    # end = time.time()
    # print('took: ', end - start)
    # print(bbx)

    # start = time.time()
    # face_det(bbx, cvt_img, pnet, rnet, onet, resize=False)
    # end = time.time()
    # print('took: ', end - start)
    # print(bbx)
    #
    # # print(bbox)
    # draw_faces(img, bbx)
    #
    # cv2.imshow('teste', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
