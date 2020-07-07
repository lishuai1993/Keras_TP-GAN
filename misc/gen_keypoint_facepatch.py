import cv2
import glob
import numpy as np
import os.path as osp
import json
import copy
import os


EYE_H = 40
EYE_W = 40
NOSE_H = 32
NOSE_W = 40
MOUTH_H = 32
MOUTH_W = 48

# nose = 30
eye_y = 40
mouth_y = 88

i_right_eye = [37, 38, 40, 41]
i_left_eye = [43, 44, 46, 47]
i_mouth = [48, 54]




def mouse_key_point_event(event, x, y, flags, param):  # 需要全局变量进行传参么

    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(show_img, (x, y), 3, (0, 255, 255), -1)
        key_points.append([x, y])

        if len(key_points) == 1:
            print('input reye')
        elif len(key_points) == 2:
            print('input nose_tip')
        elif len(key_points) == 3:
            print('input mouth')


def get_fourkeypoint(dir_path, keypointfile, suffix):

    file_list = glob.glob(r"{}\*{}".format(dir_path, suffix))

    global key_points
    global show_img

    cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("img", mouse_key_point_event)  # 设置了监测鼠标活动情况的进程(线程)

    all_keypoint = dict()
    key_points = []
    for file_path in file_list:
        img = cv2.imread(file_path)

        show_img = img
        while len(key_points) < 4:                  # 防止for循环一直前行
            cv2.imshow("img", show_img)
            key = cv2.waitKeyEx(10)
            if key == ord('q'):
                break
        if len(key_points) >= 4:
            all_keypoint['{}'.format(osp.basename(file_path))] = copy.deepcopy(key_points)
            key_points.clear()
    cv2.destroyAllWindows()
    with open(keypointfile, 'w') as f:
        json.dump(all_keypoint, f)


def get_imgpatch(dir_path, img_file, img_size):

    img_list = glob.glob("{}/*".format(dir_path))
    with open(img_file, 'r') as fl:
        imgkp = json.load(fl)

    for img_path in img_list:
        img_name = osp.basename(img_path)
        img = cv2.imread(img_path)

        img_h, img_w = img.shape[:2]
        eye_y_ratio = eye_y / img_size
        mouth_y_ratio = mouth_y / img_size
        

        print(img_name)
        key_points = imgkp.get(img_name)
        if key_points == None:
            continue

        reye, leye, nose_tip, mouth = key_points

        reye = np.array(reye)
        leye = np.array(leye)
        nose_tip = np.array(nose_tip)
        mouth = np.array(mouth)

        vec_mouth2reye = reye - mouth
        vec_mouth2leye = leye - mouth

        # 确定旋转角度，并执行图像和关键点旋转    angle reye2mouth against leye2mouth
        phi = np.arccos(vec_mouth2reye.dot(vec_mouth2leye) / (
                np.linalg.norm(vec_mouth2reye) * np.linalg.norm(vec_mouth2leye))) / np.pi * 180
        if phi < 15:                                # consider the profile image is close to 90 deg.
            eye_center = (reye + leye) / 2          # in case of 90 deg. set invisible eye with copy of visible eye.
            if nose_tip[0] > eye_center[0]:
                leye = reye
            else:
                reye = leye

        # calc angle eyes against horizontal as theta
        if np.array_equal(reye, leye) or phi < 38:      # in case of 90 deg. avoid rotation
            theta = 0
        else:
            vec_leye2reye = reye - leye
            if vec_leye2reye[0] < 0:
                vec_leye2reye = -vec_leye2reye
            theta = np.arctan(vec_leye2reye[1] / vec_leye2reye[0]) / np.pi * 180

        imgcenter = (img.shape[1] / 2, img.shape[0] / 2)                # 进行人脸旋转
        rotmat = cv2.getRotationMatrix2D(imgcenter, theta, 1)
        rot_img = cv2.warpAffine(img, rotmat, (img.shape[1], img.shape[0]))

        reye_ext = np.insert(reye, 2, 1, axis=0)
        leye_ext = np.insert(leye, 2, 1, axis=0)
        nose_tip_ext = np.insert(nose_tip, 2, 1, axis=0)
        mouth_ext = np.insert(mouth, 2, 1, axis=0)
        
        tot_reye     = np.dot(rotmat, reye_ext)
        tot_leye     = np.dot(rotmat, leye_ext)
        tot_nose_tip = np.dot(rotmat, nose_tip_ext)
        tot_mouth    = np.dot(rotmat, mouth_ext)

        # 确定裁剪范围
        crop_size = int((mouth[1] - reye[1]) / (mouth_y_ratio - eye_y_ratio))
        crop_up = int(reye[1] - crop_size * eye_y_ratio)
        if crop_up < 0:
            crop_up = 0

        crop_down = crop_up + crop_size
        if crop_down > rot_img.shape[0]:
            crop_down = rot_img.shape[0]

        crop_left = int((reye[0] + leye[0]) / 2 - crop_size / 2)
        if crop_left < 0:
            crop_left = 0

        crop_right = crop_left + crop_size
        if crop_right > rot_img.shape[1]:
            crop_right = rot_img.shape[1]

        # 对图像和关键点进行裁剪并缩放
        crop_img = rot_img[crop_up:crop_down, crop_left:crop_right]
        crop_img = cv2.resize(crop_img, (img_size, img_size))

        tot_reye[0] = int((tot_reye[0] - crop_left) / (crop_right - crop_left) * img_size)
        tot_leye[0] = int((tot_leye[0] - crop_left) / (crop_right - crop_left) * img_size)
        tot_nose_tip[0] = int((tot_nose_tip[0] - crop_left) / (crop_right - crop_left) * img_size)
        tot_mouth[0] = int((tot_mouth[0] - crop_left) / (crop_right - crop_left) * img_size)

        tot_reye[1] = int((tot_reye[1] - crop_up) / (crop_down - crop_up) * img_size)
        tot_leye[1] = int((tot_leye[1] - crop_up) / (crop_down - crop_up) * img_size)
        tot_nose_tip[1] = int((tot_nose_tip[1] - crop_up) / (crop_down - crop_up) * img_size)
        tot_mouth[1] = int((tot_mouth[1] - crop_up) / (crop_down - crop_up) * img_size)

        key_rects = []
        key_rects.append([(int(tot_leye[0] - EYE_W / 2), int(tot_leye[1] - EYE_H / 2)),
                          (int(tot_leye[0] + EYE_W / 2), int(tot_leye[1] + EYE_H / 2))])
        key_rects.append([(int(tot_reye[0] - EYE_W / 2), int(tot_reye[1] - EYE_H / 2)),
                          (int(tot_reye[0] + EYE_W / 2), int(tot_reye[1] + EYE_H / 2))])
        key_rects.append([(int(tot_nose_tip[0] - NOSE_W / 2), int(tot_nose_tip[1] - NOSE_H / 2)),
                          (int(tot_nose_tip[0] + NOSE_W / 2), int(tot_nose_tip[1] + NOSE_H / 2))])
        key_rects.append([(int(tot_mouth[0] - MOUTH_W / 2), int(tot_mouth[1] - MOUTH_H / 2)),
                          (int(tot_mouth[0] + MOUTH_W / 2), int(tot_mouth[1] + MOUTH_H / 2))])

        leye_img = crop_img[key_rects[0][0][1]:key_rects[0][1][1], key_rects[0][0][0]:key_rects[0][1][0]]
        reye_img = crop_img[key_rects[1][0][1]:key_rects[1][1][1], key_rects[1][0][0]:key_rects[1][1][0]]
        nose_img = crop_img[key_rects[2][0][1]:key_rects[2][1][1], key_rects[2][0][0]:key_rects[2][1][0]]
        mouth_img = crop_img[key_rects[3][0][1]:key_rects[3][1][1], key_rects[3][0][0]:key_rects[3][1][0]]

        crop_path = osp.join(osp.dirname(dir_path), 'image_crop')
        if not osp.exists(crop_path):
            os.makedirs(crop_path)
        cv2.imwrite(osp.join(crop_path, img_name), crop_img)

        yield crop_img, leye_img, reye_img, nose_img, mouth_img, img_name[:-4]


if __name__ == "__main__":

    dir_path = r'D:\work\code\Keras_TP-GAN\test\img_camera\images'
    file_towrite = r'D:\work\code\Keras_TP-GAN\test\img_camera\file.txt'
    suffix = '.jpg'
    get_fourkeypoint(dir_path, file_towrite, suffix)











