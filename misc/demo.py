# -*- coding: utf-8 -*-
"""
This program provides UI to input keypoints of profile image
 and generate frontal image using TP-GAN

Add Keras_TP-GAN directory to PYTHONPATH

"""
import os.path as osp
import sys

# sys.path.append(osp.join('/', *(__file__.split('/')[:-2])))
sys.path.append('/home/shuai.li/code/Keras_TP-GAN')

import numpy as np
import pickle
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import os
from keras_tpgan.tpgan import TPGAN
import gen_keypoint_facepatch as gen
from tqdm import tqdm

def main(dir_path, img_file, img_size, save_path):

    if not osp.exists(save_path):
        os.makedirs(save_path)

    repo_path = '/home/shuai.li/code/Keras_TP-GAN/'
    GENERATOR_WEIGHTS_FILE = osp.join(repo_path, 'weights_file-20200702T103254Z-001/generator/epoch0480_loss0.560.hdf5')
    lcnn_extractor_weights = osp.join(repo_path, 'weights_file-20200702T103254Z-001/lcnn_fine_tuned/extract29v2_lr0.00010_loss0.997_valacc1.000_epoch1110.hdf5')
    classifier_weights = osp.join(repo_path, 'weights_file-20200702T103254Z-001/classifier/epoch0480_loss0.560.hdf5')
    discriminator_weights = osp.join(repo_path, 'weights_file-20200702T103254Z-001/discriminator/epoch0480_loss0.222.hdf5')

    tpgan = TPGAN(lcnn_extractor_weights=lcnn_extractor_weights,
                  generator_weights=GENERATOR_WEIGHTS_FILE,
                  classifier_weights=classifier_weights,
                  discriminator_weights=discriminator_weights)

    for crop_img, leye_img, reye_img, nose_img, mouth_img, img_name in tqdm(gen.get_imgpatch(dir_path, img_file, img_size)):

        global pred_faces, pred_faces64, pred_faces32, pred_leyes, pred_reyes, pred_noses, pred_mouthes
        x_z = np.random.normal(scale=0.02, size=(1, 100))
        x_face = (cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB).astype(np.float) / 255)[np.newaxis, :]
        x_leye = (cv2.cvtColor(leye_img, cv2.COLOR_BGR2RGB).astype(np.float) / 255)[np.newaxis, :]
        x_reye = (cv2.cvtColor(reye_img, cv2.COLOR_BGR2RGB).astype(np.float) / 255)[np.newaxis, :]
        x_nose = (cv2.cvtColor(nose_img, cv2.COLOR_BGR2RGB).astype(np.float) / 255)[np.newaxis, :]
        x_mouth = (cv2.cvtColor(mouth_img, cv2.COLOR_BGR2RGB).astype(np.float) / 255)[np.newaxis, :]

        [pred_faces, pred_faces64, pred_faces32, pred_leyes, pred_reyes, pred_noses, pred_mouthes] \
            = tpgan.generate([x_face, x_leye, x_reye, x_nose, x_mouth, x_z])

        cv2.imwrite(osp.join(save_path, "{}_rotated.jpg".format(img_name)), pred_faces[0][:,:,::-1])


if __name__ == "__main__":
    dir_path = '/home/shuai.li/code/Keras_TP-GAN/test/img_paper_temp/images'
    save_path = '/home/shuai.li/code/Keras_TP-GAN/test/img_paper_temp/images_rotate/'
    img_file = '/home/shuai.li/code/Keras_TP-GAN/test/img_paper_temp/file.txt'
    img_size = 128
    main(dir_path, img_file, img_size, save_path)




