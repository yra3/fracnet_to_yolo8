import os
import numpy as np
import pydicom
import nrrd
import cv2
from pathlib import Path
import random

import nibabel as nib

def get_data_folders():
    data_home = Path('~/learning/tiu/doctors/data_raw')
    train_imgs = data_home / 'ribfrac-val-images'
    train_labels = data_home / 'ribfrac-val-labels'
    return train_imgs, train_labels

def load_nifti(nifti_path):
    """
    Загрузка NIfTI файла (.nii.gz)
    """
    nifti_img = nib.load(nifti_path)
    nifti_data = nifti_img.get_fdata()

    # NIfTI данные обычно имеют формат [x, y, z], преобразуем в [z, y, x]
    # для совместимости с остальным кодом
    if len(nifti_data.shape) == 3:
        nifti_data = np.transpose(nifti_data, (2, 1, 0))

    return nifti_data, nifti_img

def load_ct_and_annotation(image_path, annotation_path):
    # Загрузка КТ изображения
    ct_img = nib.load(image_path)
    ct_data = ct_img.get_fdata()
    
    # Загрузка аннотации (маски)
    annotation_img = nib.load(annotation_path)
    annotation_data = annotation_img.get_fdata()
    
    return ct_data, annotation_data, ct_img.header

def get_single_case_path(case_name):
    train_imgs, train_labels = get_data_folders()
    case_image_name = case_name + '-image.nii.gz'
    case_label_name = case_name + '-label.nii.gz'
    train_img = train_imgs / case_image_name
    train_label = train_labels / case_label_name
    return train_img, train_label

def get_single_case_data(case_name):
    train_img, train_label = get_single_case_path(case_name)
    # Загрузка КТ изображения
    return load_ct_and_annotation(train_img, train_label)

# RibFrac426,0,0
# RibFrac426,1,-1
# RibFrac426,2,3
# RibFrac426,3,3


get_single_case_data('RibFrac426')
