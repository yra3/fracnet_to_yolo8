import os
import numpy as np
import pydicom
import nrrd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import random
import nibabel as nib
import pandas as pd

def get_data_folders():
    data_home = Path('/home/yura/learning/tiu/doctors/data_raw')
    train_imgs = data_home / 'ribfrac-val-images'
    train_labels = data_home / 'ribfrac-val-labels'
    return train_imgs, train_labels

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


def create_segmentation_data(annotation_slice, info_df, patient_id):
    """
    Создает данные для сегментации (ограничивающие рамки и полигоны) для переломов на срезе
    """
    
    # Получаем информацию о метках для данного пациента
    patient_labels = info_df[info_df['public_id'] == patient_id]
    
    # Список для хранения данных сегментации
    segmentation_data = []
    
    # Находим уникальные метки на срезе (кроме фона)
    unique_labels = np.unique(annotation_slice)
    unique_labels = unique_labels[unique_labels > 0]
    
    # Размеры изображения для нормализации координат
    height, width = annotation_slice.shape
    
    for label_id in unique_labels:
        # Получаем тип перелома из DataFrame
        label_info = patient_labels[patient_labels['label_id'] == label_id]
        
        if not label_info.empty:
            label_code = label_info.iloc[0]['label_code']
            
            # Обрабатываем только определенные типы переломы (1-5)
            if 1 <= label_code <= 5:
                # Создаем бинарную маску для текущей метки
                mask = (annotation_slice == label_id).astype(np.uint8)
                
                # Находим контуры на маске
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Получаем ограничивающий прямоугольник
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Преобразуем в формат YOLO (класс, x_center, y_center, width, height)
                    # Все значения нормализованы от 0 до 1
                    x_center = (x + w/2) / width
                    y_center = (y + h/2) / height
                    bbox_width = w / width
                    bbox_height = h / height
                    
                    # Класс для YOLO (от 0 до 4, а не от 1 до 5)
                    yolo_class = label_code - 1
                    
                    # Создаем полигон для сегментации
                    # Упрощаем контур для уменьшения количества точек
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Преобразуем контур в формат для YOLO (нормализованные координаты)
                    polygon = []
                    for point in approx_contour:
                        x_norm = float(point[0][0]) / width
                        y_norm = float(point[0][1]) / height
                        polygon.extend([x_norm, y_norm])
                    
                    # Добавляем данные сегментации (класс, bbox, полигон)
                    segmentation_data.append({
                        'class': yolo_class,
                        'bbox': [x_center, y_center, bbox_width, bbox_height],
                        'polygon': polygon
                    })
    
    return segmentation_data

def save_slice_for_yolo_seg(ct_slice, segmentation_data, output_dir, filename_base):
    """
    Сохраняет срез КТ и соответствующие аннотации в формате YOLOv8-seg
    """
    
    # Создаем директории, если они не существуют
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Нормализуем и преобразуем КТ-срез для сохранения
    # Применяем оконирование для лучшей визуализации костей
    # Типичное окно для костей: центр = 500 HU, ширина = 1500 HU
    window_center = 500
    window_width = 1500
    window_min = window_center - window_width/2
    window_max = window_center + window_width/2
    
    # Ограничиваем значения окном и нормализуем
    ct_slice_windowed = np.clip(ct_slice, window_min, window_max)
    ct_slice_normalized = ((ct_slice_windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)
    
    # Сохраняем изображение
    image_path = os.path.join(images_dir, f"{filename_base}.png")
    cv2.imwrite(image_path, ct_slice_normalized)
    
    # Сохраняем аннотации в формате YOLOv8-seg
    if segmentation_data:
        label_path = os.path.join(labels_dir, f"{filename_base}.txt")
        
        with open(label_path, 'w') as f:
            for data in segmentation_data:
                # Формат YOLOv8-seg: класс x_center y_center width height x1 y1 x2 y2 ... xn yn
                # Сначала записываем класс и ограничивающую рамку
                bbox_str = ' '.join([str(x) for x in data['bbox']])
                
                # Затем записываем полигон
                polygon_str = ' '.join([str(x) for x in data['polygon']])
                
                # Записываем полную строку
                f.write(f"{data['class']} {bbox_str} {polygon_str}\n")


def visualize_segmentation(image_path, label_path):
    """Визуализирует изображение с сегментацией для проверки"""
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    
    # Загрузка аннотаций
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Создаем копию изображения для отрисовки
    vis_image = image.copy()
    
    height, width = image.shape[:2]
    
    # Цвета для разных классов
    colors = [
        (0, 255, 0),    # Зеленый для displaced
        (255, 0, 0),    # Синий для non-displaced
        (0, 0, 255),    # Красный для buckle
        (255, 255, 0)   # Желтый для segmental
    ]
    
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        
        # Получаем координаты ограничивающей рамки
        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
        
        # Преобразуем нормализованные координаты в абсолютные
        x1 = int((x_center - bbox_width/2) * width)
        y1 = int((y_center - bbox_height/2) * height)
        x2 = int((x_center + bbox_width/2) * width)
        y2 = int((y_center + bbox_height/2) * height)
        
        # Рисуем ограничивающую рамку
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), colors[class_id], 2)
        
        # Получаем координаты полигона
        polygon_coords = parts[5:]
        polygon_points = []
        
        for i in range(0, len(polygon_coords), 2):
            if i+1 < len(polygon_coords):
                x = float(polygon_coords[i]) * width
                y = float(polygon_coords[i+1]) * height
                polygon_points.append((int(x), int(y)))
        
        # Рисуем полигон
        if len(polygon_points) > 2:
            cv2.polylines(vis_image, [np.array(polygon_points)], True, colors[class_id], 2)
            
            # Создаем полупрозрачную маску для заливки полигона
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, [np.array(polygon_points)], colors[class_id])
            
            # Накладываем маску с прозрачностью
            alpha = 0.3  # Прозрачность
            vis_image = cv2.addWeighted(vis_image, 1, mask, alpha, 0)
            
            # Добавляем метку класса
            class_names = ['displaced', 'non-displaced', 'buckle', 'segmental']
            label = class_names[class_id]
            cv2.putText(vis_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)
    
    # Отображаем результат
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Сегментация переломов ребер')
    plt.tight_layout()
    plt.show()
    
    return vis_image

def check_annotations(output_dir, num_samples=5):
    """Проверяет случайные образцы созданных аннотаций"""
    
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    
    # Получаем список всех изображений
    image_files = [f for f in os.listdir(train_images_dir) if f.endswith('.png')]
    
    if not image_files:
        print("Изображения не найдены!")
        return
    
    # Выбираем случайные образцы
    import random
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    for sample in samples:
        image_path = os.path.join(train_images_dir, sample)
        label_path = os.path.join(train_labels_dir, sample.replace('.png', '.txt'))
        
        if os.path.exists(label_path):
            print(f"Проверка аннотации для {sample}")
            visualize_segmentation(image_path, label_path)
        else:
            print(f"Файл аннотации не найден для {sample}")

def process_patient_data(image_path, annotation_path, info_df, patient_id, output_dir):
    """Обрабатывает данные одного пациента и сохраняет срезы с переломами для YOLOv8-seg"""
    
    # Загрузка КТ изображения и аннотации
    ct_img = nib.load(image_path)
    ct_data = ct_img.get_fdata()
    
    annotation_img = nib.load(annotation_path)
    annotation_data = annotation_img.get_fdata()
    
    # Проходим по всем срезам
    for z in range(ct_data.shape[2]):
        ct_slice = ct_data[:, :, z]
        annotation_slice = annotation_data[:, :, z]
        
        # Проверяем, есть ли на срезе переломы
        unique_labels = np.unique(annotation_slice)
        unique_labels = unique_labels[unique_labels > 0]  # Исключаем фон (0)
        
        if len(unique_labels) > 0:
            # Создаем данные сегментации для среза
            segmentation_data = create_segmentation_data(annotation_slice, info_df, patient_id)
            
            if segmentation_data:
                # Сохраняем срез и аннотации для YOLOv8-seg
                filename_base = f"{patient_id}_slice_{z}"
                save_slice_for_yolo_seg(ct_slice, segmentation_data, output_dir, filename_base)

def process_dataset(images_dir, labels_dir, info_csv, output_dir):
    """Обрабатывает весь датасет для обучения YOLOv8-seg"""
    
    # Загрузка информации о метках
    info_df = pd.read_csv(info_csv)

    # Создаем копию DataFrame, чтобы избежать предупреждений о изменении исходных данных
    info_df = info_df.copy()
    
    # Заменяем значение -1 на 5 в столбце label_code
    info_df['label_code'] = info_df['label_code'].replace(-1, 5)
    
    # Создаем директории для обучения/валидации
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Получаем уникальные ID пациентов
    patient_ids = info_df['public_id'].unique()
    
    # Разделяем на обучающую и валидационную выборки (80/20)
    np.random.shuffle(patient_ids)
    split_idx = int(len(patient_ids) * 0.8)
    train_ids = patient_ids[:split_idx]
    val_ids = patient_ids[split_idx:]
    
    # Обрабатываем обучающую выборку
    for patient_id in train_ids:
        image_path, label_path = get_single_case_path(patient_id)
        
        if os.path.exists(image_path) and os.path.exists(label_path):
            process_patient_data(image_path, label_path, info_df, patient_id, train_dir)
    
    # Обрабатываем валидационную выборку
    for patient_id in val_ids:
        image_path, label_path = get_single_case_path(patient_id)
        
        if os.path.exists(image_path) and os.path.exists(label_path):
            process_patient_data(image_path, label_path, info_df, patient_id, val_dir)

def create_yaml_config(output_dir, class_names):
    """Создает файл конфигурации data.yaml для YOLOv8-seg"""
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    
    with open(yaml_path, 'w') as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n\n")
        
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
        
        # Добавляем флаг для сегментации
        f.write("task: segment\n")
    
    print(f"Файл конфигурации создан: {yaml_path}")

# Основная функция
def main():
    # Пути к данным
    home = '/home/yura/learning/tiu/doctors/data_raw/'
    images_dir = home + 'ribfrac-val-images'
    labels_dir = home + 'ribfrac-val-labels'
    info_csv = home + 'ribfrac-val-info.csv'
    output_dir = '/home/yura/learning/tiu/doctors/data/yolo_seg_dataset'
    
    # Имена классов (типы переломов)
    class_names = ['displaced', 'non-displaced', 'buckle', 'segmental', 'not-recognized']
    
    # Обработка датасета
    process_dataset(images_dir, labels_dir, info_csv, output_dir)
    
    # Создание файла конфигурации
    create_yaml_config(output_dir, class_names)
    
    print("Обработка датасета завершена!")

if __name__ == "__main__":
    main()

