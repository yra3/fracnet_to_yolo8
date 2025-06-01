import cv2

def extract_precise_contours(mask, min_contour_area=10):
    """
    Извлекает точные контуры из маски с фильтрацией маленьких контуров
    и сглаживанием для лучшего качества сегментации
    """
    # Находим контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Фильтруем маленькие контуры
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_contour_area:
            filtered_contours.append(contour)
    
    # Если нет подходящих контуров, возвращаем пустой список
    if not filtered_contours:
        return []
    
    # Для каждого контура применяем сглаживание
    smoothed_contours = []
    for contour in filtered_contours:
        # Применяем алгоритм Douglas-Peucker для упрощения контура
        # Параметр epsilon контролирует степень упрощения
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Если контур слишком упрощен (менее 4 точек), используем больше точек
        if len(approx_contour) < 4:
            epsilon = 0.0005 * cv2.arcLength(contour, True)
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        smoothed_contours.append(approx_contour)
    
    return smoothed_contours
