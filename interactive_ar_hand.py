import cv2
import mediapipe as mp
import numpy as np
import math
import os
from obj_loader import OBJLoader

class InteractiveARHandTracker:
    def __init__(self):
        # Инициализация MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Инициализация камеры
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Параметры объекта
        self.object_size = 100  # базовый размер (увеличен)
        self.object_rotation_x = 0  # поворот по оси X (pitch)
        self.object_rotation_y = 0  # поворот по оси Y (yaw)
        self.object_rotation_z = 0  # поворот по оси Z (roll)
        # Базовые углы поворота (как в Blender - сохраняют текущее положение при новом жесте)
        self.base_rotation_x = 0
        self.base_rotation_y = 0
        self.base_rotation_z = 0
        self.object_position = [640, 360]  # позиция объекта
        self.object_type = 0  # тип объекта
        
        # Состояния управления
        self.control_mode = "move"  # move, resize, rotate
        self.last_pinch_distance = None
        self.last_hand_position = None  # для отслеживания движения руки в 3D
        self.gesture_cooldown = 0
        
        # Система задержки для активации жестов
        self.gesture_hold_time = {}  # словарь для отслеживания времени удержания жестов
        self.gesture_activation_delay = 15  # 1 секунда при 60 FPS
        self.current_gesture = None
        self.gesture_timer = 0
        
        # Параметры для анимации
        self.scale_factor = 1.0
        self.scale_direction = 1
        
        # Загрузчик 3D моделей
        self.obj_loader = OBJLoader()
        self.custom_models = []
        self.load_custom_models()
        
    def load_custom_models(self):
        """Загружает пользовательские 3D модели из папки models"""
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"📁 Создана папка {models_dir} для 3D моделей")
            print("💡 Поместите .obj файлы в эту папку для загрузки")
            return
        
        obj_files = [f for f in os.listdir(models_dir) if f.endswith('.obj')]
        
        for obj_file in obj_files:
            filepath = os.path.join(models_dir, obj_file)
            loader = OBJLoader()
            
            if loader.load_obj(filepath):
                # Нормализуем модель к стандартному размеру
                loader.normalize_model(100)
                loader.triangulate_faces()
                
                model_data = {
                    'name': obj_file[:-4],  # убираем .obj
                    'vertices': loader.get_vertices(),
                    'faces': loader.get_faces(),
                    'loader': loader
                }
                self.custom_models.append(model_data)
                print(f"✅ Загружена модель: {obj_file}")
        
        if self.custom_models:
            print(f"🎉 Загружено {len(self.custom_models)} пользовательских моделей")
        else:
            print("📝 Нет .obj файлов в папке models")

    def get_distance(self, point1, point2):
        """Вычисляет расстояние между двумя точками"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_angle(self, point1, point2):
        """Вычисляет угол между двумя точками"""
        return math.atan2(point2[1] - point1[1], point2[0] - point1[0])
    
    def get_hand_orientation(self, landmarks, frame_width, frame_height):
        """Вычисляет ориентацию руки в 3D пространстве с интуитивным управлением по всем осям"""
        # Ключевые точки для определения ориентации
        wrist = landmarks[0]
        thumb_tip = landmarks[4]    # кончик большого пальца
        pinky_tip = landmarks[20]   # кончик мизинца
        middle_tip = landmarks[12]  # кончик среднего пальца
        middle_mcp = landmarks[9]   # основание среднего пальца
        index_tip = landmarks[8]    # кончик указательного пальца
        index_mcp = landmarks[5]    # основание указательного пальца
        
        # Преобразуем в пиксельные координаты
        wrist_pos = [wrist.x * frame_width, wrist.y * frame_height, wrist.z]
        thumb_pos = [thumb_tip.x * frame_width, thumb_tip.y * frame_height, thumb_tip.z]
        pinky_pos = [pinky_tip.x * frame_width, pinky_tip.y * frame_height, pinky_tip.z]
        middle_tip_pos = [middle_tip.x * frame_width, middle_tip.y * frame_height, middle_tip.z]
        middle_mcp_pos = [middle_mcp.x * frame_width, middle_mcp.y * frame_height, middle_mcp.z]
        index_tip_pos = [index_tip.x * frame_width, index_tip.y * frame_height, index_tip.z]
        index_mcp_pos = [index_mcp.x * frame_width, index_mcp.y * frame_height, index_mcp.z]
        
        # Основные векторы для интуитивного управления
        # 1. ROLL (Z-axis) - поворот по часовой стрелке: большой палец к мизинцу
        thumb_to_pinky = [
            pinky_pos[0] - thumb_pos[0],
            pinky_pos[1] - thumb_pos[1],
            pinky_pos[2] - thumb_pos[2]
        ]
        
        # 2. PITCH (X-axis) - наклон вверх/вниз: от основания ладони к кончику указательного пальца
        wrist_to_index_tip = [
            index_tip_pos[0] - wrist_pos[0],
            index_tip_pos[1] - wrist_pos[1],
            index_tip_pos[2] - wrist_pos[2]
        ]
        
        # 3. YAW (Y-axis) - поворот влево/вправо: направление среднего пальца
        middle_base_to_tip = [
            middle_tip_pos[0] - middle_mcp_pos[0],
            middle_tip_pos[1] - middle_mcp_pos[1],
            middle_tip_pos[2] - middle_mcp_pos[2]
        ]
        
        # Центр ладони для позиционирования
        palm_center = [
            (thumb_pos[0] + pinky_pos[0] + index_tip_pos[0] + wrist_pos[0]) / 4,
            (thumb_pos[1] + pinky_pos[1] + index_tip_pos[1] + wrist_pos[1]) / 4,
            (thumb_pos[2] + pinky_pos[2] + index_tip_pos[2] + wrist_pos[2]) / 4
        ]
        
        # Вычисляем углы поворота с правильным маппингом
        # ROLL (Z-axis) - вращение по часовой стрелке (thumb-pinky линия)
        roll = math.atan2(thumb_to_pinky[1], thumb_to_pinky[0])
        
        # PITCH (X-axis) - основан на расстоянии от запястья к указательному пальцу
        # Вместо угла используем нормализованное расстояние для более стабильного управления
        wrist_to_index_distance_raw = math.sqrt(
            wrist_to_index_tip[0]**2 + wrist_to_index_tip[1]**2 + wrist_to_index_tip[2]**2
        )
        
        # Нормализуем расстояние к диапазону углов (-90 до +90 градусов)
        # Базовое расстояние при нейтральном положении руки
        base_distance = 120  # примерное расстояние при горизонтальном положении
        distance_variation = wrist_to_index_distance_raw - base_distance
        
        # Преобразуем изменение расстояния в угол (чувствительность можно настроить)
        pitch_sensitivity = 0.8
        pitch = distance_variation * pitch_sensitivity
        
        # YAW (Y-axis) - основан на длине линии thumb-pinky (расстояние между большим и мизинцем)
        # Вместо угла используем нормализованную длину для более стабильного управления
        thumb_pinky_length_raw = math.sqrt(
            thumb_to_pinky[0]**2 + thumb_to_pinky[1]**2 + thumb_to_pinky[2]**2
        )
        
        # Нормализуем длину к диапазону углов
        # Базовая длина при нейтральном положении руки
        base_thumb_pinky_length = 150  # примерная длина при нормальном раскрытии руки
        length_variation = thumb_pinky_length_raw - base_thumb_pinky_length
        
        # Преобразуем изменение длины в угол (чувствительность можно настроить)
        yaw_sensitivity = 1.0
        yaw = length_variation * yaw_sensitivity
        
        # Дополнительные метрики для чувствительности
        thumb_pinky_distance = math.sqrt(
            thumb_to_pinky[0]**2 + thumb_to_pinky[1]**2 + thumb_to_pinky[2]**2
        )
        

        
        middle_finger_length = math.sqrt(
            middle_base_to_tip[0]**2 + middle_base_to_tip[1]**2 + middle_base_to_tip[2]**2
        )
        
        return {
            'roll': math.degrees(roll),      # Z-axis: thumb-pinky rotation
            'pitch': pitch,                  # X-axis: distance-based pitch (already in degrees equivalent)
            'yaw': yaw,                      # Y-axis: thumb-pinky length-based (already in degrees equivalent)
            'thumb_pinky_distance': thumb_pinky_distance,
            'wrist_to_index_distance': wrist_to_index_distance_raw,
            'middle_finger_length': middle_finger_length,
            'pitch_distance_variation': distance_variation,  # для отладки PITCH
            'thumb_pinky_length_raw': thumb_pinky_length_raw,  # для отладки YAW
            'yaw_length_variation': length_variation,  # для отладки YAW
            'center': palm_center[:2],
            'thumb_pos': thumb_pos[:2],
            'pinky_pos': pinky_pos[:2],
            'index_tip_pos': index_tip_pos[:2],
            'index_mcp_pos': index_mcp_pos[:2],
            'middle_tip_pos': middle_tip_pos[:2],
            'middle_mcp_pos': middle_mcp_pos[:2],
            'raw_vectors': {
                'thumb_to_pinky': thumb_to_pinky,
                'wrist_to_index_tip': wrist_to_index_tip,
                'middle_base_to_tip': middle_base_to_tip
            }
        }
    
    def detect_gesture(self, landmarks, frame_width, frame_height):
        """Определяет жест руки с улучшенной детекцией для любой ориентации"""
        # Преобразуем координаты в пиксели и сохраняем 3D информацию
        points = []
        points_3d = []
        for lm in landmarks:
            points.append([int(lm.x * frame_width), int(lm.y * frame_height)])
            points_3d.append([lm.x, lm.y, lm.z])
        
        # Ключевые точки пальцев
        thumb_tip = points[4]
        index_tip = points[8]
        middle_tip = points[12]
        ring_tip = points[16]
        pinky_tip = points[20]
        
        # Основания и суставы пальцев
        thumb_ip = points[3]   # межфаланговый сустав большого пальца
        index_pip = points[6]  # проксимальный межфаланговый сустав указательного
        middle_pip = points[10] # проксимальный межфаланговый сустав среднего
        ring_pip = points[14]   # проксимальный межфаланговый сустав безымянного
        pinky_pip = points[18]  # проксимальный межфаланговый сустав мизинца
        
        wrist = points[0]
        
        # Улучшенная детекция поднятых пальцев с учетом ориентации руки
        fingers_up = []
        
        # Большой палец - используем расстояние от запястья
        thumb_wrist_dist = self.get_distance(thumb_tip, wrist)
        thumb_ip_wrist_dist = self.get_distance(thumb_ip, wrist)
        fingers_up.append(1 if thumb_wrist_dist > thumb_ip_wrist_dist else 0)
        
        # Остальные пальцы - используем расстояние от запястья вместо простого сравнения Y
        finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        finger_pips = [index_pip, middle_pip, ring_pip, pinky_pip]
        
        for tip, pip in zip(finger_tips, finger_pips):
            # Расстояние от кончика пальца до запястья
            tip_wrist_dist = self.get_distance(tip, wrist)
            # Расстояние от сустава пальца до запястья
            pip_wrist_dist = self.get_distance(pip, wrist)
            
            # Палец считается поднятым, если кончик дальше от запястья чем сустав
            # Уменьшаем порог для более чувствительного определения
            fingers_up.append(1 if tip_wrist_dist > pip_wrist_dist * 1.05 else 0)
        
        # Дополнительная проверка для открытой ладони
        # Проверяем расстояния между кончиками пальцев
        finger_spread = 0
        finger_tips_all = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        
        for i in range(len(finger_tips_all)):
            for j in range(i + 1, len(finger_tips_all)):
                dist = self.get_distance(finger_tips_all[i], finger_tips_all[j])
                finger_spread += dist
        
        # Нормализуем по количеству пар
        finger_spread = finger_spread / 10
        
        # Определяем жесты
        total_fingers = sum(fingers_up)
        
        # Расстояние между большим и указательным пальцем для zoom/resize
        zoom_distance = self.get_distance(thumb_tip, index_tip)
        
        # Расстояние между указательным и средним пальцем
        two_finger_distance = self.get_distance(index_tip, middle_tip)
        
        # Щипок (большой + указательный близко)
        is_pinching = zoom_distance < 40
        
        # OK жест (большой + указательный в кольце)
        is_ok = is_pinching and fingers_up[2] == 1 and fingers_up[3] == 1 and fingers_up[4] == 1
        
        # Указательный палец - только указательный поднят
        is_pointing = (fingers_up[1] == 1 and 
                      sum(fingers_up[2:]) == 0 and 
                      fingers_up[0] == 0)
        
        # Zoom жест (большой + указательный) для изменения размера
        is_zoom_gesture = (fingers_up[0] == 1 and 
                          fingers_up[1] == 1 and 
                          sum(fingers_up[2:]) == 0)
        
        # Жест ✌️ для смены объекта (большой + указательный + средний) - улучшенное определение
        # Основной способ - через определение поднятых пальцев
        # Учитываем, что большой палец часто определяется как поднятый
        is_two_fingers_basic = (fingers_up[1] == 1 and 
                               fingers_up[2] == 1 and 
                               fingers_up[0] == 1 and 
                               fingers_up[3] == 0 and 
                               fingers_up[4] == 0 and
                               total_fingers == 3)  # теперь ожидаем 3 пальца
        
        # Альтернативный способ - через расстояния (если основной не работает)
        # Проверяем, что указательный и средний далеко от запястья, а остальные близко
        index_wrist_dist = self.get_distance(index_tip, wrist)
        middle_wrist_dist = self.get_distance(middle_tip, wrist)
        ring_wrist_dist = self.get_distance(ring_tip, wrist)
        pinky_wrist_dist = self.get_distance(pinky_tip, wrist)
        thumb_wrist_dist = self.get_distance(thumb_tip, wrist)
        
        # Средние расстояния для сравнения
        avg_extended = (index_wrist_dist + middle_wrist_dist) / 2
        avg_folded = (ring_wrist_dist + pinky_wrist_dist + thumb_wrist_dist) / 3
        
        is_two_fingers_distance = (avg_extended > avg_folded * 1.2 and 
                                  index_wrist_dist > 80 and 
                                  middle_wrist_dist > 80 and
                                  ring_wrist_dist < 70 and
                                  pinky_wrist_dist < 70)  # убираем проверку большого пальца
        
        # Комбинируем оба способа
        is_two_fingers_change = is_two_fingers_basic or is_two_fingers_distance
        
        # Открытая ладонь - все пальцы подняты И достаточное расстояние между ними
        is_open_palm = (total_fingers >= 4 and finger_spread > 80)
        
        # Четыре пальца (без большого) - альтернативный жест открытой ладони
        is_four_fingers = (sum(fingers_up[1:]) == 4 and 
                          fingers_up[0] == 0 and 
                          finger_spread > 60)
        
        return {
            'fingers_up': fingers_up,
            'total_fingers': total_fingers,
            'finger_spread': finger_spread,
            'is_pinching': is_pinching,
            'is_ok': is_ok,
            'is_pointing': is_pointing,
            'is_zoom_gesture': is_zoom_gesture,
            'is_two_fingers_change': is_two_fingers_change,
            'is_open_palm': is_open_palm,
            'is_four_fingers': is_four_fingers,
            'zoom_distance': zoom_distance,
            'two_finger_distance': two_finger_distance,
            'thumb_tip': thumb_tip,
            'index_tip': index_tip,
            'middle_tip': middle_tip
        }
    
    def handle_gesture_control(self, gesture, landmarks, frame_width, frame_height):
        """Обрабатывает управление объектом через жесты с системой задержки"""
        
        # Определяем текущий активный жест
        detected_gesture = None
        if gesture['is_pointing']:
            detected_gesture = "move"
        elif gesture['is_zoom_gesture']:
            detected_gesture = "resize"
        elif gesture['is_open_palm'] or gesture['is_four_fingers']:
            detected_gesture = "rotate"
        elif gesture['is_two_fingers_change']:
            detected_gesture = "change_object"
        
        # Система задержки для активации жестов
        if detected_gesture:
            if self.current_gesture == detected_gesture:
                # Продолжаем отсчет для того же жеста
                self.gesture_timer += 1
                
                # Если жест удерживается достаточно долго, активируем режим
                if self.gesture_timer >= self.gesture_activation_delay:
                    if detected_gesture in ["move", "resize", "rotate"] and self.gesture_cooldown <= 0:
                        self.control_mode = detected_gesture
                        self.gesture_cooldown = 30
                    elif detected_gesture == "change_object" and self.gesture_cooldown <= 0:
                         total_objects = 1 + len(self.custom_models)  # только куб + пользовательские модели
                         self.object_type = (self.object_type + 1) % total_objects
                         self.gesture_cooldown = 60
                         self.gesture_timer = 0  # сбрасываем таймер после смены объекта
            else:
                # Новый жест - начинаем отсчет заново
                self.current_gesture = detected_gesture
                self.gesture_timer = 1
                
                # Если начинается новый жест вращения, сохраняем текущее положение как базовое
                if detected_gesture == "rotate":
                    # Сохраняем текущие углы как базовые (как в Blender)
                    self.base_rotation_x += self.object_rotation_x
                    self.base_rotation_y += self.object_rotation_y
                    self.base_rotation_z += self.object_rotation_z
                    # Сбрасываем текущие углы для нового жеста
                    self.object_rotation_x = 0
                    self.object_rotation_y = 0
                    self.object_rotation_z = 0
        else:
            # Нет активного жеста - сбрасываем
            self.current_gesture = None
            self.gesture_timer = 0
        
        # Управление в зависимости от режима (только если режим активирован)
        if self.control_mode == "move":
            # Перемещение объекта указательным пальцем (только если режим полностью активирован)
            if gesture['is_pointing'] and self.gesture_timer >= self.gesture_activation_delay:
                self.object_position = list(gesture['index_tip'])
                
        elif self.control_mode == "resize":
            # Изменение размера zoom жестом (большой + указательный)
            if gesture['is_zoom_gesture'] and self.gesture_timer >= self.gesture_activation_delay:
                if self.last_pinch_distance is not None:
                    # Используем расстояние между большим и указательным пальцем
                    size_change = gesture['zoom_distance'] - self.last_pinch_distance
                    self.object_size += size_change * 0.8  # настраиваем чувствительность
                    self.object_size = max(5, self.object_size)  # минимальный размер для видимости
                self.last_pinch_distance = gesture['zoom_distance']
            else:
                self.last_pinch_distance = None
                
        elif self.control_mode == "rotate":
            # Вращение объекта открытой ладонью используя большой палец и мизинец как ориентиры
            if (gesture['is_open_palm'] or gesture['is_four_fingers']) and self.gesture_timer >= self.gesture_activation_delay:
                # Получаем ориентацию руки в 3D
                hand_orientation = self.get_hand_orientation(landmarks, frame_width, frame_height)
                
                if self.last_hand_position is not None:
                    # Вычисляем изменения углов с правильным маппингом
                    roll_change = hand_orientation['roll'] - self.last_hand_position['roll']
                    pitch_change = hand_orientation['pitch'] - self.last_hand_position['pitch']
                    yaw_change = hand_orientation['yaw'] - self.last_hand_position['yaw']
                    
                    # Обрабатываем переходы через 180/-180 градусов
                    if abs(roll_change) > 180:
                        roll_change = roll_change - 360 if roll_change > 0 else roll_change + 360
                    if abs(pitch_change) > 180:
                        pitch_change = pitch_change - 360 if pitch_change > 0 else pitch_change + 360
                    if abs(yaw_change) > 180:
                        yaw_change = yaw_change - 360 if yaw_change > 0 else yaw_change + 360
                    
                    # Адаптивная чувствительность
                    base_sensitivity = 1.0
                    
                    # Разная чувствительность для каждой оси
                    roll_sensitivity = base_sensitivity * 1.5   # thumb-pinky очень заметен
                    pitch_sensitivity = base_sensitivity * 1.2  # указательный палец
                    yaw_sensitivity = base_sensitivity * 1.2    # средний палец
                    
                    # Применяем изменения с интуитивным маппингом
                    # ROLL: thumb-pinky линия -> Z-axis (вращение по часовой стрелке)
                    self.object_rotation_z += roll_change * roll_sensitivity
                    
                    # PITCH: указательный палец вверх/вниз -> X-axis (инвертировано)
                    self.object_rotation_x -= pitch_change * pitch_sensitivity
                    
                    # YAW: длина thumb-pinky линии -> Y-axis
                    self.object_rotation_y += yaw_change * yaw_sensitivity
                    
                    # Нормализуем углы
                    self.object_rotation_x = self.object_rotation_x % 360
                    self.object_rotation_y = self.object_rotation_y % 360
                    self.object_rotation_z = self.object_rotation_z % 360
                
                self.last_hand_position = hand_orientation
            else:
                self.last_hand_position = None
        
        # Уменьшаем кулдаун
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
    
    def apply_3d_rotation(self, point, rx, ry, rz):
        """Применяет 3D поворот к точке"""
        x, y, z = point
        
        # Поворот вокруг X (pitch)
        cos_x, sin_x = math.cos(math.radians(rx)), math.sin(math.radians(rx))
        y, z = y * cos_x - z * sin_x, y * sin_x + z * cos_x
        
        # Поворот вокруг Y (yaw)
        cos_y, sin_y = math.cos(math.radians(ry)), math.sin(math.radians(ry))
        x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y
        
        # Поворот вокруг Z (roll)
        cos_z, sin_z = math.cos(math.radians(rz)), math.sin(math.radians(rz))
        x, y = x * cos_z - y * sin_z, x * sin_z + y * cos_z
        
        return [x, y, z]
    
    def perspective_projection(self, point_3d, center_x, center_y, focal_length=800, camera_distance=500):
        """
        Настоящая перспективная проекция как в 3D движках
        
        Args:
            point_3d: 3D точка [x, y, z]
            center_x, center_y: центр экрана
            focal_length: фокусное расстояние камеры (больше = меньше искажений)
            camera_distance: расстояние от камеры до объекта
        
        Returns:
            [x_2d, y_2d]: 2D координаты на экране
        """
        x, y, z = point_3d
        
        # Сдвигаем объект от камеры
        z_camera = z + camera_distance
        
        # Избегаем деления на ноль или отрицательные значения
        if z_camera <= 0:
            z_camera = 1
        
        # Перспективная проекция: x' = f * x / z, y' = f * y / z
        x_projected = (focal_length * x) / z_camera
        y_projected = (focal_length * y) / z_camera
        
        # Переводим в экранные координаты
        screen_x = center_x + int(x_projected)
        screen_y = center_y + int(y_projected)
        
        return [screen_x, screen_y]
    
    def draw_rotated_cube(self, frame, center_x, center_y, size, rx, ry, rz):
        """Рисует куб с 3D поворотом и правильной сортировкой граней"""
        half_size = size // 2
        
        # 8 вершин куба в 3D
        vertices_3d = [
            [-half_size, -half_size, -half_size], [half_size, -half_size, -half_size],
            [half_size, half_size, -half_size], [-half_size, half_size, -half_size],
            [-half_size, -half_size, half_size], [half_size, -half_size, half_size],
            [half_size, half_size, half_size], [-half_size, half_size, half_size]
        ]
        
        # Применяем 3D поворот и проецируем на 2D
        vertices_2d = []
        vertices_3d_rotated = []
        
        for vertex in vertices_3d:
            rotated = self.apply_3d_rotation(vertex, rx, ry, rz)
            vertices_3d_rotated.append(rotated)
            
            # Настоящая перспективная проекция
            projected = self.perspective_projection(rotated, center_x, center_y, 
                                                  focal_length=600, camera_distance=400)
            vertices_2d.append(projected)
        
        # Определяем грани куба (порядок вершин важен для правильного отображения)
        faces = [
            [0, 1, 2, 3],  # передняя грань
            [4, 7, 6, 5],  # задняя грань
            [0, 4, 5, 1],  # нижняя грань
            [2, 6, 7, 3],  # верхняя грань
            [0, 3, 7, 4],  # левая грань
            [1, 5, 6, 2]   # правая грань
        ]
        
        # Цвета граней (более серые и приглушенные)
        colors = [
            (150, 120, 120),  # приглушенный красный
            (120, 120, 150),  # приглушенный синий
            (120, 150, 120),  # приглушенный зеленый
            (150, 150, 120),  # приглушенный желтый
            (150, 120, 150),  # приглушенная магента
            (120, 150, 150)   # приглушенный циан
        ]
        
        # Вычисляем среднюю глубину для каждой грани и сортируем
        faces_with_depth = []
        for i, face in enumerate(faces):
            # Вычисляем среднюю Z-координату грани
            avg_z = sum(vertices_3d_rotated[j][2] for j in face) / len(face)
            faces_with_depth.append((avg_z, i, face))
        
        # Сортируем грани по глубине (дальние сначала - painter's algorithm)
        faces_with_depth.sort(key=lambda x: x[0], reverse=True)
        
        # Рисуем грани в правильном порядке (от дальних к ближним)
        for depth, face_index, face in faces_with_depth:
            face_points = np.array([vertices_2d[j] for j in face], np.int32)
            
            # Добавляем затенение в зависимости от глубины
            depth_factor = max(0.4, min(1.0, 1.0 - (depth / (half_size * 2))))
            
            base_color = colors[face_index]
            shaded_color = (
                int(base_color[0] * depth_factor),
                int(base_color[1] * depth_factor),
                int(base_color[2] * depth_factor)
            )
            
            cv2.fillPoly(frame, [face_points], shaded_color)
            cv2.polylines(frame, [face_points], True, (0, 0, 0), 2)
        
        return frame
    
    def draw_rotated_pyramid(self, frame, center_x, center_y, size, rx, ry, rz):
        """Рисует пирамиду с 3D поворотом и правильной сортировкой граней"""
        half_size = size // 2
        
        # 5 вершин пирамиды в 3D
        vertices_3d = [
            [-half_size, -half_size, -half_size],  # основание
            [half_size, -half_size, -half_size],
            [half_size, half_size, -half_size],
            [-half_size, half_size, -half_size],
            [0, 0, half_size]  # вершина пирамиды
        ]
        
        # Применяем 3D поворот и проецируем на 2D
        vertices_2d = []
        vertices_3d_rotated = []
        
        for vertex in vertices_3d:
            rotated = self.apply_3d_rotation(vertex, rx, ry, rz)
            vertices_3d_rotated.append(rotated)
            
            # Настоящая перспективная проекция
            projected = self.perspective_projection(rotated, center_x, center_y, 
                                                  focal_length=600, camera_distance=400)
            vertices_2d.append(projected)
        
        # Определяем грани пирамиды
        faces = [
            [0, 1, 2, 3],  # основание
            [0, 1, 4],     # боковая грань 1
            [1, 2, 4],     # боковая грань 2
            [2, 3, 4],     # боковая грань 3
            [3, 0, 4]      # боковая грань 4
        ]
        
        # Цвета граней (более серые и приглушенные)
        colors = [
            (120, 150, 120),  # приглушенный зеленый - основание
            (150, 140, 120),  # приглушенный оранжевый
            (140, 150, 120),  # приглушенный светло-зеленый
            (120, 140, 150),  # приглушенный голубой
            (150, 120, 140)   # приглушенный розовый
        ]
        
        # Вычисляем среднюю глубину для каждой грани и сортируем
        faces_with_depth = []
        for i, face in enumerate(faces):
            # Вычисляем среднюю Z-координату грани
            avg_z = sum(vertices_3d_rotated[j][2] for j in face) / len(face)
            faces_with_depth.append((avg_z, i, face))
        
        # Сортируем грани по глубине (дальние сначала - painter's algorithm)
        faces_with_depth.sort(key=lambda x: x[0], reverse=True)
        
        # Рисуем грани в правильном порядке (от дальних к ближним)
        for depth, face_index, face in faces_with_depth:
            face_points = np.array([vertices_2d[j] for j in face], np.int32)
            
            # Добавляем затенение в зависимости от глубины
            depth_factor = max(0.4, min(1.0, 1.0 - (depth / (half_size * 2))))
            
            base_color = colors[face_index]
            shaded_color = (
                int(base_color[0] * depth_factor),
                int(base_color[1] * depth_factor),
                int(base_color[2] * depth_factor)
            )
            
            cv2.fillPoly(frame, [face_points], shaded_color)
            cv2.polylines(frame, [face_points], True, (0, 0, 0), 2)
        
        return frame

    def draw_floating_sphere(self, frame, center_x, center_y, size):
        """Рисует анимированную сферу"""
        animated_size = int(size * self.scale_factor * 0.5)
        
        cv2.circle(frame, (center_x, center_y), animated_size, (255, 255, 100), -1)
        cv2.circle(frame, (center_x, center_y), animated_size, (200, 200, 50), 3)
        
        highlight_offset = animated_size // 3
        cv2.circle(frame, 
                  (center_x - highlight_offset, center_y - highlight_offset), 
                  animated_size // 4, (255, 255, 255), -1)
        
        return frame
    
    def draw_custom_model(self, frame, center_x, center_y, size, rx, ry, rz, model_index):
        """Рисует пользовательскую 3D модель с правильной сортировкой по глубине"""
        if model_index >= len(self.custom_models):
            return frame
        
        model = self.custom_models[model_index]
        vertices_3d = model['vertices']
        faces = model['faces']
        
        # Масштабируем модель под текущий размер
        scale = size / 100  # базовый размер модели 100
        
        # Применяем 3D поворот и проецируем на 2D
        vertices_2d = []
        vertices_3d_rotated = []
        
        for vertex in vertices_3d:
            # Масштабируем
            scaled_vertex = [v * scale for v in vertex]
            
            # Поворачиваем
            rotated = self.apply_3d_rotation(scaled_vertex, rx, ry, rz)
            vertices_3d_rotated.append(rotated)
            
            # Настоящая перспективная проекция
            projected = self.perspective_projection(rotated, center_x, center_y, 
                                                  focal_length=300, camera_distance=400)
            vertices_2d.append(projected)
        
        # Вычисляем среднюю глубину (Z) для каждой грани и сортируем
        faces_with_depth = []
        
        for i, face in enumerate(faces):
            if len(face) >= 3:  # Только валидные грани
                try:
                    # Вычисляем среднюю Z-координату грани
                    valid_vertices = [j for j in face if j < len(vertices_3d_rotated)]
                    if len(valid_vertices) >= 3:
                        avg_z = sum(vertices_3d_rotated[j][2] for j in valid_vertices) / len(valid_vertices)
                        faces_with_depth.append((avg_z, i, face))
                except (IndexError, ValueError):
                    continue
        
        # Сортируем грани по глубине (дальние сначала - painter's algorithm)
        faces_with_depth.sort(key=lambda x: x[0], reverse=True)
        
        # Генерируем цвета для граней
        num_faces = len(faces)
        colors = []
        for i in range(num_faces):
            # Создаем более серые и приглушенные цвета
            hue = (i * 137.5) % 360  # золотое сечение для равномерного распределения
            saturation = 0.3  # уменьшена насыщенность для более серого вида
            value = 0.6       # уменьшена яркость для более приглушенного вида
            
            # Простое HSV -> RGB преобразование
            c = value * saturation
            x = c * (1 - abs((hue / 60) % 2 - 1))
            m = value - c
            
            if 0 <= hue < 60:
                r, g, b = c, x, 0
            elif 60 <= hue < 120:
                r, g, b = x, c, 0
            elif 120 <= hue < 180:
                r, g, b = 0, c, x
            elif 180 <= hue < 240:
                r, g, b = 0, x, c
            elif 240 <= hue < 300:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            colors.append((int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)))
        
        # Рисуем грани в правильном порядке (от дальних к ближним)
        for depth, face_index, face in faces_with_depth:
            try:
                face_points = np.array([vertices_2d[j] for j in face if j < len(vertices_2d)], np.int32)
                if len(face_points) >= 3:
                    # Добавляем затенение в зависимости от глубины
                    depth_factor = max(0.3, min(1.0, 1.0 - (depth / (size * 2))))
                    
                    base_color = colors[face_index % len(colors)]
                    shaded_color = (
                        int(base_color[0] * depth_factor),
                        int(base_color[1] * depth_factor),
                        int(base_color[2] * depth_factor)
                    )
                    
                    cv2.fillPoly(frame, [face_points], shaded_color)
                    cv2.polylines(frame, [face_points], True, (0, 0, 0), 1)
            except (IndexError, ValueError):
                # Пропускаем поврежденные грани
                continue
        
        return frame
    
    def draw_rotation_guides(self, frame, landmarks, frame_width, frame_height):
        """Рисует визуальные ориентиры для вращения (большой палец и мизинец)"""
        if self.control_mode == "rotate" and (self.current_gesture == "rotate" and self.gesture_timer >= self.gesture_activation_delay):
            # Получаем позиции большого пальца и мизинца
            thumb_tip = landmarks[4]
            pinky_tip = landmarks[20]
            
            thumb_pos = [int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)]
            pinky_pos = [int(pinky_tip.x * frame_width), int(pinky_tip.y * frame_height)]
            
            # Рисуем линию между большим пальцем и мизинцем (ось Roll)
            cv2.line(frame, tuple(thumb_pos), tuple(pinky_pos), (255, 255, 0), 3)
            
            # Рисуем большой палец (Roll control)
            cv2.circle(frame, tuple(thumb_pos), 12, (255, 100, 100), -1)
            cv2.circle(frame, tuple(thumb_pos), 15, (255, 255, 255), 2)
            cv2.putText(frame, "ROLL", (thumb_pos[0] + 20, thumb_pos[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
            
            # Рисуем мизинец (Roll control)
            cv2.circle(frame, tuple(pinky_pos), 12, (100, 100, 255), -1)
            cv2.circle(frame, tuple(pinky_pos), 15, (255, 255, 255), 2)
            cv2.putText(frame, "ROLL", (pinky_pos[0] + 20, pinky_pos[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 2)
            
            # Рисуем ориентиры для PITCH (от запястья к указательному пальцу)
            wrist = landmarks[0]
            index_tip = landmarks[8]
            
            wrist_pos = [int(wrist.x * frame_width), int(wrist.y * frame_height)]
            index_tip_pos = [int(index_tip.x * frame_width), int(index_tip.y * frame_height)]
            
            # Линия от запястья к указательному пальцу (Pitch - X-axis)
            cv2.line(frame, tuple(wrist_pos), tuple(index_tip_pos), (255, 150, 100), 3)
            cv2.circle(frame, tuple(wrist_pos), 6, (255, 150, 100), -1)
            cv2.circle(frame, tuple(index_tip_pos), 8, (255, 150, 100), -1)
            cv2.circle(frame, tuple(index_tip_pos), 12, (255, 255, 255), 2)
            cv2.putText(frame, "PITCH", (index_tip_pos[0] + 15, index_tip_pos[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 100), 2)
            
            # Рисуем ориентиры для YAW (длина thumb-pinky линии)
            # YAW теперь основан на длине желтой линии между большим пальцем и мизинцем
            # Добавляем визуальную подсказку о том, что длина линии влияет на YAW
            line_center_x = (thumb_pos[0] + pinky_pos[0]) // 2
            line_center_y = (thumb_pos[1] + pinky_pos[1]) // 2
            cv2.putText(frame, "YAW LENGTH", (line_center_x - 30, line_center_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 2)
    
    def draw_gesture_info(self, frame, gesture, control_mode):
        """Рисует информацию о жестах и управлении"""
        height, width = frame.shape[:2]
        
        # Основная панель (увеличиваем высоту для дополнительной информации)
        panel_height = 300 if (self.current_gesture and self.gesture_timer > 0) else 250
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (520, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Заголовок
        cv2.putText(frame, "Interactive AR Hand Tracker", (20, 35), font, 0.7, (255, 255, 255), 2)
        
        # Текущий режим с индикацией активности
        mode_colors = {"move": (100, 255, 100), "resize": (255, 100, 100), "rotate": (100, 100, 255)}
        
        # Проверяем, активен ли текущий режим
        is_mode_active = (
            self.current_gesture and 
            self.gesture_timer >= self.gesture_activation_delay and
            ((control_mode == "move" and self.current_gesture == "move") or
             (control_mode == "resize" and self.current_gesture == "resize") or
             (control_mode == "rotate" and self.current_gesture == "rotate"))
        )
        
        mode_text = f"Mode: {control_mode.upper()}"
        if is_mode_active:
            mode_text += " [ACTIVE]"
            mode_color = mode_colors[control_mode]
        else:
            mode_text += " [WAITING]"
            mode_color = (150, 150, 150)  # серый цвет для неактивного режима
            
        cv2.putText(frame, mode_text, (20, 65), font, 0.6, mode_color, 2)
        
        # Показываем прогресс активации жеста
        if self.current_gesture and self.gesture_timer > 0:
            progress = min(self.gesture_timer / self.gesture_activation_delay, 1.0)
            progress_text = f"Activating {self.current_gesture}: {int(progress * 100)}%"
            progress_color = (255, 255, 0) if progress < 1.0 else (0, 255, 0)
            cv2.putText(frame, progress_text, (20, 260), font, 0.5, progress_color, 2)
            
            # Прогресс-бар
            bar_width = 200
            bar_height = 10
            bar_x, bar_y = 20, 275
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), progress_color, -1)
        
        # Параметры объекта (показываем общие углы как в Blender)
        total_rotation_x = self.base_rotation_x + self.object_rotation_x
        total_rotation_y = self.base_rotation_y + self.object_rotation_y
        total_rotation_z = self.base_rotation_z + self.object_rotation_z
        cv2.putText(frame, f"Size: {int(self.object_size)}", (20, 90), font, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Rotation X: {int(total_rotation_x)}° ({int(self.object_rotation_x)}°)", (20, 110), font, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Rotation Y: {int(total_rotation_y)}° ({int(self.object_rotation_y)}°)", (20, 125), font, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Rotation Z: {int(total_rotation_z)}° ({int(self.object_rotation_z)}°)", (20, 140), font, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Position: ({self.object_position[0]}, {self.object_position[1]})", (20, 155), font, 0.4, (200, 200, 200), 1)
        
        # Отладочная информация для PITCH и YAW (только в режиме вращения)
        if self.control_mode == 'rotate' and hasattr(self, 'last_hand_position') and self.last_hand_position:
            wrist_distance = self.last_hand_position.get('wrist_to_index_distance', 0)
            distance_var = self.last_hand_position.get('pitch_distance_variation', 0)
            thumb_pinky_length = self.last_hand_position.get('thumb_pinky_length_raw', 0)
            yaw_length_var = self.last_hand_position.get('yaw_length_variation', 0)
            
            cv2.putText(frame, f"Wrist-Index dist: {int(wrist_distance)}", (20, 170), font, 0.35, (150, 150, 150), 1)
            cv2.putText(frame, f"Pitch variation: {int(distance_var)}", (20, 185), font, 0.35, (150, 150, 150), 1)
            cv2.putText(frame, f"Thumb-Pinky length: {int(thumb_pinky_length)}", (20, 200), font, 0.35, (150, 150, 150), 1)
            cv2.putText(frame, f"Yaw variation: {int(yaw_length_var)}", (20, 215), font, 0.35, (150, 150, 150), 1)
        
        # Информация о текущем объекте
        total_objects = 1 + len(self.custom_models)
        object_names = ["Cube"] + [model['name'] for model in self.custom_models]
        current_object_name = object_names[self.object_type] if self.object_type < len(object_names) else "Unknown"
        cv2.putText(frame, f"Object: {self.object_type}/{total_objects-1} - {current_object_name}", (20, 175), font, 0.5, (100, 255, 255), 2)
        
        # Инструкции
        cv2.putText(frame, "Gestures:", (20, 195), font, 0.5, (255, 255, 100), 1)
        cv2.putText(frame, "1 finger - move | thumb+index - zoom", (20, 210), font, 0.4, (150, 150, 150), 1)
        cv2.putText(frame, "open palm - 3D rotate | peace sign ✌️ - change object", (20, 225), font, 0.4, (150, 150, 150), 1)
        cv2.putText(frame, "Rotation: thumb-pinky=ROLL+YAW, index distance=PITCH", (20, 240), font, 0.4, (100, 255, 255), 1)
        
        # Панель справа для состояния жестов
        cv2.rectangle(overlay, (width-280, 10), (width-10, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "Gestures:", (width-270, 35), font, 0.5, (255, 255, 100), 1)
        
        # Показываем состояние пальцев с подробной информацией
        fingers_text = "".join([str(f) for f in gesture['fingers_up']])
        cv2.putText(frame, f"Fingers: {fingers_text}", (width-270, 55), font, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Total: {gesture.get('total_fingers', 0)}", (width-270, 70), font, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Spread: {int(gesture.get('finger_spread', 0))}", (width-270, 85), font, 0.4, (200, 200, 200), 1)
        
        # Специальная отладка для жеста ✌️
        fingers = gesture.get('fingers_up', [0,0,0,0,0])
        if fingers[1] == 1 and fingers[2] == 1:
            debug_color = (100, 255, 100) if gesture.get('is_two_fingers_change', False) else (255, 100, 100)
            cv2.putText(frame, "✌️ Thumb+Index+Middle", (width-270, 100), font, 0.35, debug_color, 1)
        
        # Показываем, если жест ✌️ активен
        if gesture.get('is_two_fingers_change', False):
            cv2.putText(frame, "✌️ PEACE SIGN DETECTED!", (20, height-30), font, 0.6, (0, 255, 0), 2)
        
        gesture_status = [
            f"Pointing: {'✓' if gesture['is_pointing'] else '✗'}",
            f"Zoom gesture: {'✓' if gesture['is_zoom_gesture'] else '✗'}",
            f"Peace sign ✌️ (change): {'✓' if gesture['is_two_fingers_change'] else '✗'}",
            f"Open palm: {'✓' if gesture['is_open_palm'] else '✗'}",
            f"Four fingers: {'✓' if gesture['is_four_fingers'] else '✗'}"
        ]
        
        for i, status in enumerate(gesture_status):
            color = (100, 255, 100) if '✓' in status else (100, 100, 100)
            cv2.putText(frame, status, (width-270, 115 + i*15), font, 0.35, color, 1)
        
        return frame
    
    def run(self):
        """Основной цикл программы"""
        print("🎮 Interactive AR Hand Tracker запущен!")
        print("Управление жестами (с задержкой 1 секунда):")
        print("- 1 палец (указательный) - перемещение объекта")
        print("- Большой + указательный палец - zoom (изменение размера)")
        print("- Открытая ладонь (5 пальцев) - 3D вращение с интуитивным управлением:")
        print("  🔄 Большой палец ↔ Мизинец = ROLL (поворот по часовой стрелке)")
        print("  👆 Расстояние указательного пальца = PITCH (наклон объекта)")
        print("  📏 Длина линии большой-мизинец = YAW (поворот объекта)")
        print("- Знак мира ✌️ (большой + указательный + средний) - смена типа объекта")
        print("- Удерживайте жест 1 секунду для активации!")
        print(f"📦 Доступные объекты: Куб + {len(self.custom_models)} пользовательских моделей")
        print("- ESC - выход")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Ошибка захвата видео")
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(frame_rgb)
            
            height, width = frame.shape[:2]
            
            # Обработка рук
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Рисуем скелет руки
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Определяем жест
                    gesture = self.detect_gesture(hand_landmarks.landmark, width, height)
                    
                    # Обрабатываем управление
                    self.handle_gesture_control(gesture, hand_landmarks.landmark, width, height)
                    
                    # Рисуем информацию о жестах
                    frame = self.draw_gesture_info(frame, gesture, self.control_mode)
                    
                    # Рисуем ориентиры для вращения
                    self.draw_rotation_guides(frame, hand_landmarks.landmark, width, height)
            else:
                # Если рук нет, показываем базовую информацию
                empty_gesture = {
                    'is_pointing': False, 'is_zoom_gesture': False, 'is_two_fingers_change': False,
                    'is_pinching': False, 'is_open_palm': False, 'is_four_fingers': False,
                    'fingers_up': [0, 0, 0, 0, 0], 'finger_spread': 0
                }
                frame = self.draw_gesture_info(frame, empty_gesture, self.control_mode)
            
            # Рисуем объект в текущей позиции с 3D поворотом
            total_objects = 1 + len(self.custom_models)  # только куб + пользовательские модели
            
            if self.object_type == 0:
                # Передаем сумму базовых и текущих углов (как в Blender)
                total_rotation_x = self.base_rotation_x + self.object_rotation_x
                total_rotation_y = self.base_rotation_y + self.object_rotation_y
                total_rotation_z = self.base_rotation_z + self.object_rotation_z
                frame = self.draw_rotated_cube(frame, self.object_position[0], self.object_position[1], 
                                             int(self.object_size), total_rotation_x, 
                                             total_rotation_y, total_rotation_z)
            elif self.object_type >= 1 and self.custom_models:
                # Пользовательские модели
                model_index = (self.object_type - 1) % len(self.custom_models)
                # Передаем сумму базовых и текущих углов (как в Blender)
                total_rotation_x = self.base_rotation_x + self.object_rotation_x
                total_rotation_y = self.base_rotation_y + self.object_rotation_y
                total_rotation_z = self.base_rotation_z + self.object_rotation_z
                frame = self.draw_custom_model(frame, self.object_position[0], self.object_position[1],
                                             int(self.object_size), total_rotation_x,
                                             total_rotation_y, total_rotation_z, model_index)
            
            # Обновление анимации для сферы
            self.scale_factor += 0.02 * self.scale_direction
            if self.scale_factor >= 1.3 or self.scale_factor <= 0.7:
                self.scale_direction *= -1
            
            cv2.imshow('Interactive AR Hand Tracker', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Interactive AR Hand Tracker завершен")

if __name__ == "__main__":
    ar_tracker = InteractiveARHandTracker()
    ar_tracker.run() 