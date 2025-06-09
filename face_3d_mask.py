import cv2
import mediapipe as mp
import numpy as np
import math
import os

class Face3DMask:
    def __init__(self):
        # Инициализация MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Инициализация камеры
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Настройки
        self.current_mask = 'glasses'
        self.image_scale = 1.0
        self.show_landmarks = False
        self.show_3d_debug = False
        
        # Загружаем изображения масок
        self.mask_images = {}
        self.load_mask_images()
        
        # 3D модельные точки лица (приблизительные координаты в мм)
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),        # Кончик носа
            (0.0, -330.0, -65.0),   # Подбородок
            (-225.0, 170.0, -135.0), # Левый угол глаза
            (225.0, 170.0, -135.0),  # Правый угол глаза
            (-150.0, -150.0, -125.0), # Левый угол рта
            (150.0, -150.0, -125.0),  # Правый угол рта
        ], dtype=np.float64)
        
        # Соответствующие индексы точек MediaPipe
        self.model_points_indices = [1, 152, 33, 263, 61, 291]
        
        # Параметры камеры (приблизительные для веб-камеры)
        self.focal_length = 1280
        self.camera_center = (640, 360)
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.camera_center[0]],
            [0, self.focal_length, self.camera_center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros((4, 1))  # Предполагаем отсутствие искажений
        
        # Ключевые точки для разных типов масок
        self.face_regions = {
            'eyes': {
                'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
                'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
                'center_points': [33, 133, 362, 263]
            },
            'nose': {
                'tip': [1, 2],
                'bridge': [6, 8, 9, 10]
            },
            'forehead': [10, 151, 9, 8, 107, 55],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        }
        
        # 3D предустановки масок
        self.mask_presets = {
            'glasses': {
                'region': 'eyes',
                'image': 'glasses.png',
                'scale': 1.5,
                'offset_3d': (0, 0, 20),  # смещение в 3D пространстве (x, y, z)
                'plane_size': (200, 80),   # размер 3D плоскости
                'anchor_points': [33, 133, 362, 263]  # точки привязки
            },
            'mustache': {
                'region': 'nose',
                'image': 'mustache.png',
                'scale': 0.8,
                'offset_3d': (0, -30, 15),
                'plane_size': (120, 60),
                'anchor_points': [1, 2, 20, 94]
            },
            'crown': {
                'region': 'forehead',
                'image': 'crown.png',
                'scale': 1.2,
                'offset_3d': (0, 80, 0),
                'plane_size': (180, 100),
                'anchor_points': [10, 151, 9, 8]
            },
            'heart_eyes': {
                'region': 'eyes',
                'image': 'heart.png',
                'scale': 0.6,
                'offset_3d': (0, 0, 25),
                'plane_size': (60, 60),
                'anchor_points': [33, 133, 362, 263]
            }
        }
    
    def create_default_images(self):
        """Создает простые изображения масок если файлы не найдены"""
        mask_dir = "mask_images"
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        
        # Создаем простые очки
        glasses = np.zeros((100, 200, 4), dtype=np.uint8)
        cv2.circle(glasses, (50, 50), 35, (0, 0, 0, 200), -1)
        cv2.circle(glasses, (50, 50), 30, (0, 0, 0, 0), -1)
        cv2.circle(glasses, (50, 50), 35, (255, 255, 255, 255), 3)
        cv2.circle(glasses, (150, 50), 35, (0, 0, 0, 200), -1)
        cv2.circle(glasses, (150, 50), 30, (0, 0, 0, 0), -1)
        cv2.circle(glasses, (150, 50), 35, (255, 255, 255, 255), 3)
        cv2.line(glasses, (85, 50), (115, 50), (255, 255, 255, 255), 3)
        cv2.imwrite(f"{mask_dir}/glasses.png", glasses)
        
        # Создаем усы
        mustache = np.zeros((60, 120, 4), dtype=np.uint8)
        pts = np.array([[20, 40], [40, 20], [60, 30], [80, 20], [100, 40], 
                       [90, 50], [60, 45], [30, 50]], np.int32)
        cv2.fillPoly(mustache, [pts], (50, 25, 0, 255))
        cv2.imwrite(f"{mask_dir}/mustache.png", mustache)
        
        # Создаем корону
        crown = np.zeros((80, 150, 4), dtype=np.uint8)
        cv2.rectangle(crown, (10, 60), (140, 75), (255, 215, 0, 255), -1)
        for i in range(5):
            x = 20 + i * 25
            pts = np.array([[x, 60], [x+10, 20], [x+20, 60]], np.int32)
            cv2.fillPoly(crown, [pts], (255, 215, 0, 255))
        for i in range(3):
            x = 35 + i * 40
            cv2.circle(crown, (x, 40), 5, (255, 0, 0, 255), -1)
        cv2.imwrite(f"{mask_dir}/crown.png", crown)
        
        # Создаем сердечки
        heart = np.zeros((60, 60, 4), dtype=np.uint8)
        cv2.circle(heart, (20, 25), 15, (255, 0, 100, 255), -1)
        cv2.circle(heart, (40, 25), 15, (255, 0, 100, 255), -1)
        pts = np.array([[30, 35], [15, 50], [45, 50]], np.int32)
        cv2.fillPoly(heart, [pts], (255, 0, 100, 255))
        cv2.imwrite(f"{mask_dir}/heart.png", heart)
    
    def load_mask_images(self):
        """Загружает изображения масок"""
        mask_dir = "mask_images"
        
        if not os.path.exists(mask_dir):
            self.create_default_images()
        
        image_files = ['glasses.png', 'mustache.png', 'crown.png', 'heart.png']
        
        for filename in image_files:
            filepath = os.path.join(mask_dir, filename)
            if os.path.exists(filepath):
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.shape[2] == 3:
                        alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
                        img = np.concatenate([img, alpha], axis=2)
                    
                    name = filename.split('.')[0]
                    self.mask_images[name] = img
                    print(f"✅ Загружено изображение: {filename}")
        
        if not self.mask_images:
            self.create_default_images()
            for filename in image_files:
                filepath = os.path.join(mask_dir, filename)
                if os.path.exists(filepath):
                    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        name = filename.split('.')[0]
                        self.mask_images[name] = img
    
    def get_landmark_points(self, landmarks, frame_width, frame_height):
        """Преобразует landmarks в пиксельные координаты"""
        points = []
        for landmark in landmarks:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            points.append((x, y))
        return points
    
    def get_head_pose(self, points):
        """Вычисляет позу головы (углы поворота)"""
        if len(points) < len(self.model_points_indices):
            return None, None
        
        # Извлекаем 2D точки для PnP
        image_points = np.array([
            points[idx] for idx in self.model_points_indices
        ], dtype=np.float64)
        
        # Решаем PnP задачу
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points_3d,
            image_points,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        if success:
            return rotation_vector, translation_vector
        return None, None
    
    def create_3d_plane_points(self, center_3d, size, offset_3d=(0, 0, 0)):
        """Создает 3D точки плоскости"""
        width, height = size
        offset_x, offset_y, offset_z = offset_3d
        
        # Создаем 4 угла плоскости в 3D пространстве
        plane_points_3d = np.array([
            [center_3d[0] - width/2 + offset_x, center_3d[1] - height/2 + offset_y, center_3d[2] + offset_z],
            [center_3d[0] + width/2 + offset_x, center_3d[1] - height/2 + offset_y, center_3d[2] + offset_z],
            [center_3d[0] + width/2 + offset_x, center_3d[1] + height/2 + offset_y, center_3d[2] + offset_z],
            [center_3d[0] - width/2 + offset_x, center_3d[1] + height/2 + offset_y, center_3d[2] + offset_z]
        ], dtype=np.float64)
        
        return plane_points_3d
    
    def project_3d_to_2d(self, points_3d, rotation_vector, translation_vector):
        """Проецирует 3D точки в 2D экранные координаты"""
        if rotation_vector is None or translation_vector is None:
            return None
        
        projected_points, _ = cv2.projectPoints(
            points_3d,
            rotation_vector,
            translation_vector,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        return projected_points.reshape(-1, 2).astype(int)
    
    def warp_image_to_quad(self, image, quad_points, frame_shape):
        """Деформирует изображение в четырехугольник (перспективное преобразование)"""
        if quad_points is None or len(quad_points) != 4:
            return None
        
        # Проверяем, что точки находятся в пределах кадра
        frame_h, frame_w = frame_shape[:2]
        for point in quad_points:
            if point[0] < 0 or point[0] >= frame_w or point[1] < 0 or point[1] >= frame_h:
                return None
        
        # Исходные точки изображения (углы)
        h, w = image.shape[:2]
        src_points = np.array([
            [0, 0],
            [w-1, 0],
            [w-1, h-1],
            [0, h-1]
        ], dtype=np.float32)
        
        # Целевые точки (проецированная 3D плоскость)
        dst_points = quad_points.astype(np.float32)
        
        # Вычисляем матрицу перспективного преобразования
        try:
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Применяем преобразование
            warped = cv2.warpPerspective(image, matrix, (frame_w, frame_h), 
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_TRANSPARENT)
            return warped
        except:
            return None
    
    def overlay_warped_image(self, background, warped_image, quad_points):
        """Накладывает деформированное изображение на фон"""
        if warped_image is None or quad_points is None:
            return
        
        # Создаем маску из четырехугольника
        mask = np.zeros(background.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [quad_points], 255)
        
        # Применяем альфа-смешивание
        if warped_image.shape[2] == 4:
            alpha = warped_image[:, :, 3] / 255.0
            mask_normalized = mask / 255.0
            
            # Комбинируем альфа-канал с маской
            combined_alpha = alpha * mask_normalized
            
            for c in range(3):
                background[:, :, c] = (
                    combined_alpha * warped_image[:, :, c] +
                    (1 - combined_alpha) * background[:, :, c]
                )
    
    def apply_3d_mask(self, frame, points, preset_name, rotation_vector, translation_vector):
        """Применяет 3D маску"""
        if preset_name not in self.mask_presets:
            return
        
        preset = self.mask_presets[preset_name]
        image_name = preset['image'].split('.')[0]
        
        if image_name not in self.mask_images:
            return
        
        mask_image = self.mask_images[image_name]
        
        # Получаем центр привязки маски
        anchor_indices = preset['anchor_points']
        if not anchor_indices:
            return
        
        anchor_points = [points[i] for i in anchor_indices if i < len(points)]
        if not anchor_points:
            return
        
        # Создаем 3D плоскость
        center_3d = np.array([0, 0, 0], dtype=np.float64)  # Центр лица
        
        plane_size = (
            int(preset['plane_size'][0] * self.image_scale),
            int(preset['plane_size'][1] * self.image_scale)
        )
        
        plane_points_3d = self.create_3d_plane_points(
            center_3d, plane_size, preset['offset_3d']
        )
        
        # Проецируем 3D плоскость в 2D
        projected_quad = self.project_3d_to_2d(
            plane_points_3d, rotation_vector, translation_vector
        )
        
        if projected_quad is not None:
            # Деформируем изображение под 3D плоскость
            warped_image = self.warp_image_to_quad(
                mask_image, projected_quad, frame.shape
            )
            
            # Накладываем на кадр
            if warped_image is not None:
                self.overlay_warped_image(frame, warped_image, projected_quad)
            
            # Отладочная визуализация
            if self.show_3d_debug:
                cv2.polylines(frame, [projected_quad], True, (0, 255, 0), 2)
                for i, point in enumerate(projected_quad):
                    cv2.circle(frame, tuple(point), 5, (255, 0, 0), -1)
                    cv2.putText(frame, str(i), (point[0]+10, point[1]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_head_pose_axes(self, frame, rotation_vector, translation_vector):
        """Рисует оси координат для визуализации позы головы"""
        if rotation_vector is None or translation_vector is None:
            return
        
        # 3D точки осей
        axes_points_3d = np.array([
            [0, 0, 0],      # Центр
            [100, 0, 0],    # X ось (красная)
            [0, 100, 0],    # Y ось (зеленая)
            [0, 0, -100]    # Z ось (синяя)
        ], dtype=np.float64)
        
        # Проецируем в 2D
        projected_axes = self.project_3d_to_2d(
            axes_points_3d, rotation_vector, translation_vector
        )
        
        if projected_axes is not None:
            center = tuple(projected_axes[0])
            x_axis = tuple(projected_axes[1])
            y_axis = tuple(projected_axes[2])
            z_axis = tuple(projected_axes[3])
            
            # Рисуем оси
            cv2.line(frame, center, x_axis, (0, 0, 255), 3)  # X - красная
            cv2.line(frame, center, y_axis, (0, 255, 0), 3)  # Y - зеленая
            cv2.line(frame, center, z_axis, (255, 0, 0), 3)  # Z - синяя
    
    def draw_info_panel(self, frame):
        """Рисует информационную панель"""
        height, width = frame.shape[:2]
        
        # Фон панели
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Заголовок
        cv2.putText(frame, "Face 3D Mask", (20, 35), font, 0.7, (255, 255, 255), 2)
        
        # Текущая маска
        cv2.putText(frame, f"Mask: {self.current_mask}", (20, 60), font, 0.5, (0, 255, 255), 1)
        
        # Управление
        cv2.putText(frame, "Controls:", (20, 85), font, 0.5, (255, 255, 100), 1)
        cv2.putText(frame, "1: Glasses  2: Mustache  3: Crown  4: Hearts", (20, 105), font, 0.4, (150, 150, 150), 1)
        cv2.putText(frame, "+/-: Scale  L: Landmarks  D: 3D Debug  ESC: Exit", (20, 125), font, 0.4, (150, 150, 150), 1)
        
        # Статистика справа
        cv2.rectangle(overlay, (width-200, 10), (width-10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "Stats:", (width-190, 35), font, 0.5, (255, 255, 100), 1)
        cv2.putText(frame, f"Scale: {self.image_scale:.1f}", (width-190, 55), font, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"3D Debug: {'On' if self.show_3d_debug else 'Off'}", (width-190, 75), font, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Images: {len(self.mask_images)}", (width-190, 95), font, 0.4, (200, 200, 200), 1)
    
    def run(self):
        """Основной цикл программы"""
        print("🎭 Face 3D Mask запущен!")
        print("Управление:")
        print("- 1: Очки (3D)")
        print("- 2: Усы (3D)")
        print("- 3: Корона (3D)")
        print("- 4: Сердечки на глазах (3D)")
        print("- +/-: Масштаб изображения")
        print("- L: Показать ключевые точки")
        print("- D: Отладка 3D плоскостей")
        print("- ESC: Выход")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Ошибка захвата видео")
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.face_mesh.process(frame_rgb)
            
            height, width = frame.shape[:2]
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Получаем точки лица
                    points = self.get_landmark_points(face_landmarks.landmark, width, height)
                    
                    # Вычисляем позу головы
                    rotation_vector, translation_vector = self.get_head_pose(points)
                    
                    # Показываем ключевые точки если включено
                    if self.show_landmarks:
                        for idx in self.model_points_indices:
                            if idx < len(points):
                                cv2.circle(frame, points[idx], 3, (0, 255, 0), -1)
                    
                    # Рисуем оси координат для отладки
                    if self.show_3d_debug:
                        self.draw_head_pose_axes(frame, rotation_vector, translation_vector)
                    
                    # Применяем 3D маску
                    self.apply_3d_mask(frame, points, self.current_mask, 
                                     rotation_vector, translation_vector)
            
            # Рисуем информационную панель
            self.draw_info_panel(frame)
            
            cv2.imshow('Face 3D Mask', frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('1'):
                self.current_mask = 'glasses'
            elif key == ord('2'):
                self.current_mask = 'mustache'
            elif key == ord('3'):
                self.current_mask = 'crown'
            elif key == ord('4'):
                self.current_mask = 'heart_eyes'
            elif key == ord('+') or key == ord('='):
                self.image_scale = min(3.0, self.image_scale + 0.1)
            elif key == ord('-'):
                self.image_scale = max(0.3, self.image_scale - 0.1)
            elif key == ord('l') or key == ord('L'):
                self.show_landmarks = not self.show_landmarks
            elif key == ord('d') or key == ord('D'):
                self.show_3d_debug = not self.show_3d_debug
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Face 3D Mask завершен")

if __name__ == "__main__":
    mask = Face3DMask()
    mask.run() 