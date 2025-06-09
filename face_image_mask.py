import cv2
import mediapipe as mp
import numpy as np
import math
import os

class FaceImageMask:
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
        
        # Загружаем изображения масок
        self.mask_images = {}
        self.load_mask_images()
        
        # Ключевые точки для разных типов масок
        self.face_regions = {
            'eyes': {
                'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
                'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
                'center_points': [33, 133, 362, 263]  # углы глаз для позиционирования
            },
            'nose': {
                'tip': [1, 2],
                'bridge': [6, 8, 9, 10],
                'nostrils': [20, 94, 125, 141, 235, 236, 250]
            },
            'mouth': {
                'outer': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
                'inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
            },
            'forehead': [10, 151, 9, 8, 107, 55, 8, 9, 151, 337, 299, 333, 298, 301],
            'cheeks': {
                'left': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147],
                'right': [345, 346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 410, 454]
            }
        }
        
        # Предустановки масок
        self.mask_presets = {
            'glasses': {
                'region': 'eyes',
                'image': 'glasses.png',
                'scale': 1.5,
                'offset': (0, -10)
            },
            'mustache': {
                'region': 'nose',
                'image': 'mustache.png',
                'scale': 0.8,
                'offset': (0, 15)
            },
            'crown': {
                'region': 'forehead',
                'image': 'crown.png',
                'scale': 1.2,
                'offset': (0, -50)
            },
            'heart_eyes': {
                'region': 'eyes',
                'image': 'heart.png',
                'scale': 0.6,
                'offset': (0, 0)
            }
        }
    
    def create_default_images(self):
        """Создает простые изображения масок если файлы не найдены"""
        mask_dir = "mask_images"
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        
        # Создаем простые очки
        glasses = np.zeros((100, 200, 4), dtype=np.uint8)
        # Левая линза
        cv2.circle(glasses, (50, 50), 35, (0, 0, 0, 200), -1)
        cv2.circle(glasses, (50, 50), 30, (0, 0, 0, 0), -1)
        cv2.circle(glasses, (50, 50), 35, (255, 255, 255, 255), 3)
        # Правая линза
        cv2.circle(glasses, (150, 50), 35, (0, 0, 0, 200), -1)
        cv2.circle(glasses, (150, 50), 30, (0, 0, 0, 0), -1)
        cv2.circle(glasses, (150, 50), 35, (255, 255, 255, 255), 3)
        # Перемычка
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
        # Основание короны
        cv2.rectangle(crown, (10, 60), (140, 75), (255, 215, 0, 255), -1)
        # Зубцы короны
        for i in range(5):
            x = 20 + i * 25
            pts = np.array([[x, 60], [x+10, 20], [x+20, 60]], np.int32)
            cv2.fillPoly(crown, [pts], (255, 215, 0, 255))
        # Украшения
        for i in range(3):
            x = 35 + i * 40
            cv2.circle(crown, (x, 40), 5, (255, 0, 0, 255), -1)
        cv2.imwrite(f"{mask_dir}/crown.png", crown)
        
        # Создаем сердечки
        heart = np.zeros((60, 60, 4), dtype=np.uint8)
        # Простое сердце
        cv2.circle(heart, (20, 25), 15, (255, 0, 100, 255), -1)
        cv2.circle(heart, (40, 25), 15, (255, 0, 100, 255), -1)
        pts = np.array([[30, 35], [15, 50], [45, 50]], np.int32)
        cv2.fillPoly(heart, [pts], (255, 0, 100, 255))
        cv2.imwrite(f"{mask_dir}/heart.png", heart)
    
    def load_mask_images(self):
        """Загружает изображения масок"""
        mask_dir = "mask_images"
        
        # Создаем изображения если их нет
        if not os.path.exists(mask_dir):
            self.create_default_images()
        
        # Загружаем изображения
        image_files = ['glasses.png', 'mustache.png', 'crown.png', 'heart.png']
        
        for filename in image_files:
            filepath = os.path.join(mask_dir, filename)
            if os.path.exists(filepath):
                img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    # Если нет альфа-канала, добавляем его
                    if img.shape[2] == 3:
                        alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
                        img = np.concatenate([img, alpha], axis=2)
                    
                    name = filename.split('.')[0]
                    self.mask_images[name] = img
                    print(f"✅ Загружено изображение: {filename}")
                else:
                    print(f"❌ Ошибка загрузки: {filename}")
            else:
                print(f"❌ Файл не найден: {filepath}")
        
        if not self.mask_images:
            print("⚠️ Не удалось загрузить изображения масок")
            self.create_default_images()
            # Повторная попытка загрузки
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
    
    def get_region_bounds(self, points, region_indices):
        """Получает границы региона по индексам точек"""
        if not region_indices:
            return None
        
        region_points = [points[i] for i in region_indices if i < len(points)]
        if not region_points:
            return None
        
        xs = [p[0] for p in region_points]
        ys = [p[1] for p in region_points]
        
        return {
            'min_x': min(xs),
            'max_x': max(xs),
            'min_y': min(ys),
            'max_y': max(ys),
            'center_x': sum(xs) // len(xs),
            'center_y': sum(ys) // len(ys),
            'width': max(xs) - min(xs),
            'height': max(ys) - min(ys)
        }
    
    def overlay_image(self, background, overlay, x, y, scale=1.0):
        """Накладывает изображение с альфа-каналом на фон"""
        if overlay is None:
            return
        
        # Масштабируем изображение
        if scale != 1.0:
            new_width = int(overlay.shape[1] * scale)
            new_height = int(overlay.shape[0] * scale)
            overlay = cv2.resize(overlay, (new_width, new_height))
        
        h, w = overlay.shape[:2]
        
        # Центрируем изображение
        x = x - w // 2
        y = y - h // 2
        
        # Проверяем границы
        if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
            return
        
        # Извлекаем альфа-канал
        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3] / 255.0
            
            # Накладываем изображение
            for c in range(3):
                background[y:y+h, x:x+w, c] = (
                    alpha * overlay[:, :, c] + 
                    (1 - alpha) * background[y:y+h, x:x+w, c]
                )
    
    def apply_mask_preset(self, frame, points, preset_name):
        """Применяет предустановленную маску"""
        if preset_name not in self.mask_presets:
            return
        
        preset = self.mask_presets[preset_name]
        region = preset['region']
        image_name = preset['image'].split('.')[0]
        scale = preset['scale'] * self.image_scale
        offset = preset['offset']
        
        if image_name not in self.mask_images:
            return
        
        mask_image = self.mask_images[image_name]
        
        if region == 'eyes':
            # Для глаз используем центральные точки
            bounds = self.get_region_bounds(points, self.face_regions['eyes']['center_points'])
            if bounds:
                center_x = bounds['center_x'] + offset[0]
                center_y = bounds['center_y'] + offset[1]
                self.overlay_image(frame, mask_image, center_x, center_y, scale)
                
                # Для сердечек - отдельно на каждый глаз
                if preset_name == 'heart_eyes':
                    left_bounds = self.get_region_bounds(points, self.face_regions['eyes']['left_eye'])
                    right_bounds = self.get_region_bounds(points, self.face_regions['eyes']['right_eye'])
                    
                    if left_bounds:
                        self.overlay_image(frame, mask_image, 
                                         left_bounds['center_x'], left_bounds['center_y'], scale)
                    if right_bounds:
                        self.overlay_image(frame, mask_image, 
                                         right_bounds['center_x'], right_bounds['center_y'], scale)
        
        elif region == 'nose':
            bounds = self.get_region_bounds(points, self.face_regions['nose']['tip'])
            if bounds:
                center_x = bounds['center_x'] + offset[0]
                center_y = bounds['center_y'] + offset[1]
                self.overlay_image(frame, mask_image, center_x, center_y, scale)
        
        elif region == 'forehead':
            bounds = self.get_region_bounds(points, self.face_regions['forehead'])
            if bounds:
                center_x = bounds['center_x'] + offset[0]
                center_y = bounds['min_y'] + offset[1]
                self.overlay_image(frame, mask_image, center_x, center_y, scale)
    
    def draw_landmarks(self, frame, points, indices, color=(0, 255, 0), size=2):
        """Рисует ключевые точки для отладки"""
        for idx in indices:
            if idx < len(points):
                cv2.circle(frame, points[idx], size, color, -1)
    
    def draw_info_panel(self, frame):
        """Рисует информационную панель"""
        height, width = frame.shape[:2]
        
        # Фон панели
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Заголовок
        cv2.putText(frame, "Face Image Mask", (20, 35), font, 0.7, (255, 255, 255), 2)
        
        # Текущая маска
        cv2.putText(frame, f"Mask: {self.current_mask}", (20, 60), font, 0.5, (0, 255, 255), 1)
        
        # Управление
        cv2.putText(frame, "Controls:", (20, 85), font, 0.5, (255, 255, 100), 1)
        cv2.putText(frame, "1: Glasses  2: Mustache  3: Crown  4: Hearts", (20, 105), font, 0.4, (150, 150, 150), 1)
        cv2.putText(frame, "+/-: Scale  L: Toggle landmarks  ESC: Exit", (20, 125), font, 0.4, (150, 150, 150), 1)
        
        # Статистика справа
        cv2.rectangle(overlay, (width-200, 10), (width-10, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "Stats:", (width-190, 35), font, 0.5, (255, 255, 100), 1)
        cv2.putText(frame, f"Scale: {self.image_scale:.1f}", (width-190, 55), font, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Images: {len(self.mask_images)}", (width-190, 75), font, 0.4, (200, 200, 200), 1)
    
    def run(self):
        """Основной цикл программы"""
        print("🖼️ Face Image Mask запущен!")
        print("Управление:")
        print("- 1: Очки")
        print("- 2: Усы") 
        print("- 3: Корона")
        print("- 4: Сердечки на глазах")
        print("- +/-: Масштаб изображения")
        print("- L: Показать ключевые точки")
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
                    
                    # Показываем ключевые точки если включено
                    if self.show_landmarks:
                        if self.current_mask == 'glasses':
                            self.draw_landmarks(frame, points, self.face_regions['eyes']['center_points'], (0, 255, 0))
                        elif self.current_mask == 'mustache':
                            self.draw_landmarks(frame, points, self.face_regions['nose']['tip'], (255, 0, 0))
                        elif self.current_mask == 'crown':
                            self.draw_landmarks(frame, points, self.face_regions['forehead'], (255, 255, 0))
                        elif self.current_mask == 'heart_eyes':
                            self.draw_landmarks(frame, points, self.face_regions['eyes']['left_eye'], (255, 0, 255))
                            self.draw_landmarks(frame, points, self.face_regions['eyes']['right_eye'], (255, 0, 255))
                    
                    # Применяем маску
                    self.apply_mask_preset(frame, points, self.current_mask)
            
            # Рисуем информационную панель
            self.draw_info_panel(frame)
            
            cv2.imshow('Face Image Mask', frame)
            
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
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Face Image Mask завершен")

if __name__ == "__main__":
    mask = FaceImageMask()
    mask.run() 