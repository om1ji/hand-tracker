import numpy as np

class OBJLoader:
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.normals = []
        self.texture_coords = []
    
    def load_obj(self, filepath):
        """Загружает OBJ файл и возвращает вершины и грани"""
        self.vertices = []
        self.faces = []
        self.normals = []
        
        try:
            with open(filepath, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if not parts:
                        continue
                    
                    # Вершины (v x y z)
                    if parts[0] == 'v':
                        vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                        self.vertices.append(vertex)
                    
                    # Нормали (vn x y z)
                    elif parts[0] == 'vn':
                        normal = [float(parts[1]), float(parts[2]), float(parts[3])]
                        self.normals.append(normal)
                    
                    # Грани (f v1 v2 v3 или f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3)
                    elif parts[0] == 'f':
                        face = []
                        for vertex_data in parts[1:]:
                            # Парсим индексы вершин (могут быть в формате v, v/vt, v/vt/vn)
                            vertex_indices = vertex_data.split('/')
                            vertex_index = int(vertex_indices[0]) - 1  # OBJ индексы начинаются с 1
                            face.append(vertex_index)
                        self.faces.append(face)
            
            print(f"✅ Загружена модель: {len(self.vertices)} вершин, {len(self.faces)} граней")
            return True
            
        except FileNotFoundError:
            print(f"❌ Файл {filepath} не найден")
            return False
        except Exception as e:
            print(f"❌ Ошибка загрузки OBJ: {e}")
            return False
    
    def get_vertices(self):
        """Возвращает список вершин"""
        return self.vertices
    
    def get_faces(self):
        """Возвращает список граней"""
        return self.faces
    
    def get_bounds(self):
        """Возвращает границы модели для нормализации размера"""
        if not self.vertices:
            return None
        
        vertices_array = np.array(self.vertices)
        min_bounds = np.min(vertices_array, axis=0)
        max_bounds = np.max(vertices_array, axis=0)
        center = (min_bounds + max_bounds) / 2
        size = np.max(max_bounds - min_bounds)
        
        return {
            'min': min_bounds,
            'max': max_bounds,
            'center': center,
            'size': size
        }
    
    def normalize_model(self, target_size=1.0):
        """Нормализует модель к заданному размеру"""
        if not self.vertices:
            return
        
        bounds = self.get_bounds()
        if bounds is None:
            return
        
        # Центрируем модель
        center = bounds['center']
        scale = target_size / bounds['size']
        
        # Применяем трансформацию
        for i, vertex in enumerate(self.vertices):
            # Центрируем и масштабируем
            self.vertices[i] = [
                (vertex[0] - center[0]) * scale,
                (vertex[1] - center[1]) * scale,
                (vertex[2] - center[2]) * scale
            ]
    
    def triangulate_faces(self):
        """Преобразует все грани в треугольники"""
        triangulated_faces = []
        
        for face in self.faces:
            if len(face) == 3:
                # Уже треугольник
                triangulated_faces.append(face)
            elif len(face) == 4:
                # Четырехугольник -> два треугольника
                triangulated_faces.append([face[0], face[1], face[2]])
                triangulated_faces.append([face[0], face[2], face[3]])
            elif len(face) > 4:
                # Полигон -> веер треугольников
                for i in range(1, len(face) - 1):
                    triangulated_faces.append([face[0], face[i], face[i + 1]])
        
        self.faces = triangulated_faces

# Пример использования и тестирования
if __name__ == "__main__":
    loader = OBJLoader()
    
    # Создаем простой тестовый OBJ файл
    test_obj_content = """# Simple cube
v -1.0 -1.0 -1.0
v  1.0 -1.0 -1.0
v  1.0  1.0 -1.0
v -1.0  1.0 -1.0
v -1.0 -1.0  1.0
v  1.0 -1.0  1.0
v  1.0  1.0  1.0
v -1.0  1.0  1.0

f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 3 7 8 4
f 1 4 8 5
f 2 6 7 3
"""
    
    # Сохраняем тестовый файл
    with open("test_cube.obj", "w") as f:
        f.write(test_obj_content)
    
    # Тестируем загрузку
    if loader.load_obj("test_cube.obj"):
        print("Вершины:", len(loader.get_vertices()))
        print("Грани:", len(loader.get_faces()))
        print("Границы:", loader.get_bounds())
        
        loader.normalize_model(100)  # Нормализуем к размеру 100
        loader.triangulate_faces()   # Преобразуем в треугольники
        
        print("После нормализации:")
        print("Границы:", loader.get_bounds()) 