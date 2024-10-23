import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

class FaceMeshDetector:
    def __init__(self, static_mode=False, max_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            refine_landmarks=True
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)

        # Store MediaPipe's topology information
        self.FACE_CONNECTIONS = self.mp_face_mesh.FACEMESH_TESSELATION

    def detect_faces(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        faces = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=self.FACE_CONNECTIONS,
                        landmark_drawing_spec=self.drawing_spec,
                        connection_drawing_spec=self.drawing_spec
                    )
                
                face = []
                for lm in face_landmarks.landmark:
                    # Improve scaling and position
                    x = (lm.x - 0.5) * 1.5
                    y = -(lm.y - 0.5) * 1.8
                    z = -lm.z * 2.5
                    face.append([x, y, z])
                faces.append(face)
        
        return img, faces

    def create_mesh_topology(self, vertices):
        """
        Create mesh faces using MediaPipe's topology information.
        """
        faces = []
        edges = {}
        
        # Create edges from MediaPipe's connections
        for connection in self.FACE_CONNECTIONS:
            idx1, idx2 = connection
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            edge = (idx1, idx2)
            if edge not in edges:
                edges[edge] = []

        # Create triangles from connected edges
        for edge in edges:
            connected_vertices = []
            for other_edge in edges:
                if edge[0] in other_edge or edge[1] in other_edge:
                    v = other_edge[0] if other_edge[0] not in edge else other_edge[1]
                    connected_vertices.append(v)
            
            # Create triangles with connected vertices
            for v in connected_vertices:
                if v < len(vertices):
                    triangle = sorted([edge[0], edge[1], v])
                    if triangle not in faces:
                        faces.append(triangle)

        return np.array(faces)

    def save_obj_file(self, vertices, filename):
        """
        Save the 3D mesh as an OBJ file using MediaPipe's topology.
        """
        Path('output').mkdir(exist_ok=True)
        full_path = f"output/{filename}.obj"
        
        try:
            # Create faces using MediaPipe's topology
            faces = self.create_mesh_topology(vertices)
            
            with open(full_path, 'w') as f:
                # Write vertices
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
                # Write faces
                for face in faces:
                    # OBJ files are 1-indexed
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                
                # Write vertex normals
                f.write("# Vertex normals\n")
                for face in faces:
                    v1, v2, v3 = vertices[face]
                    normal = np.cross(v2 - v1, v3 - v1)
                    normal = normal / (np.linalg.norm(normal) + 1e-10)
                    f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
            
            print(f"Saved 3D model to {full_path}")
            return True
        except Exception as e:
            print(f"Error saving OBJ file: {e}")
            return False

def main():
    detector = FaceMeshDetector()
    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    model_saved = False
    
    print("Press 's' to capture and save 3D model")
    print("Press 'q' to quit")
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        img, faces = detector.detect_faces(img)
        
        y_pos = 30
        cv2.putText(img, f"FPS: {fps:.1f}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 30
        
        if model_saved:
            cv2.putText(img, "Model Saved!", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if time.time() - current_time > 2:
                model_saved = False
        else:
            cv2.putText(img, "Press 's' to save 3D model", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Face Mesh", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and faces and not model_saved:
            vertices = np.array(faces[0]) * 150
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            if detector.save_obj_file(vertices, f"face_mesh_{timestamp}"):
                model_saved = True
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()