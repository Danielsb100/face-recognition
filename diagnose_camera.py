import cv2

def test_cameras(max_tested=5):
    print("Testing cameras... Please wait.")
    available_cameras = []
    for i in range(max_tested):
        # Try different backends
        for backend in [None, cv2.CAP_DSHOW, cv2.CAP_MSMF]:
            if backend is None:
                cap = cv2.VideoCapture(i)
            else:
                cap = cv2.VideoCapture(i, backend)
                
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    backend_name = "Default" if backend is None else ("DSHOW" if backend == cv2.CAP_DSHOW else "MSMF")
                    print(f"Index {i} works with backend {backend_name}")
                    available_cameras.append((i, backend))
                cap.release()
    
    if not available_cameras:
        print("No working cameras found.")
    return available_cameras

if __name__ == "__main__":
    test_cameras()
