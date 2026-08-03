import cv2
import os

cam = cv2.VideoCapture(0)

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

student_id = input("Enter your Student ID: ")

if not os.path.exists("Dataset"):
    os.makedirs("Dataset")

count = 0 

student_folder = os.path.join("Dataset", student_id)
os.makedirs(student_folder, exist_ok=True)

for file in os.listdir(student_folder):
    os.remove(os.path.join(student_folder, file))

while True:
    ret,img = cam.read()
    if not ret:
        print("[Error] Failed to grab frame from camera.")
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces :
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        count += 1
        
        offset = 20
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        
        face_crop = img[y1:y2, x1:x2]
        
        cv2.imwrite(os.path.join(student_folder,f"{count}.jpg"),
                    gray[y:y+h, x:x+w]
                    )
        
    cv2.imshow("Image", img)
    
    k = cv2.waitKey(50) & 0xff
    
    if k == 27:
        break
    elif count >= 30:
        break
    
print("\n[INFO] Face capture complete.")

cam.release()
cv2.destroyAllWindows()