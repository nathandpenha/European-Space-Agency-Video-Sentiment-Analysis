## Requirement/dependency
```python3```\
 ```Opencv``` 
 
## Running face detector library on Raspberry and Local
In ```face_extractor.py``` we use ```cv2.CascadeClassifier``` to detect face in given frames. In raspberry we need to manually put the ```haarcascade_....xml``` files and change the code to be able to load them.
Here you can see the code to be able to read them in local machine:
  
  
`
 self.__face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
 self.__face_cascade1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
 self.__face_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')
 self.__face_cascade3 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
 self.__face_cascade4 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
`


To load these libraries in Raspberry after puting the ```xml``` files in a directory we need to change the code in  ```face_extractor.py``` to be able to load them.
Here you can see the code to be able to load them in raspberry:


`self.__face_cascade = cv2.CascadeClassifier(
            '/home/pi/esaProject/workspace/ir_model/src/preprocessing/haarcascade_frontalface_alt.xml')
        self.__face_cascade1 = cv2.CascadeClassifier(
            '/home/pi/esaProject/workspace/ir_model/src/preprocessing/haarcascade_frontalface_alt2.xml')
        self.__face_cascade2 = cv2.CascadeClassifier(
            '/home/pi/esaProject/workspace/ir_model/src/preprocessing/haarcascade_frontalface_alt_tree.xml')
        self.__face_cascade3 = cv2.CascadeClassifier(
            '/home/pi/esaProject/workspace/ir_model/src/preprocessing/haarcascade_profileface.xml')
        self.__face_cascade4 = cv2.CascadeClassifier(
            '/home/pi/esaProject/workspace/ir_model/src/preprocessing/haarcascade_frontalface_default.xml')` 
   

  
 