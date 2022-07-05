from imutils import paths
import face_recognition
import pickle
import cv2
import os
 
#Include the path to the dataset at "image_path_here" (this is kind of a folder that contains multiple folders of different persons). <br>
imagePaths = list(paths.list_images("image_path_here"))
print(imagePaths)
knownEncodings = []
knownNames = []
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

data = {"encodings": knownEncodings, "names": knownNames}

f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()
