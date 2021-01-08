import cv2

cascPath = "haarcascade_frontalface_default.xml"
imPath = "../images/people.jpg"

# Load the cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Load the image and convert to grayscale
image = cv2.imread(imPath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor = 1.1,
    minNeighbors = 5,
    minSize = (20, 20)
)

print(f"Found {len(faces)} faces.")
# Draw rectangles around the faces on the image
for x, y, w, h in faces:
	cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)

# Draw the image with face boxes
cv2.imshow("Faces found", image)
cv2.waitKey(0)
