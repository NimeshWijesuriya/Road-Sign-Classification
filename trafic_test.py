import numpy as np
import cv2
import pickle

# Define parameters
framewidth = 200
frameHeight = 200
brightness = 180
threshold = 0.9
font = cv2.FONT_HERSHEY_SIMPLEX

# Set up video capture
cap = cv2.VideoCapture(0)
cap.set(3, framewidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Load the trained model using pickle
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

# Define image preprocessing functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0  # Normalize image to [0, 1]
    return img

# Define a function to get class name from class index
def getclassName(classNo):
    if classNo == 0: return 'Speed limit (20km/h)'
    elif classNo == 1: return 'Speed limit (30km/h)'
    elif classNo == 2: return 'Speed limit (50km/h)'
    elif classNo==3: return 'Speed limit (60km/h)'
    elif classNo==4: return 'Speed limit (70km/h)'
    elif classNo==5: return 'Speed limit (80km/h)'
    elif classNo==6: return 'End of speed limit (80km/h)'
    elif classNo==7: return 'speed limit (100km/h)'
    elif classNo==8: return 'Speed limit (120km/h)'
    elif classNo==9: return 'No passing'
    elif classNo==10: return 'No passing veh over 3.5 tons'
    elif classNo==11: return 'Right-of-way at intersection'
    elif classNo==12: return 'Priority road'
    elif classNo==13: return 'Yield'
    elif classNo==14: return 'Stop'
    elif classNo==15: return 'No vehicles'
    elif classNo==16: return 'Veh > 3.5 tons prohibited'
    elif classNo==17: return 'No entry'
    elif classNo==18: return 'General caution'
    elif classNo==19: return 'Dangerous curve left'
    elif classNo==20: return 'Dangerous curve right'
    elif classNo==21: return 'Double curve'
    elif classNo==22: return 'Bumpy road'
    elif classNo==23: return 'Slippery road'
    elif classNo==24: return 'Road narrows on the right'
    elif classNo==25: return 'Road work'
    elif classNo==26: return 'Traffic signals'
    elif classNo==27: return 'Pedestrians'
    elif classNo==28: return 'Children crossing'
    elif classNo==29: return 'Bicycles crossing'
    elif classNo==30: return 'Beware of ice/snow'
    elif classNo==31: return 'Wild animals crossing'
    elif classNo==32: return 'End speed + passing limits'
    elif classNo==33: return 'Turn right ahead'
    elif classNo==34: return 'Turn left ahead'
    elif classNo==35: return 'Ahead only'
    elif classNo==36: return 'Go straight or right'
    elif classNo==37: return 'Go straight or left'
    elif classNo==38: return 'Keep right'
    elif classNo==39: return 'Keep left'
    elif classNo==40: return 'Roundabout mandatory'
    elif classNo==41: return 'End of no passing'
    elif classNo==42: return 'End no passing veh > 3.5 tons'
   


while True:
    success, imgOriginal = cap.read()
    
    if not success:
        break

    # Preprocess the captured image
    img = cv2.resize(np.asarray(imgOriginal), (64, 64))
    img = preprocessing(img)
    img = img.reshape(64, 64)  # Reshape to 2D array (grayscale)

    # Convert image back to [0, 255] range for display
    img_display = (img * 255).astype(np.uint8)

    # Debugging: Check the shape and type of the image
    print(f"Processed Image Shape: {img_display.shape}, dtype: {img_display.dtype}")

    # Ensure the image has valid dimensions and type before displaying
    if img_display.ndim == 2 and img_display.dtype == np.uint8:
        cv2.imshow("Processed Image", img_display)  # Show the valid displayable format
    else:
        print("Image format is not valid for display.")

    # Make prediction
    img = img.reshape(1, 64, 64, 1)  # Reshape to the format expected by the model
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)  # Get the class index
    probabilityValue = np.amax(predictions)

    # Display class and probability if it meets the threshold
    if probabilityValue > threshold:
        cv2.putText(imgOriginal, f"Class: {getclassName(classIndex)}", (120, 135), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, f"Probability: {round(probabilityValue * 100, 2)}%", (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the original image with results
    cv2.imshow('Result', imgOriginal)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()