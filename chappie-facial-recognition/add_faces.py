import cv2
import argparse
import boto3
import time
import os

def capture_and_index_face(collection, name):
    # Initialize OpenCV VideoCapture for webcam
    cap = cv2.VideoCapture(0)  # Use the first webcam (index 0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        exit()

    # Capture image from webcam
    ret, frame = cap.read()

    # Generate image file path
    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, 'faces')
    os.makedirs(directory, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    image_path = '{0}/image_{1}.jpg'.format(directory, timestr)

    # Save captured image to file
    cv2.imwrite(image_path, frame)

    # Release the webcam
    cap.release()

    print('Your image was saved to:', image_path)

    # Initialize Amazon Rekognition client
    client = boto3.client('rekognition')

    # Index faces in the captured image
    with open(image_path, 'rb') as file:
        response = client.index_faces(
            Image={'Bytes': file.read()},
            CollectionId=collection,
            ExternalImageId=name,
            DetectionAttributes=['ALL']
        )

    print(response)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Capture image and add to collection.')
parser.add_argument('--collection', help='Collection Name', default='my-face-collection')
parser.add_argument('--name', help='Face Name')
args = parser.parse_args()

# Call the function to capture and index face
capture_and_index_face(args.collection, args.name)
