from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
#changed .jpg to .jpeg
#detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpeg"), output_image_path=os.path.join(execution_path , "imagenew.jpeg"))

#output type = "array"
#returned_image, detections = detector.detectObjectsFromImage(input_image="image.jpeg", output_type="array")
#print(type(returned_image))

#print(returned_image)  --> returned numpy array

#extracted object and output type image
detections, extracted_objects = detector.detectObjectsFromImage(input_image="image.jpeg", output_image_path="imagenew.jpeg", extract_detected_objects=True)

#extracted object and output type array
#returned_image, detections, extracted_objects = detector.detectObjectsFromImage(input_image="image.jpeg", output_type="array", extract_detected_objects=True)

#print(type(extracted_objects)) --> returns <class 'list'>

#print(extracted_objects)

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )