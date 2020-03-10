from imageai.Detection import ObjectDetection
import os
import cv2

line = "------------------------------"

#Default values for settings
#TODO: Allow user to specify values via cli args
#TODO: Make use of pythons logging library
saveRecording = True
fps = 10
true_logging = False #used in the detectCustomObjectsFromViceo function
logging = True #for frame custom logging
confidence = 30 #30% confidence required to detect
target_size = 80 #Size of a length of targeting rectangle

expath = os.getcwd() #Current execution directory

camera = cv2.VideoCapture(0) #Camera is default camera ID: 0

cam_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH) #float
cam_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT) #float
cam_center = (cam_width / 2, cam_height / 2)
print("%s\nCAMERA DIMENSIONS DETECTED AS: %s,%s" % (line, cam_width, cam_height))
print("CAMERA CENTER CALCULATED AT: %s\n%s" % (cam_center, line))

#Set up VideoWriter object for saving recording of video stream
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outputStream = cv2.VideoWriter("output.avi", fourcc, 20.0, (int(cam_width), int(cam_height)))

def drawRect(frame, preyList):

	thickness = 5 #in pixels
	retVal = False
	targetColor = (0, 0, 255)
	targetStatus = 'Hunting'

	target_start = (int(cam_center[0] - (target_size / 2)), int(cam_center[1] - (target_size / 2)))
	target_end = (int(cam_center[0] + (target_size / 2)), int(cam_center[1] + (target_size / 2)))
	textCoords = (target_start[0], target_start[1] - 10)

	#Calculate if the boxpoints of a target is within target box
	for prey in preyList:
		prey_start = (prey["box_points"][0], prey["box_points"][1])
		prey_end = (prey["box_points"][2], prey["box_points"][3])

		withinTarget = (prey_start[0] < target_end[0] and prey_end[0] > target_start[0]) #within x coords
		withinTarget = withinTarget and (prey_start[1] < target_end[1] and prey_end[1] > target_start[1]) #within y coords
		if withinTarget:
			targetStatus = 'Kill?'
			targetColor = (0,255,0) #green
			retVal = True
			break #else: drive?

	cv2.putText(frame, targetStatus, textCoords, cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0,0,255), thickness=3, lineType=8)
	cv2.rectangle(frame, target_start, target_end, targetColor, thickness)

	return retVal

#Instantiate detector object. Use YOLOv3 model type
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(expath, "yolo.h5"))
detector.loadModel(detection_speed="flash")

#Only detect 'person' objects, used below
custom_objects = detector.CustomObjects(person=True)

#Begin video
while(True):
		#Capture frame by frame
		ret, frame = camera.read() #type of numpy array


		#Run detections
		#Save scanned image to frame
		#Save information about detected objects to targetArray
		frame, targetArray = detector.detectCustomObjectsFromImage(
			custom_objects = custom_objects,
			input_type = "array",
			output_type = "array",
			input_image=frame,
			minimum_percentage_probability=confidence
			)

		#Draw target square. Color depends on if a target is within targetting rect
		onTarget = drawRect(frame, targetArray)

		#Display and save frame to output video
		cv2.imshow("deathCar v1.0b", frame)

		#Output frame to outputStream
		if saveRecording:
			outputStream.write(frame)

		#Check if the user has pressed the 'q' key to stop capture
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

camera.release()
outputStream.release()
cv2.destroyAllWindows()
