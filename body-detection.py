import cv2
from mediapipe.python.solutions import drawing_utils as mp_drawing, pose as mp_pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
	while cap.isOpened():
		ret, img = cap.read()

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img.flags.writeable = False

		# Body Detection
		results = pose.process(img)

		img.flags.writeable = True
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

		mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

		cv2.imshow('raw cam', img)
		if cv2.waitKey(1) & 0xFF == ord('e'):
			break

cap.release()
cv2.destroyAllWindows()
