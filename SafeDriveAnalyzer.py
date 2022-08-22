from imutils import face_utils
from cv2 import VideoCapture, cvtColor, rectangle, putText, FONT_HERSHEY_SIMPLEX, circle, imshow,waitKey, COLOR_BGR2GRAY, destroyAllWindows
import numpy as np
import dlib
from winsound import Beep as bp

running = True
class safeDriveAnalyzer:
	def __init__(self):
		self.frequency = 2500  # Set Frequency To 2500 Hertz
		self.duration = 1000  # Set Duration To 1000 ms == 1 second

		self.cap = VideoCapture(0)

		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

		self.sleep = 0
		self.drowsy = 0
		self.active = 0
		self.status = ""
		self.color = (0, 0, 0)
		self.running = True
		self.run()

	def distance_evaluation(self, point_A, point_B):
		euclidean_dist = np.linalg.norm(point_A - point_B)
		return euclidean_dist

	def blinked(self, a, b, c, d, e, f):
		longitudinal = self.distance_evaluation(b, d) + self.distance_evaluation(c, e)
		lateral = self.distance_evaluation(a, f)
		ratio = longitudinal / (2.0 * lateral)
		# Checking if it is blinked
		if (ratio > 0.25):
			return 2
		elif (ratio > 0.21 and ratio <= 0.25):
			return 1
		else:
			return 0
	def start(self):
		global running
		running = True
	def stop(self):
		global running
		running = False
	def run(self):
		while True:
			_, frame = self.cap.read()
			gray = cvtColor(frame, COLOR_BGR2GRAY)
			faces = self.detector(gray)
			# detected face in faces array
			for face in faces:
				x1 = face.left()
				y1 = face.top()
				x2 = face.right()
				y2 = face.bottom()

				face_frame = frame.copy()
				rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

				landmarks = self.predictor(gray, face)
				landmarks = face_utils.shape_to_np(landmarks)

				# The numbers are actually the landmarks which will show eye
				left_blink = self.blinked(landmarks[36], landmarks[37],
										  landmarks[38], landmarks[41], landmarks[40], landmarks[39])
				right_blink = self.blinked(landmarks[42], landmarks[43],
										   landmarks[44], landmarks[47], landmarks[46], landmarks[45])

				# Now judge what to do for the eye blinks
				if (left_blink == 0 or right_blink == 0):
					self.sleep += 1
					self.drowsy = 0
					self.active = 0
					if (self.sleep > 6):
						self.status = "Sleeping"
						self.color = (26, 101, 158)
						bp(self.frequency, self.duration)


				elif (left_blink == 1 or right_blink == 1):
					self.sleep = 0
					self.active = 0
					self.drowsy += 1
					if (self.drowsy > 6):
						self.status = "Drowsy"
						self.color = (143, 184, 222)
						bp(self.frequency, self.duration)

				else:
					self.drowsy = 0
					self.sleep = 0
					self.active += 1
					if (self.active > 6):
						self.status = "Active"
						self.color = (205, 247, 246)

				putText(frame, self.status, (100, 100), FONT_HERSHEY_SIMPLEX, 1.2, self.color, 3)

				for n in range(0, 68):
					(x, y) = landmarks[n]
					circle(face_frame, (x, y), 1, (255, 255, 255), -1)

			imshow("Frame", frame)
			# imshow("Result of detector", face_frame)
			waitKey(1)
		destroyAllWindows()

safeDriveAnalyzer()


