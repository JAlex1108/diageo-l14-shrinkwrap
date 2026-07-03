import cv2

video_path = r"videos_for_processing\your_video.ts"
cap = cv2.VideoCapture(video_path)
ok, frame = cap.read()
cap.release()

if not ok:
    raise ValueError("Could not read frame")

roi = cv2.selectROI("Select Conveyor ROI", frame, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

print("ROI =", tuple(int(v) for v in roi))