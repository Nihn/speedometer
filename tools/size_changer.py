import cv2

file_name = '../test1'
cap = cv2.VideoCapture(file_name)

ret, frame = cap.read()
frame = frame[120:, 110:-10]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(file_name + '_out.avi', fourcc, 25, (frame.shape[1], frame.shape[0]))

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        frame = frame[120:, 110:-10]

        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
