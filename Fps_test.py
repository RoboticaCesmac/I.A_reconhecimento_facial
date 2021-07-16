import cv2
import time


cam = cv2.VideoCapture(0)


start_time = time.time()
# FPS update time in seconds
display_time = 2
fc = 0
FPS = 0


while True:

    # ler imagem
    ret, frame = cam.read()
    fc+=1
    TIME = time.time() - start_time
    
    if (TIME) >= display_time :
	    FPS = fc / (TIME)
	    fc = 0
	    start_time = time.time()
    fps_disp = "FPS: "+str(FPS)[:5]
    # Add FPS count on frame
    frame = cv2.putText(frame, fps_disp, (10, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # exibe a imagem
    cv2.imshow("RECONHECIMENTO FACIAL", frame)

    # aguarda alguma tecla
    key = cv2.waitKey(1)

    # para o sistema qando pressiona "q"
    if key == ord('q'):
        break
    if key == ord('Q'):
        break


print (frame_count)
print (len(all_frames))
# libera a camera ou vídeo
cam.release()

# destrói as janelas do opencv
cv2.destroyAllWindows()
