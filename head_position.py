import cv2
import mediapipe as mp
import numpy as np
import time



def face_scan(name):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


    mp_drawing = mp.solutions.drawing_utils
    drawing_spec=mp_drawing.DrawingSpec (thickness=1, circle_radius=1)

    cap = cv2.VideoCapture(0)

    key_of_forward,key_of_left,key_of_right,key_of_up,key_of_down=0,0,0,0,0

    countForward,countForwardSaved = 0,0
    countLeft, countLeftSaved = 0,0
    countRight, countRightSaved = 0,0
    countUp,countUpSaved = 0,0
    countDown,countDownSaved = 0,0

    while cap.isOpened ():
        success, image = cap.read()
        start = time.time()
        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        #Perform histogram equalization
        equalized = cv2.equalizeHist(gray)
    
        # Convert back to BGR color space
        equalized_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        
        image.flags.writeable = False
        # Get the result
        results = face_mesh.process (image)
        # To improve performance
        image.flags.writeable = True
        # Convert the color space from RGB to BGR
        image = cv2.cvtColor (image, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate (face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x  * img_w,img_w,  lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y* img_h, lm.z* 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        # Get the 2D Coordinates
                        face_2d.append([x, y])
                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)
                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np. float64)
                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],[0, focal_length, img_w / 2],[0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np. float64)
                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP (face_3d, face_2d, cam_matrix, dist_matrix)
                # Get rotational matrix
                rmat, jac = cv2.Rodrigues (rot_vec)
                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3 (rmat)
                # Get the y rotation degree
                x = angles [0] * 360
                y = angles [1] * 360
                z = angles [2] * 360
                # See where the user's head tilting
                if y < -10:
                    if key_of_forward == 1:
                        countLeft += 1
                        if countLeft % 3 == 0 and countLeftSaved < 25:
                            cv2.imwrite(f"./train_img/{name}/leftFace{countLeft // 3}.jpg", equalized_image)
                            countLeftSaved += 1
                            #print(f"writing left{countLeft // 3}")
                        elif countLeftSaved >=25:
                            key_of_left = 1
                    text = "Looking Left"
                elif y> 10:
                    if key_of_left == 1:
                        countRight += 1
                        if countRight % 3 == 0 and countRightSaved < 25:
                            cv2.imwrite(f"./train_img/{name}/rightFace{countRight // 3}.jpg", equalized_image)
                            countRightSaved += 1
                            #print(f"writing right{countRight // 3}")
                        elif countRightSaved >=25:
                            key_of_right = 1
                    text = "Looking Right"
                elif x < -10:
                    if key_of_up == 1:
                        countDown += 1
                        if countDown % 3 == 0 and countDownSaved < 25:
                            cv2.imwrite(f"./train_img/{name}/upFace{countDown // 3}.jpg", equalized_image)
                            countDownSaved += 1
                            #print(f"writing up{countDown // 3}")
                        elif countDownSaved >= 25:
                            key_of_down = 1
                    text = "Looking Down"
                elif x > 10:
                    if key_of_right == 1:
                        countUp += 1
                        if countUp % 3 == 0 and countUpSaved < 25:
                            cv2.imwrite(f"./train_img/{name}/downFace{countUp // 3}.jpg", equalized_image)
                            countUpSaved += 1
                            #print(f"writing down{countUp // 3}")
                        elif countUpSaved >= 25:
                            key_of_up = 1
                    text = "Looking Up"
                else:
                    countForward+=1
                    if countForward % 3 == 0 and countForwardSaved < 25:
                        cv2.imwrite(f"./train_img/{name}/forwardFace{countForward//3}.jpg",equalized_image)
                        countForwardSaved += 1
                        #print(f"writing forward{countForward//3}")
                    elif countForwardSaved >= 25:
                        key_of_forward = 1
                    text= "Forward"
                if key_of_forward!=1 and key_of_left!=1 and key_of_right!=1 and key_of_up!=1 and key_of_down!=1:
                    print("Look forward!")
                if key_of_forward==1 and key_of_left!=1 and key_of_right!=1 and key_of_up!=1 and key_of_down!=1:
                    print("Look left!")
                if key_of_left==1 and key_of_right!=1 and key_of_up!=1 and key_of_down!=1:
                    print("Look Right!")
                if key_of_right==1 and key_of_up!=1 and key_of_down!=1:
                    print("Look Up!")
                if key_of_up==1 and key_of_down!=1:
                    print("Look Down!")
                if key_of_down==1:
                    print("Scan Completed")
                nose_3d_projection, jacobian = cv2.projectPoints (nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int (nose_2d[0]), int(nose_2d[1]))
                p2 = (int (nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                cv2.line (image, p1, p2, (255, 0, 0), 3)
                # Add the text on the image
                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText (image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
                cv2.putText (image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
                cv2.putText (image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)

            end = time.time()
            totalTime = end - start

            if totalTime != 0:
                fps = 1 / totalTime
            else:
                fps = 0

            cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections = mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

        cv2.imshow('Head Pose Estimation', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if key_of_down == 1:
            break
    cap.release()

if __name__ == "__main__":
    face_scan(name)