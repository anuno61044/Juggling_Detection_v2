import cv2
import numpy as np
import mediapipe as mp
import math

cap = cv2.VideoCapture("2.mp4")
beats = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1280, 720))

# under_blue = np.array([100, 100, 23], np.uint8)
# lower_blue = np.array([125, 255, 255], np.uint8)

# under_red = np.array([130,80,30], np.uint8)
# lower_red = np.array([190,255,255], np.uint8)


# upper_yellow = np.array([68, 180, 200], np.uint8)
# lower_yellow = np.array([55, 180, 180], np.uint8)

# lower_green = np.array([60, 100, 50], np.uint8)
# upper_green = np.array([70, 200, 158], np.uint8)


lower_blue = np.array([130 - 60, 200 - 60, 170 - 60], np.uint8)
upper_blue = np.array([130 + 40, 200 + 40, 170 + 40], np.uint8)

lower_green = np.array([40, 100, 149], np.uint8)
upper_green = np.array([70, 255, 255], np.uint8)

lower_purple = np.array([210 - 80, 200 - 80, 200 - 80], np.uint8)
upper_purple = np.array([210 + 40, 200 + 40, 200 + 40], np.uint8)

last_landmark_type = 0

def detect_centers(mask):
    # hallar contornos y quitarnos los pequenos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 600:  # Umbral para filtrar contornos pequeños
            # punto central y area del objeto
            M = cv2.moments(c)
            if M["m00"] == 0 : M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            
            
            centers.append(((x, y), c))
            centers.append(((x,y), cv2.convexHull(c)))
    
    return centers
    
def drawContours(frame, centers, color):
    
    for (x,y), c in centers:
            
            # dibujar el centro
            cv2.circle(frame, (x, y), 7, color, -1)
            fort = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, "{}, {}".format(x, y), (x+10, y), fort, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
            
            new_contour = cv2.convexHull(c)
            cv2.drawContours(frame, [new_contour], -1, (0, 255, 0), thickness=2)
    return centers


def check_catch(hands, ball, balls, catch):
    global beats
    global last_landmark_type
    cota = 70
    number = []
    center = ball.get_current_position()
    
    for hand in hands:
        vector = (center[0]-hand[0],center[1]-hand[1])
        distance = math.sqrt(vector[0]**2 + vector[1]**2)
        if distance < cota: # pelota en la mano
            if ball.isCatch == -1:
                beats += 1
                ball.isCatch = hand[2]
                return len(number) > 0, number
            return False, []

    if ball.isCatch != -1: # pelota lanzada
        # verificar que los lanzamientos sean alternos
        # toma en cuenta el 2 en el siteswap
        for other_ball in balls:
            if other_ball.isCatch != -1 and other_ball.isCatch != ball.isCatch and last_landmark_type == ball.isCatch:
                number.append(beats - other_ball.beats_behide)
                print("o",beats , other_ball.beats_behide, other_ball.color)
                other_ball.beats_behide = beats
                beats += 1
                break

        last_landmark_type = ball.isCatch
        ball.isCatch = -1
        number.append(beats - ball.beats_behide)
        print("b",beats , ball.beats_behide, ball.color)
        ball.beats_behide = beats

    # print(number)
    return len(number) > 0, number
       
       
class Ball:
    def __init__(self, id, initial_position, color):
        self.id = id
        self.positions = [initial_position]  # Historial de posiciones como una lista de tuplas (x, y)
        self.color = color
        self.isCatch = -1
        self.beats_behide = 0 # pelotas cacheadas antes de esta

    def update_position(self, new_position):
        """Agrega la nueva posición al historial y actualiza la posición actual."""
        
        self.positions.append(new_position)
        # if len(self.positions) > 5:
        #     self.positions.pop(0)
    
    def get_current_position(self):
        """Devuelve la última posición registrada."""
            
        return self.positions[-1]

    def get_direction(self):
        """Calcula y devuelve la velocidad promedio entre la última y penúltima posición."""
        
        if len(self.positions) < 2:
            return (0, 0)  # Evita division por cero
        else:
            dx = self.positions[-1][0] - self.positions[-2][0]
            dy = self.positions[-1][1] - self.positions[-2][1]
            return (dx, dy)
               
class BallTracker:
    def __init__(self):
        self.balls = {}  # Ahora, un diccionario de {id: Ball}
     

    def update(self, centers, hands, color):
        """Actualiza las posiciones de las pelotas rastreadas y asigna nuevos IDs si es necesario."""
        
        balls_detected = None
        for c in centers:
            balls_detected = c[0]
            
        thrown_balls = []
        catched_balls = []
        cota = 70
        for ball in self.balls.values():
            catched = False
            center = ball.get_current_position()
            for hand in hands:
                vector = (center[0]-hand[0],center[1]-hand[1])
                distance = math.sqrt(vector[0]**2 + vector[1]**2)
                if distance < cota:
                    catched = True
                    break
            if catched:
                catched_balls.append(ball)
            else:
                thrown_balls.append(ball)
            
        for ball in catched_balls:
            if (ball.color == color and balls_detected != None):
                ball.positions.append(balls_detected)
                return check_catch(hands, ball, self.balls.values(), True)
            
        if (balls_detected == None):
            return False, []
        new_id = max(self.balls.keys(), default=0) + 1
        new_ball = Ball(new_id, balls_detected, color)
        self.balls[new_id] = new_ball
        return False, []
        

def find_siteswap_patterns(secuence, size):
    if len(secuence) < 2 * size:
        return None
    secuence.reverse()

    if secuence[0:size] == secuence[size:2*size]:
        return secuence[0:size]
    else:
        return None

def checkSecuence(sequence):
    lenght = len(sequence)

    # for i in range(2, 6):
    #     a = find_siteswap_patterns(secuence, i)
    #     if (a != None):
    #         return a
    
    if lenght > 10:
        print(1)
        secuence.pop(0)
    return secuence 
    
  
        

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

ballTracker = BallTracker()
ballTracker = BallTracker()

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1) as holistic:
    
    pattern_actual = []
    pattern_temp = []
    secuence = []

    while True:
        ret, frame = cap.read()
        
        height,width = frame.shape[:2]
        
        if ret == False:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results = holistic.process(frame_rgb)

        
        # Postura
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
        
        
        
        if ret:
            # convertir a hvs para buscar por color
            frameHVS = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # mask = cv2.inRange(frameHVS, under_red, lower_red)
            # maskBlue = cv2.inRange(frameHVS, under_blue, lower_blue)
            # maskYellow = cv2.inRange(frameHVS, lower_yellow, upper_yellow)
            # maskGreen = cv2.inRange(frameHVS, lower_green, upper_green)

            mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)
            mask_green = cv2.inRange(hsv_frame, lower_green, upper_green)
            mask_purple = cv2.inRange(hsv_frame, lower_purple, upper_purple)
            
            
            # centers = detect_centers(mask)
            # centersBlue = detect_centers(maskBlue)
            # centersYellow = detect_centers(maskYellow)
            # centersGreen = detect_centers(maskGreen)
            
            centersBlue = detect_centers(mask_blue)
            centersPurple= detect_centers(mask_purple)
            centersGreen = detect_centers(mask_green)
            
            hands_positions = []
            
            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks.landmark
                for landmark_type, lm in enumerate(pose_landmarks):
                    if landmark_type == 18 or landmark_type == 17:
                        hands_positions.append((lm.x*width, lm.y*height, landmark_type))
                


            # detectar donde estan actualmente las pelotas
            # catchRed, countRed = ballTracker.update(centers, hands_positions, "red")
            # catchBlue, countBlue = ballTracker.update(centersBlue, hands_positions, "blue")
            # catchYellow, countYellow = ballTracker.update(centersYellow, hands_positions, "yellow")
            # catchGreen, countGreen = ballTracker.update(centersGreen, hands_positions, "green")

            catchBlue, countBlue = ballTracker.update(centersBlue, hands_positions, "blue")
            catchPurple, countPurple = ballTracker.update(centersPurple, hands_positions, "purple")
            catchGreen, countGreen = ballTracker.update(centersGreen, hands_positions, "green")



            # dibujar
            # drawContours(frame, centers, (0, 0, 255))
            drawContours(frame, centersPurple, (238, 64, 194))
            drawContours(frame, centersBlue, (255, 0, 0))
            # drawContours(frame, centersYellow, (255, 250, 0))
            drawContours(frame, centersGreen,   (0, 255, 0))
            
            # revisar y contar las atrapadas de pelota
            if catchBlue == True:
                secuence.append(countBlue)
            # if catchRed == True:
            #     secuence.append(countRed)
            # if catchYellow == True:
            #     secuence.append(countYellow)
            if catchPurple == True:
                secuence.append(countPurple)
            if catchGreen == True:
                secuence.append(countGreen)
            result = ' '.join(str(item) for sublist in checkSecuence(secuence) for item in sublist)

            fort = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, "{},{},{}".format(True, beats, result), (100, 100), fort, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, (100,100), 40, (60, 200, 220), 5)
            
            # under_yellow = np.array([40, 200, 200], np.uint8)
            # lower_yellow = np.array([10, 100, 50], np.uint8)

            # lower_green = np.array([30, 50, 40], np.uint8)
            # under_green = np.array([60, 200, 220], np.uint8)
            
            # cv2.imshow("mask", mask)
            out.write(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                frame1 = frame
                break

cap.release()
cv2.destroyAllWindows()
