import cv2
import numpy as np
import random
import mediapipe as mp

# Initialize MediaPipe Hands for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize score and question
score = 0
question, correct_answer = None, None

# Generate a random math question
def generate_question():
    num1 = random.randint(1, 10)
    num2 = random.randint(1, 10)
    correct = num1 + num2
    options = [correct, random.randint(1, 20), random.randint(1, 20), random.randint(1, 20)]
    random.shuffle(options)
    return f"{num1} + {num2}", correct, options

# Create the interface
def draw_interface(frame, question, options, hover_index):
    h, w, _ = frame.shape
    cv2.putText(frame, f"Question: {question}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw options
    circle_positions = []
    for i, option in enumerate(options):
        x = 150 + i * 150
        y = h // 2
        color = (0, 255, 0) if hover_index == i else (255, 0, 0)
        cv2.circle(frame, (x, y), 50, color, -1)
        cv2.putText(frame, str(option), (x - 15, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        circle_positions.append((x, y))
    return circle_positions

# Check if hand is hovering over a circle
def detect_hover(hand_x, hand_y, circle_positions):
    for i, (cx, cy) in enumerate(circle_positions):
        distance = np.sqrt((hand_x - cx) ** 2 + (hand_y - cy) ** 2)
        if distance < 50:  # Within the radius
            return i
    return -1

# Start video capture
cap = cv2.VideoCapture(0)

# Generate the first question
question, correct_answer, options = generate_question()
answered = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror the image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hover_index = -1
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
            hover_index = detect_hover(x, y, draw_interface(frame, question, options, -1))
            break

    circle_positions = draw_interface(frame, question, options, hover_index)
    
    if hover_index != -1 and not answered:
        if options[hover_index] == correct_answer:
            score += 1
        answered = True
        question, correct_answer, options = generate_question()

    cv2.putText(frame, f"Score: {score}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Math Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()