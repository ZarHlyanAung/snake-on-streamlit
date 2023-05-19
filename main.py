import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the pre-trained gesture classification model
model = tf.keras.models.load_model('gesture_model.h5')

# Set up MediaPipe for hand detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set up the game constants
WIDTH = 20
HEIGHT = 20

# Define the game variables
snake = [(HEIGHT//2, WIDTH//2)]
direction = (0, 1)
apple = (np.random.randint(0, HEIGHT), np.random.randint(0, WIDTH))
score = 0
game_over = False

# Function to update the game state
def update_game_state(gesture):
    global snake, direction, apple, score, game_over

    # Map the predicted gesture to snake movement
    if gesture == 'go':
        direction = (-1, 0)
    elif gesture == 'left':
        direction = (0, -1)
    elif gesture == 'right':
        direction = (0, 1)

    # Update snake position
    head = snake[0]
    new_head = (head[0] + direction[0], head[1] + direction[1])
    snake.insert(0, new_head)

    # Check if snake ate the apple
    if snake[0] == apple:
        score += 1
        apple = (np.random.randint(0, HEIGHT), np.random.randint(0, WIDTH))
    else:
        snake.pop()

    # Check if snake collided with the walls or itself
    if (
        snake[0][0] < 0 or snake[0][0] >= HEIGHT or
        snake[0][1] < 0 or snake[0][1] >= WIDTH or
        snake[0] in snake[1:]
    ):
        game_over = True

# Function to process video frames and make predictions
def process_frame(frame):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # Convert the frame to RGB and feed it to MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract the hand region
                height, width, _ = image.shape
                x_min, x_max = int(hand_landmarks.bounding_box.min_x * width), int(hand_landmarks.bounding_box.max_x * width)
                y_min, y_max = int(hand_landmarks.bounding_box.min_y * height), int(hand_landmarks.bounding_box.max_y * height)
                hand_img = image[y_min:y_max, x_min:x_max]
                
                # Resize the hand image to the desired input shape of the model
                hand_img = cv2.resize(hand_img, (224, 224))
                hand_img = np.expand_dims(hand_img, axis=0)
                hand_img = hand_img / 255.0
                
                # Make a prediction with the model
                prediction = model.predict(hand_img)
                gesture = ['go', 'left', 'right'][np.argmax(prediction)]
                
                # Update the game state based on the predicted gesture
                update_game_state(gesture)

        # Draw hand landmarks on the image
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert the image back to BGR for display in Streamlit
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

# Main Streamlit app
def main():
    st.title('Gesture Snake Game')

    # Set up the video capture
    cap = cv2.VideoCapture(0)

    # Run the game loop
    while not game_over:
        ret, frame = cap.read()
        if ret:
            processed_frame = process_frame(frame)
            st.image(processed_frame, channels='BGR', use_column_width=True)

    # Release the video capture and display the final score
    cap.release()
    st.write(f"Game Over! Your score is: {score}")

if __name__ == '__main__':
    main()
