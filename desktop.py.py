import speech_recognition as sr
import datetime
import subprocess
import pywhatkit
import pyttsx3
import webbrowser
import matplotlib.pyplot as plt
from jiwer import wer
from sklearn.metrics import accuracy_score
from fuzzywuzzy import process

# Initialize text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Initialize recognizer
recognizer = sr.Recognizer()

# Store expected commands and results
expected_commands = []
recognized_commands = []
success_flags = []

# Define known command mappings
command_keywords = {
    'chrome': 'open chrome',
    'youtube': 'open youtube',
    'time': 'what is the time',
    'play': 'play youtube video'
}

def speak(text):
    engine.say(text)
    engine.runAndWait()

def cmd():
    global expected_commands, recognized_commands, success_flags
    with sr.Microphone() as source:
        print("Clearing background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language='en-US')
        text = text.lower()
        print("You said:", text)
        recognized_commands.append(text)

        matched_keys = []
        for key in command_keywords:
            if key in text:
                matched_keys.append(key)

        if matched_keys:
            for key in matched_keys:
                expected_commands.append(command_keywords[key])
                success_flags.append(True)

                if key == 'chrome':
                    speak("Opening Chrome...")
                    subprocess.Popen(["C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"])
                elif key == 'youtube':
                    speak("Opening YouTube...")
                    webbrowser.open("https://www.youtube.com")
                elif key == 'time':
                    current_time = datetime.datetime.now().strftime('%I:%M %p')
                    print(current_time)
                    speak(current_time)
                elif key == 'play':
                    speak("Playing on YouTube...")
                    pywhatkit.playonyt(text)
        else:
            speak("Sorry, I didn’t understand.")
            expected_commands.append("unknown command")
            success_flags.append(False)

    except Exception as e:
        print("Error:", e)
        expected_commands.append("unknown command")
        recognized_commands.append("none")
        success_flags.append(False)

def show_results():
    print("\n----- Evaluation Metrics -----")
    combined_expected = " ".join(expected_commands)
    combined_recognized = " ".join(recognized_commands)
    error_rate = wer(combined_expected, combined_recognized)
    print(f"Word Error Rate (WER): {error_rate:.2f}")

    y_true = [1 if cmd != "unknown command" else 0 for cmd in expected_commands]
    y_pred = [1 if success else 0 for success in success_flags]

    if len(y_true) == len(y_pred):
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc * 100:.2f}%")
    else:
        print("Mismatch in true/predicted lengths. Cannot compute accuracy.")

    # Visualization
    labels = ['Success', 'Fail']
    counts = [sum(success_flags), len(success_flags) - sum(success_flags)]
    colors = ['green', 'red']

    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title("Command Success Rate")
    plt.show()

# Run the voice command system N times
for _ in range(3):
    cmd()



import matplotlib.pyplot as plt
from collections import Counter

def visualize_results(expected_commands, recognized_commands, success_flags):
    # Visualization 1: Pie chart for success rate
    labels = ['Success', 'Fail']
    counts = [sum(success_flags), len(success_flags) - sum(success_flags)]
    colors = ['green', 'red']
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title("Command Success Rate")
    plt.show()

    # Visualization 2: Bar chart for command frequency
    cmd_counter = Counter(expected_commands)
    plt.figure(figsize=(8, 5))
    plt.bar(cmd_counter.keys(), cmd_counter.values(), color='skyblue')
    plt.title("Frequency of Commands")
    plt.xlabel("Command")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Visualization 3: Line plot of success/failure trend
    plt.figure(figsize=(8, 4))
    plt.plot(success_flags, marker='o', linestyle='--', color='purple')
    plt.title("Success/Fail Trend Over Attempts")
    plt.xlabel("Attempt Number")
    plt.ylabel("Success (1) / Fail (0)")
    plt.grid(True)
    plt.yticks([0, 1], ['Fail', 'Success'])
    plt.tight_layout()
    plt.show()
show_results()
visualize_results(expected_commands, recognized_commands, success_flags)

