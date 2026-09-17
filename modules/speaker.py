import subprocess


class Speaker:

    def speak(self, text):

        print("Speaking:", text)

        text = text.replace('"', "'")

        command = f'''
Add-Type -AssemblyName System.Speech;
$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;
$speak.Rate = 0;
$speak.Speak("{text}");
'''

        subprocess.run(
            ["powershell", "-Command", command],
            capture_output=True
        )