import requests
import speech_recognition as sr
import threading
from mtranslate import translate
from colorama import Fore, init
from indic_transliteration import sanscript
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import tempfile
import os
import pyttsx3
import platform
import sys
import re
import time
from datetime import datetime
import pytz

init(autoreset=True)

speech_lock = threading.Lock()
stop_news = threading.Event()
news_thread = None

def speak(text):
    """Create a fresh pyttsx3 engine each time to avoid hangs."""
    try:
        with speech_lock:
            print(f"[Speaking: {text[:50]}...]")
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1)

            if platform.system() == "Linux":
                try:
                    engine.setProperty('driver', 'espeak')
                except Exception:
                    pass

            engine.say(text)
            engine.runAndWait()
            engine.stop()
    except Exception as e:
        print(Fore.RED + f"Error in speech synthesis: {e}", flush=True)
        print(Fore.CYAN + f"Speech: {text}", flush=True)


def recognize_with_timeout(recognizer, audio, language, timeout=5):
    """Recognize speech with a timeout to prevent hanging."""
    result = [None]
    exception = [None]

    def recognize_thread():
        try:
            result[0] = recognizer.recognize_google(audio, language=language).lower()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=recognize_thread)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None
    elif exception[0]:
        raise exception[0]
    else:
        return result[0]


def Translate_bengali_to_english(text):
    try:
        bengali_script = sanscript.transliterate(text, sanscript.ITRANS, sanscript.BENGALI)
        return translate(bengali_script, "en", "bn")
    except:
        return text


def Translate_hindi_to_english(text):
    try:
        hindi_script = sanscript.transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
        return translate(hindi_script, "en", "hi")
    except:
        return text


def news():
    """Fetch and speak top news headlines, checking for stop signal."""
    global news_thread
    try:
        print("Fetching news from API...")
        main_url = 'https://newsapi.org/v2/top-headlines?country=us&apiKey=a9a513a6ed2b46308fcd2ee644c72be0'
        main_page = requests.get(main_url, timeout=10).json()
        articles = main_page['articles']
        head = [ar['title'] for ar in articles]
        day = ["first", "second", "third", "fourth", "fifth"]
        print(f"Got {len(head)} news articles")
        for i in range(min(len(day), len(head))):
            if stop_news.is_set():
                print("\nNews reading interrupted.")
                break
            news_text = f"Today's {day[i]} news is: {head[i]}"
            print(f"Speaking: {news_text[:60]}...")
            speak(news_text)
            time.sleep(3)
        print("News fetching completed")
    except Exception as e:
        error_msg = f"Sorry, I couldn't fetch the news. Error: {str(e)}"
        print(Fore.RED + error_msg, flush=True)
        speak(error_msg)
    finally:
        stop_news.clear()
        news_thread = None


def Speech_to_text_Python():
    global news_thread, stop_news
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_adjustment_damping = 0.015
    recognizer.dynamic_energy_ratio = 1.5
    recognizer.pause_threshold = 0.5
    recognizer.operation_timeout = None
    recognizer.non_speaking_duration = 0.2

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 2048
    SILENCE_THRESHOLD = 1.0

    while True:
        print(Fore.GREEN + "Listening...", end="", flush=True)
        try:
            audio_chunks = []
            silence_duration = 0.0
            is_recording = False

            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE) as stream:
                print(Fore.GREEN + " (recording until 1 sec pause)", end="", flush=True)
                while True:
                    audio_chunk, _ = stream.read(CHUNK_SIZE)
                    audio_chunks.append(audio_chunk)
                    chunk_energy = np.sqrt(np.mean(audio_chunk ** 2))

                    if chunk_energy > 0.02:
                        is_recording = True
                        silence_duration = 0.0
                    elif is_recording:
                        silence_duration += CHUNK_SIZE / SAMPLE_RATE
                        if silence_duration >= SILENCE_THRESHOLD:
                            break

            audio_data = np.concatenate(audio_chunks, axis=0)
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            wavfile.write(temp_wav, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))

            print("\r" + Fore.LIGHTCYAN_EX + "Recog....", end="", flush=True)

            with sr.AudioFile(temp_wav) as source:
                audio = recognizer.record(source)

            recognizer_text = ""
            is_bengali = False

            try:
                recognizer_text = recognize_with_timeout(recognizer, audio, "bn-IN", timeout=5)
                if recognizer_text is None:
                    raise sr.WaitTimeoutError("Recognition timeout")
                is_bengali = True
            except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
                try:
                    recognizer_text = recognize_with_timeout(recognizer, audio, "en-US", timeout=5)
                    if recognizer_text is None:
                        raise sr.WaitTimeoutError("Recognition timeout")
                    is_bengali = False
                except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
                    recognizer_text = ""

            if recognizer_text:
                eng_text = Translate_bengali_to_english(recognizer_text) if is_bengali else recognizer_text

                # ---------- COMMAND HANDLING ----------
                # 0. Exit command (highest priority)
                if re.search(r'\b(exit|terminate)\s+(?:the\s+)?(program|programme)\b', eng_text, re.IGNORECASE):
                    print("\r" + Fore.RED + "Exiting program...")
                    speak("Goodbye!")
                    if news_thread is not None and news_thread.is_alive():
                        stop_news.set()
                    time.sleep(1)
                    break

                # 1. News command
                elif re.search(r'\bnews\b', eng_text, re.IGNORECASE):
                    if news_thread is not None and news_thread.is_alive():
                        print(Fore.YELLOW + "News already playing. Say 'pause' to stop.")
                    else:
                        print("Please wait, fetching the latest news")
                        speak("Please wait, fetching the latest news")
                        time.sleep(1)
                        stop_news.clear()
                        news_thread = threading.Thread(target=news)
                        news_thread.daemon = True
                        news_thread.start()

                # 2. Pause command
                elif re.search(r'\bpause\b', eng_text, re.IGNORECASE):
                    if news_thread is not None and news_thread.is_alive():
                        print("Pausing news...")
                        stop_news.set()
                        speak("Pausing news.")
                    else:
                        print("No news is currently playing.")

                # 3. Location command
                elif re.search(r'\b(where (?:are we|am i)|(?:our|my) location)\b', eng_text, re.IGNORECASE):
                    try:
                        ipAdd = requests.get('https://api.ipify.org').text
                        url = f'https://get.geojs.io/v1/ip/geo/{ipAdd}.json'
                        geo_requests = requests.get(url)
                        geo_data = geo_requests.json()
                        city = geo_data.get('city', 'unknown city')
                        country = geo_data.get('country', 'unknown country')
                        location_msg = f"We are in {city} city of {country}."
                        print(Fore.CYAN + location_msg)
                        speak(location_msg)
                    except Exception as e:
                        print(Fore.RED + f"Location error: {e}")
                        speak("Sorry sir, due to network issue I am not able to find where we are.")

                elif re.search(r'\b(?:what is the time|what is today\'s date|what is todays date|what is today date|what is the day)\b', eng_text, re.IGNORECASE):
                    try:
                        ipAdd = requests.get('https://api.ipify.org').text
                        url = f'https://get.geojs.io/v1/ip/geo/{ipAdd}.json'
                        geo_requests = requests.get(url)
                        geo_data = geo_requests.json()
                        timezone = geo_data.get("timezone")
                        tz = pytz.timezone(timezone)
                        now = datetime.now(tz)
                        if(re.search(r'\bwhat is the time\b', eng_text, re.IGNORECASE)):
                            time_msg = f"The current time is {now.strftime('%I:%M %p')}."
                            print(Fore.CYAN + time_msg)
                            speak(time_msg)
                        elif(re.search(r'\bwhat is today\'s date\b', eng_text, re.IGNORECASE) or re.search(r'\bwhat is todays date\b', eng_text, re.IGNORECASE) or re.search(r'\bwhat is today date\b', eng_text, re.IGNORECASE)):
                            date_msg = f"Today's date is {now.strftime('%d, %m, %Y')}."
                            print(Fore.CYAN + date_msg)
                            speak(date_msg)
                        elif(re.search(r'\bwhat is the dayy\b', eng_text, re.IGNORECASE)):
                            day_msg = f"Today is {now.strftime('%A')}."
                            print(Fore.CYAN + day_msg)
                            speak(day_msg)
                    except Exception as e:
                        print(Fore.RED + f"Location error: {e}")
                        speak("Sorry sir, due to network issue I am not able to find where we are.")

                # 4. Farewell command (goodbye, good bye, bye) - does NOT exit
                elif re.search(r'\b(goodbye|good bye|bye)\b', eng_text, re.IGNORECASE):
                    farewell = "Goodbye! Have a nice day!"
                    print("\r" + Fore.MAGENTA + farewell)
                    speak(farewell)

                # 5. Greeting command (hi/hello)
                elif re.search(r'\b(hi|hello)\b', eng_text, re.IGNORECASE):
                    greeting = "Hello! My name is ARIA. I have been made by Angshuk Dutta and Rishabh Gupta. How may I help you?"
                    print("\r" + Fore.MAGENTA + greeting)
                    speak(greeting)

                # 6. How are you command
                elif re.search(r'\bhow are you\b', eng_text, re.IGNORECASE):
                    how_response = "I am good, thank you. How are you?"
                    print("\r" + Fore.MAGENTA + how_response)
                    speak(how_response)

                # 7. No command – just echo what was said
                else:
                    print("\r" + Fore.BLUE + "You said" + (" (Bengali): " if is_bengali else ": ") + Fore.CYAN + eng_text)
            else:
                print("\r" + Fore.YELLOW + "No text recognized", end="", flush=True)

            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        except sr.UnknownValueError:
            pass
        except KeyboardInterrupt:
            print("\r" + Fore.RED + "Recording cancelled")
            break
        except Exception as e:
            print("\r" + Fore.RED + f"Error: {e}")
        finally:
            print("\r", end="", flush=True)


# Start the main loop
speech_thread = threading.Thread(target=Speech_to_text_Python)
speech_thread.daemon = True
speech_thread.start()
speech_thread.join()