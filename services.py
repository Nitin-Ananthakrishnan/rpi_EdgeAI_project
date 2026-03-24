import requests
import datetime

# --- CONFIGURATION ---
API_KEY = " "
CITY = " "  # Change to your city

def get_daily_briefing():
    # 1. TIME LOGIC (Calculated locally)
    now = datetime.datetime.now()
    current_time = now.strftime("%I:%M %p")
    
    hour = now.hour
    if hour < 12: greeting = "Good morning"
    elif hour < 17: greeting = "Good afternoon"
    else: greeting = "Good evening"

    # 2. REAL API INTEGRATION
    try:
        # We call the OpenWeatherMap API
        url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
        
        # We send the request and convert the response to JSON
        response = requests.get(url, timeout=5)
        data = response.json()

        if response.status_code == 200:
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            weather_sentence = f"The weather in {CITY} is {temp} degrees with {desc}."
        else:
            weather_sentence = "I can access the time, but the weather service is currently unreachable."
            
    except Exception as e:
        # FALLBACK: If the internet is down, the system doesn't crash!
        print(f"API Error: {e}")
        weather_sentence = "The weather data is unavailable, but the internet is not required for my core AI functions."

    # 3. CONSTRUCT THE FINAL SENTENCE
    return f"{greeting}! It is {current_time}. {weather_sentence} Have a wonderful day."
