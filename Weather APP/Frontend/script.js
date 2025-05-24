document.getElementById("searchBtn").addEventListener("click", fetchWeather);

async function fetchWeather() {
    const cityInput = document.getElementById("cityInput");
    const city = cityInput.value.trim();
    const errorDiv = document.getElementById("error");
    const weatherDiv = document.getElementById("weather");
    const loadingDiv = document.getElementById("loading");

    if (!city) {
        errorDiv.textContent = "Please enter a city name";
        errorDiv.classList.remove("hidden");
        weatherDiv.classList.add("hidden");
        return;
    }

    errorDiv.classList.add("hidden");
    weatherDiv.classList.add("hidden");
    loadingDiv.classList.remove("hidden");

    try {
        const response = await fetch(`/api/weather/${encodeURIComponent(city)}`);
        const data = await response.json();

        if (response.ok) {
            document.getElementById("cityName").textContent = `${data.name}, ${data.sys.country}`;
            document.getElementById("weatherIcon").src = `http://openweathermap.org/img/wn/${data.weather[0].icon}@2x.png`;
            document.getElementById("description").textContent = data.weather[0].description;
            document.getElementById("temperature").textContent = `Temperature: ${Math.round(data.main.temp)}°C`;
            document.getElementById("humidity").textContent = `Humidity: ${data.main.humidity}%`;
            document.getElementById("wind").textContent = `Wind Speed: ${data.wind.speed} m/s`;
            weatherDiv.classList.remove("hidden");
        } else {
            errorDiv.textContent = data.error || "City not found";
            errorDiv.classList.remove("hidden");
        }
    } catch (error) {
        errorDiv.textContent = "Failed to fetch weather data";
        errorDiv.classList.remove("hidden");
    } finally {
        loadingDiv.classList.add("hidden");
    }
}