# Configuration constants and maps for the Sri Lanka Climate App

ZONE_COLORS = {0:'#2d6abf', 1:'#7ab648', 2:'#c47d2d', 3:'#9b59b6', 4:'#e74c3c', -1:'#888888'}
ZONE_BG     = {0:'#e8f0fb', 1:'#edf6e0', 2:'#fdf0e0', 3:'#f5e8fb', 4:'#fbe8e8', -1:'#f0f0f0'}
ZONE_TEXT   = {0:'#1a3d8f', 1:'#2d5a1b', 2:'#8f5a1a', 3:'#5a1a8f', 4:'#8f1a1a', -1:'#555555'}

MONTH_NAMES = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

WMO_CODES = {
    0:'Clear sky ☀️', 1:'Mainly clear 🌤️', 2:'Partly cloudy ⛅', 3:'Overcast ☁️',
    51:'Light drizzle 🌦️', 53:'Moderate drizzle 🌦️', 55:'Dense drizzle 🌧️',
    61:'Slight rain 🌧️', 63:'Moderate rain 🌧️', 65:'Heavy rain ⛈️'
}

# Feature labels for display/LIME (clean versions)
FEAT_LABELS_CLEAN = {
    'temperature_2m_mean': 'Mean Temp',
    'temperature_2m_max': 'Max Temp',
    'temperature_2m_min': 'Min Temp',
    'windspeed_10m_max': 'Wind Speed',
    'shortwave_radiation_sum': 'Solar Radiation',
    'et0_fao_evapotranspiration': 'Evapotranspiration',
    'temp_range': 'Temp Range',
    'month': 'Month',
    'climate_zone': 'Climate Zone',
    'elevation': 'Elevation',
    'sw_rain': 'SW Monsoon Rain',
    'ne_rain': 'NE Monsoon Rain',
    'monsoon_ratio': 'Monsoon Ratio'
}
