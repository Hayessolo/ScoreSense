from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
from datetime import datetime

# Set up Chrome options
chrome_options = Options()
# Uncomment the line below if you want to run in headless mode
# chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")

# Initialize the driver
driver = webdriver.Chrome(options=chrome_options)

# URL of the betting site - replace with the actual URL
url = ""
driver.get(url)

# Wait for page to load
time.sleep(5)  # Basic wait, I'll improve this