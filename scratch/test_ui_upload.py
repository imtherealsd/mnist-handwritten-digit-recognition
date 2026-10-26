import os
import time
from playwright.sync_api import sync_playwright

def run():
    collage_path = os.path.abspath("sample_images/user_uploaded_collage.png")
    screenshot_path = r"C:\Users\sdoff\.gemini\antigravity-ide\brain\b2dc7757-0c94-47c0-84d9-7aadc17fb6c4\collage_recognition_fullpage.png"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 2200})
        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", timeout=60000)
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        
        # Click on 'Upload Digit Image' tab
        print("Clicking 'Upload Digit Image' tab...")
        page.locator("button[data-baseweb='tab']").nth(1).click()
        time.sleep(1)
        
        # Upload file directly to input[type=file]
        print(f"Setting input file: {collage_path}")
        page.locator("input[type='file']").set_input_files(collage_path)
        
        # Wait for Streamlit to finish upload and display "Uploaded Image"
        print("Waiting for Uploaded Image preview to appear...")
        page.locator("text=Uploaded Image").wait_for(state="visible", timeout=25000)
        print("Image uploaded and preview rendered!")
        time.sleep(2)
        
        # Click 'Predict Digit(s)'
        print("Clicking 'Predict Digit(s)' button...")
        page.locator("button:has-text('Predict Digit(s)')").click()
        
        # Wait for inference and rendering
        print("Waiting for predictions...")
        page.locator("text=2D MULTI-DIGIT / COLLAGE DETECTION RESULT").wait_for(state="visible", timeout=30000)
        time.sleep(2)
        
        # Take full page screenshot
        print("Taking full page screenshot...")
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"Saved screenshot to: {screenshot_path}")
        
        browser.close()

if __name__ == "__main__":
    run()
