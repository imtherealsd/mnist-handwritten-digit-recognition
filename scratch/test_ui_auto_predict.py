import os
import time
from playwright.sync_api import sync_playwright

def run():
    img_typed = os.path.abspath("sample_images/typed_numbers_test.png")
    img_dark = os.path.abspath("sample_images/test_dark_printed.png")
    screenshot1 = r"C:\Users\sdoff\.gemini\antigravity-ide\brain\b2dc7757-0c94-47c0-84d9-7aadc17fb6c4\auto_pred_typed.png"
    screenshot2 = r"C:\Users\sdoff\.gemini\antigravity-ide\brain\b2dc7757-0c94-47c0-84d9-7aadc17fb6c4\auto_pred_dark.png"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1800})
        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", timeout=60000)
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # 1. Upload first image: typed_numbers_test.png
        print(f"Uploading 1st image: {img_typed}")
        file_input = page.locator("input[type='file']")
        file_input.set_input_files(img_typed)

        print("Waiting for auto-prediction of 1st image...")
        page.locator("text=COLLAGE DETECTION RESULT").wait_for(state="visible", timeout=30000)
        time.sleep(2)
        page.screenshot(path=screenshot1, full_page=True)
        print(f"Saved 1st screenshot to: {screenshot1}")

        # 2. Upload second image: test_dark_printed.png
        print(f"Uploading 2nd image (different file): {img_dark}")
        file_input.set_input_files(img_dark)

        print("Waiting for auto-prediction of 2nd image...")
        page.locator("text=MULTI-DIGIT RECOGNITION RESULT").wait_for(state="visible", timeout=30000)
        time.sleep(2)
        page.screenshot(path=screenshot2, full_page=True)
        print(f"Saved 2nd screenshot to: {screenshot2}")

        browser.close()

if __name__ == "__main__":
    run()
