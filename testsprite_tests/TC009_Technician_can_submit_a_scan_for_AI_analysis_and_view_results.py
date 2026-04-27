import asyncio
from playwright import async_api
from playwright.async_api import expect

async def run_test():
    pw = None
    browser = None
    context = None

    try:
        # Start a Playwright session in asynchronous mode
        pw = await async_api.async_playwright().start()

        # Launch a Chromium browser in headless mode with custom arguments
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--window-size=1280,720",         # Set the browser window size
                "--disable-dev-shm-usage",        # Avoid using /dev/shm which can cause issues in containers
                "--ipc=host",                     # Use host-level IPC for better stability
                "--single-process"                # Run the browser in a single process mode
            ],
        )

        # Create a new browser context (like an incognito window)
        context = await browser.new_context()
        context.set_default_timeout(5000)

        # Open a new page in the browser context
        page = await context.new_page()

        # Interact with the page elements to simulate user flow
        # -> Navigate to http://localhost:5173
        await page.goto("http://localhost:5173")
        
        # -> Open the screening/login flow by clicking the 'Start Screening' button to reach the login or role-selection screen.
        frame = context.pages[-1]
        # Click element
        elem = frame.locator('xpath=/html/body/div/div/nav/div/div/div[2]/button').nth(0)
        await asyncio.sleep(3); await elem.click()
        
        # -> Navigate to the login page (/login) to begin the Technician sign-in flow.
        await page.goto("http://localhost:5173/login")
        
        # -> Click the 'Technician' role button to proceed to the Technician sign-in/login form.
        frame = context.pages[-1]
        # Click element
        elem = frame.locator('xpath=/html/body/div/div/div[2]/div[3]/div/div[2]/button[3]').nth(0)
        await asyncio.sleep(3); await elem.click()
        
        # -> Fill the Technician credentials (email and password) into the login form and click 'Sign In'.
        frame = context.pages[-1]
        # Input text
        elem = frame.locator('xpath=/html/body/div/div/div[2]/div[3]/div/div[2]/div/input').nth(0)
        await asyncio.sleep(3); await elem.fill('tech@eyeassist.demo')
        
        frame = context.pages[-1]
        # Input text
        elem = frame.locator('xpath=/html/body/div/div/div[2]/div[3]/div/div[2]/div[2]/input').nth(0)
        await asyncio.sleep(3); await elem.fill('Demo@1234')
        
        frame = context.pages[-1]
        # Click element
        elem = frame.locator('xpath=/html/body/div/div/div[2]/div[3]/div/div[2]/button').nth(0)
        await asyncio.sleep(3); await elem.click()
        
        # -> Click the 'OD' laterality button to set the eye side. After the UI reflects the selection, fill patient ID, name, age, upload the fundus image, then click Analyze and verify results.
        frame = context.pages[-1]
        # Click element
        elem = frame.locator('xpath=/html/body/div/div/div[2]/main/div/div/div/div/div/div/div/div[2]/div/div[2]/div[2]/div/button').nth(0)
        await asyncio.sleep(3); await elem.click()
        
        # -> Fill patient metadata (Patient ID, Patient Name, Age), upload a valid fundus image, click Analyze, then wait and extract the AI results to verify severity, confidence, per-disease probabilities, original image, and heatmap.
        frame = context.pages[-1]
        # Input text
        elem = frame.locator('xpath=/html/body/div/div/div[2]/main/div/div/div/div/div/div/div/div[2]/div/div/div/input').nth(0)
        await asyncio.sleep(3); await elem.fill('P-2026-1001')
        
        frame = context.pages[-1]
        # Input text
        elem = frame.locator('xpath=/html/body/div/div/div[2]/main/div/div/div/div/div/div/div/div[2]/div/div/div[2]/input').nth(0)
        await asyncio.sleep(3); await elem.fill('Test Patient')
        
        frame = context.pages[-1]
        # Input text
        elem = frame.locator('xpath=/html/body/div/div/div[2]/main/div/div/div/div/div/div/div/div[2]/div/div[2]/div/input').nth(0)
        await asyncio.sleep(3); await elem.fill('45')
        
        # -> Upload a valid fundus image using the file input (index 930), then click the 'Analyze Image' button (index 963) to submit for analysis.
        frame = context.pages[-1]
        # Click element
        elem = frame.locator('xpath=/html/body/div/div/div[2]/main/div/div/div[1]/div/div/div/div/div[3]/button[2]').nth(0)
        await asyncio.sleep(3); await elem.click()
        
        # --> Assertions to verify final state
        frame = context.pages[-1]
        assert "Severity" in await frame.locator("xpath=//*[contains(., 'Severity')]" ).nth(0).text_content() and "Confidence" in await frame.locator("xpath=//*[contains(., 'Severity')]" ).nth(0).text_content() and "Diabetic retinopathy" in await frame.locator("xpath=//*[contains(., 'Severity')]" ).nth(0).text_content(), "The AI results should display severity, confidence, and per-disease probabilities after analysis"
        assert await frame.locator("xpath=//*[contains(., 'Original image')]" ).nth(0).is_visible() and await frame.locator("xpath=//*[contains(., 'Heatmap')]" ).nth(0).is_visible(), "The original image and explainability heatmap should be visible after analysis"
        await asyncio.sleep(5)

    finally:
        if context:
            await context.close()
        if browser:
            await browser.close()
        if pw:
            await pw.stop()

asyncio.run(run_test())
    