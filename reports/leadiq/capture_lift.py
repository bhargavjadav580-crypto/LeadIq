import asyncio
from playwright.async_api import async_playwright
BASE="https://leadiq-2.preview.emergentagent.com"
TOKEN="b5-8Lk1LiEX_d4pH-DKIlGOrqbc2EhyegcAUa7sGN4E"; PID="653622fa8476494fadbc62df2bfa405b"
OUT="/app/reports/leadiq/assets"
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(args=["--no-sandbox"])
        ctx=await b.new_context(viewport={"width":1680,"height":1050},device_scale_factor=2)
        page=await ctx.new_page()
        await page.goto(BASE,wait_until="networkidle")
        await page.evaluate(f'localStorage.setItem("leadiq-project", JSON.stringify({{id:"{PID}",token:"{TOKEN}"}}))')
        await page.goto(BASE,wait_until="networkidle"); await page.wait_for_timeout(3800)
        await page.click('[data-testid="step-6-button"]',force=True); await page.wait_for_timeout(3000)
        await page.click("text=Lift & Gain",force=True); await page.wait_for_timeout(2500)
        await page.evaluate("window.scrollBy(0,640)"); await page.wait_for_timeout(1500)
        await page.screenshot(path=f"{OUT}/ss_step6_lift.png")
        print("saved lift")
        await b.close()
asyncio.run(main())
