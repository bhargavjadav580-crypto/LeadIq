import asyncio
from playwright.async_api import async_playwright

BASE = "https://leadiq-2.preview.emergentagent.com"
TOKEN = "b5-8Lk1LiEX_d4pH-DKIlGOrqbc2EhyegcAUa7sGN4E"
PID = "653622fa8476494fadbc62df2bfa405b"
OUT = "/app/reports/leadiq/assets"

async def shot(page, name, scroll=0, clip_h=None):
    if scroll:
        await page.evaluate(f"window.scrollBy(0,{scroll})")
        await page.wait_for_timeout(1200)
    opts = {"path": f"{OUT}/{name}"}
    await page.screenshot(**opts)
    print("saved", name)

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--no-sandbox"])
        ctx = await b.new_context(viewport={"width":1680,"height":1050}, device_scale_factor=2)
        page = await ctx.new_page()
        # landing
        await page.goto(BASE, wait_until="networkidle")
        await page.wait_for_timeout(2500)
        await page.screenshot(path=f"{OUT}/ss_landing.png")
        print("saved landing")
        # set project
        await page.evaluate(f'localStorage.setItem("leadiq-project", JSON.stringify({{id:"{PID}",token:"{TOKEN}"}}))')

        async def load(step_btn):
            await page.goto(BASE, wait_until="networkidle")
            await page.wait_for_timeout(3800)
            await page.click(f'[data-testid="{step_btn}"]', force=True)
            await page.wait_for_timeout(4200)

        # step1
        await load("step-1-button")
        await page.screenshot(path=f"{OUT}/ss_step1_selection.png")
        print("saved step1")
        # step3 explore
        await load("step-3-button")
        await page.evaluate("window.scrollBy(0,430)"); await page.wait_for_timeout(1200)
        await page.screenshot(path=f"{OUT}/ss_step3_explore.png")
        print("saved step3")
        # step5 train
        await load("step-5-button")
        await page.evaluate("window.scrollBy(0,400)"); await page.wait_for_timeout(1200)
        await page.screenshot(path=f"{OUT}/ss_step5_train.png")
        print("saved step5")
        # step6 overview
        await load("step-6-button")
        await page.screenshot(path=f"{OUT}/ss_step6.png")
        await page.evaluate("window.scrollBy(0,560)"); await page.wait_for_timeout(1200)
        await page.screenshot(path=f"{OUT}/ss_step6_overview.png")
        print("saved step6 overview")
        # step6 model comparison tab
        await load("step-6-button")
        try:
            await page.click("text=Model Comparison", force=True); await page.wait_for_timeout(2500)
            await page.evaluate("window.scrollBy(0,620)"); await page.wait_for_timeout(1200)
            await page.screenshot(path=f"{OUT}/ss_step6_compare.png")
            print("saved step6 compare")
        except Exception as e:
            print("compare err", e)
        # step7 ranking top + table
        await load("step-7-button")
        await page.screenshot(path=f"{OUT}/ss_step7_ranking.png")
        await page.evaluate("window.scrollBy(0,1050)"); await page.wait_for_timeout(1200)
        await page.screenshot(path=f"{OUT}/ss_step7_table.png")
        print("saved step7")
        # step8 explain + scroll for shap
        await load("step-8-button")
        await page.wait_for_timeout(1500)
        await page.screenshot(path=f"{OUT}/ss_step8_explain.png")
        await page.evaluate("window.scrollBy(0,560)"); await page.wait_for_timeout(1200)
        await page.screenshot(path=f"{OUT}/ss_step8_shap.png")
        print("saved step8")
        # step9 export
        await load("step-9-button")
        await page.screenshot(path=f"{OUT}/ss_step9_export.png")
        print("saved step9")
        await b.close()

asyncio.run(main())
print("CAPTURE DONE")
