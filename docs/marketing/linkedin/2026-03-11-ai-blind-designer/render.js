const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1080, height: 1350 },
    deviceScaleFactor: 2,
  });
  const page = await context.newPage();
  await page.goto('http://localhost:8099/docs/marketing/linkedin/2026-03-11-ai-blind-designer/ai-blind-designer-carousel.html');
  await page.waitForTimeout(2000);

  const slides = await page.$$('.slide');
  const dir = '/workspaces/corvia-workspace/docs/marketing/linkedin/2026-03-11-ai-blind-designer';

  for (let i = 0; i < slides.length; i++) {
    const num = String(i + 1).padStart(2, '0');
    await slides[i].screenshot({ path: `${dir}/slide-${num}.png` });
    console.log(`Rendered slide-${num}.png`);
  }

  // PDF
  const pdfPage = await context.newPage();
  await pdfPage.goto('http://localhost:8099/docs/marketing/linkedin/2026-03-11-ai-blind-designer/ai-blind-designer-carousel.html');
  await pdfPage.waitForTimeout(2000);
  await pdfPage.pdf({
    path: `${dir}/ai-blind-designer-carousel.pdf`,
    width: '1080px',
    height: '1350px',
    margin: { top: 0, right: 0, bottom: 0, left: 0 },
    printBackground: true,
  });
  console.log('Rendered PDF');

  await browser.close();
})();
