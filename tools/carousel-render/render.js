#!/usr/bin/env node
/**
 * carousel-render — Render and validate LinkedIn carousel slides.
 *
 * Usage:
 *   node render.js --url <http-url> --outdir <path> [--format linkedin-carousel] [--skip-validation]
 *
 * Prerequisites:
 *   - HTTP server running (serves workspace root on port 8099)
 *   - Playwright + Chromium installed
 *
 * Output:
 *   - slide-01.png through slide-NN.png (2x retina)
 *   - carousel.pdf (multi-page, print-background)
 *   - validation report to stdout
 */

const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const DESIGN_SYSTEM_PATH = path.resolve(__dirname, '../../docs/marketing/design-system.json');

const DEFAULTS = {
  format: 'linkedin-carousel',
  port: 8099,
};

const FORMATS = {
  'linkedin-carousel': {
    width: 1080,
    height: 1350,
    deviceScaleFactor: 2,
    slideSelector: '.slide',
    minFontSize: 15,
    safeAreaHeight: 1110,
  },
};

// ---------------------------------------------------------------------------
// Arg parsing
// ---------------------------------------------------------------------------

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = { format: DEFAULTS.format, skipValidation: false };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--url':       opts.url = args[++i]; break;
      case '--outdir':    opts.outdir = args[++i]; break;
      case '--format':    opts.format = args[++i]; break;
      case '--skip-validation': opts.skipValidation = true; break;
      case '--help':
        console.log('Usage: node render.js --url <url> --outdir <path> [--format linkedin-carousel] [--skip-validation]');
        process.exit(0);
    }
  }

  if (!opts.url) { console.error('Error: --url is required'); process.exit(1); }
  if (!opts.outdir) { console.error('Error: --outdir is required'); process.exit(1); }

  return opts;
}

// ---------------------------------------------------------------------------
// Pre-flight checks
// ---------------------------------------------------------------------------

async function preflight(url) {
  const issues = [];

  // Check HTTP server is reachable
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) issues.push(`HTTP ${res.status} for ${url} — is the server running from the workspace root?`);
  } catch (e) {
    issues.push(`Cannot reach ${url} — start HTTP server: python3 -m http.server 8099 --bind 127.0.0.1`);
  }

  // Check Chromium is installed
  try {
    const browser = await chromium.launch();
    await browser.close();
  } catch (e) {
    issues.push(`Chromium not available: ${e.message}\nFix: npx playwright install chromium`);
  }

  if (issues.length > 0) {
    console.error('\n--- Pre-flight FAILED ---');
    issues.forEach((msg, i) => console.error(`  ${i + 1}. ${msg}`));
    console.error('');
    process.exit(1);
  }

  console.log('[preflight] OK — server reachable, Chromium available');
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

async function renderSlides(page, format, outdir) {
  const slides = await page.$$(format.slideSelector);
  if (slides.length === 0) {
    console.error(`Error: No elements matching "${format.slideSelector}" found on page.`);
    process.exit(1);
  }

  const pngPaths = [];
  for (let i = 0; i < slides.length; i++) {
    const num = String(i + 1).padStart(2, '0');
    const pngPath = path.join(outdir, `slide-${num}.png`);
    await slides[i].screenshot({ path: pngPath });
    pngPaths.push(pngPath);
    console.log(`[render] slide-${num}.png`);
  }

  return pngPaths;
}

async function renderPDF(context, url, format, outdir) {
  const pdfPage = await context.newPage();
  await pdfPage.goto(url);
  await pdfPage.waitForFunction(() => document.fonts.ready);
  await pdfPage.waitForTimeout(1000);

  // Find the HTML filename for naming the PDF
  const urlPath = new URL(url).pathname;
  const htmlName = path.basename(urlPath, '.html');
  const pdfPath = path.join(outdir, `${htmlName}.pdf`);

  await pdfPage.pdf({
    path: pdfPath,
    width: `${format.width}px`,
    height: `${format.height}px`,
    margin: { top: 0, right: 0, bottom: 0, left: 0 },
    printBackground: true,
  });

  await pdfPage.close();
  console.log(`[render] ${path.basename(pdfPath)}`);
  return pdfPath;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

async function validate(page, format, pngPaths) {
  const results = { pass: 0, warn: 0, fail: 0, messages: [] };

  function pass(msg) { results.pass++; results.messages.push(`  PASS  ${msg}`); }
  function warn(msg) { results.warn++; results.messages.push(`  WARN  ${msg}`); }
  function fail(msg) { results.fail++; results.messages.push(`  FAIL  ${msg}`); }

  // 1. Check PNG files exist and are non-trivial
  for (const p of pngPaths) {
    if (!fs.existsSync(p)) { fail(`Missing: ${path.basename(p)}`); continue; }
    const size = fs.statSync(p).size;
    if (size < 10240) { warn(`${path.basename(p)} is only ${(size / 1024).toFixed(1)}KB — may be blank`); }
    else { pass(`${path.basename(p)} (${(size / 1024).toFixed(0)}KB)`); }
  }

  // 2. Check for text overflow (scrollHeight > clientHeight on text containers)
  const overflowReport = await page.evaluate((selector) => {
    const slides = document.querySelectorAll(selector);
    const issues = [];
    slides.forEach((slide, idx) => {
      // Check all elements with text content
      const els = slide.querySelectorAll('div, p, span, h1, h2, h3, h4, li');
      els.forEach(el => {
        if (el.scrollHeight > el.clientHeight + 2 && el.textContent.trim().length > 0) {
          const style = getComputedStyle(el);
          // Skip intentionally hidden overflow (decorative)
          if (style.overflow === 'hidden' && el.children.length > 0 && el.textContent.trim().length < 5) return;
          issues.push({
            slide: idx + 1,
            text: el.textContent.trim().substring(0, 50),
            scrollH: el.scrollHeight,
            clientH: el.clientHeight,
            overflow: el.scrollHeight - el.clientHeight,
          });
        }
      });
    });
    return issues;
  }, format.slideSelector);

  if (overflowReport.length === 0) {
    pass('No text overflow detected');
  } else {
    overflowReport.forEach(o => {
      warn(`Slide ${o.slide}: overflow ${o.overflow}px — "${o.text}..."`);
    });
  }

  // 3. Check font sizes (nothing below minimum)
  const fontReport = await page.evaluate(({ selector, minSize }) => {
    const slides = document.querySelectorAll(selector);
    const issues = [];
    slides.forEach((slide, idx) => {
      const els = slide.querySelectorAll('*');
      els.forEach(el => {
        if (el.textContent.trim().length === 0) return;
        if (el.children.length > 0 && el.children[0].textContent === el.textContent) return;
        const size = parseFloat(getComputedStyle(el).fontSize);
        if (size < minSize && size > 0) {
          issues.push({
            slide: idx + 1,
            text: el.textContent.trim().substring(0, 40),
            fontSize: size,
          });
        }
      });
    });
    return issues;
  }, { selector: format.slideSelector, minSize: format.minFontSize });

  if (fontReport.length === 0) {
    pass(`All text >= ${format.minFontSize}px minimum`);
  } else {
    fontReport.forEach(f => {
      warn(`Slide ${f.slide}: ${f.fontSize}px font — "${f.text}..."`);
    });
  }

  // 4. Check content height vs safe area
  const heightReport = await page.evaluate(({ selector, maxHeight }) => {
    const slides = document.querySelectorAll(selector);
    const issues = [];
    slides.forEach((slide, idx) => {
      // Measure total content height excluding absolute-positioned and flex-grow spacer elements
      let totalHeight = 0;
      const children = slide.children;
      for (const child of children) {
        const style = getComputedStyle(child);
        if (style.position === 'absolute') continue;
        // Skip flex-grow spacers (empty divs used for vertical push)
        if (style.flexGrow !== '0' && child.textContent.trim().length === 0) continue;
        const rect = child.getBoundingClientRect();
        totalHeight += rect.height;
      }
      // Add gap estimates (flex-direction: column with gap)
      const slideStyle = getComputedStyle(slide);
      const gap = parseFloat(slideStyle.gap) || parseFloat(slideStyle.rowGap) || 0;
      // Count visible children for gap calculation
      let visibleChildren = 0;
      for (const child of children) {
        const s = getComputedStyle(child);
        if (s.position === 'absolute') continue;
        if (s.flexGrow !== '0' && child.textContent.trim().length === 0) continue;
        visibleChildren++;
      }
      if (visibleChildren > 1) totalHeight += gap * (visibleChildren - 1);

      if (totalHeight > maxHeight) {
        issues.push({ slide: idx + 1, contentHeight: Math.round(totalHeight), max: maxHeight });
      }
    });
    return issues;
  }, { selector: format.slideSelector, maxHeight: format.height - 96 }); // 96px reserved for footer

  if (heightReport.length === 0) {
    pass('All slides within height budget');
  } else {
    heightReport.forEach(h => {
      fail(`Slide ${h.slide}: content ${h.contentHeight}px exceeds ${h.max}px safe area (${h.contentHeight - h.max}px over)`);
    });
  }

  // Summary
  console.log('\n--- Validation Report ---');
  results.messages.forEach(m => console.log(m));
  console.log(`\n  Total: ${results.pass} pass, ${results.warn} warn, ${results.fail} fail`);
  if (results.fail > 0) console.log('  Action needed: fix FAIL items and re-render');
  console.log('');

  return results;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

(async () => {
  const opts = parseArgs();
  const format = FORMATS[opts.format];
  if (!format) {
    console.error(`Unknown format: ${opts.format}. Available: ${Object.keys(FORMATS).join(', ')}`);
    process.exit(1);
  }

  // Ensure output directory exists
  fs.mkdirSync(opts.outdir, { recursive: true });

  // Pre-flight
  await preflight(opts.url);

  // Launch browser
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: format.width, height: format.height },
    deviceScaleFactor: format.deviceScaleFactor,
  });

  const page = await context.newPage();
  await page.goto(opts.url);
  await page.waitForFunction(() => document.fonts.ready);
  await page.waitForTimeout(1000);

  // Render PNGs
  const pngPaths = await renderSlides(page, format, opts.outdir);

  // Render PDF
  await renderPDF(context, opts.url, format, opts.outdir);

  // Validate
  if (!opts.skipValidation) {
    await validate(page, format, pngPaths);
  }

  await browser.close();
  console.log('[done] Rendered', pngPaths.length, 'slides + PDF');
})();
