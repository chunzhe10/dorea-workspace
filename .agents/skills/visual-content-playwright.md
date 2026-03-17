# Visual Content Production with Playwright MCP

Guidelines for AI agents producing visual content (LinkedIn carousels, dashboards, brand guides) using HTML/CSS + Playwright MCP for rendering and verification.

## Content Strategy: Slides Carry the Story

**The #1 rule:** Slides must tell the complete story standalone. Many LinkedIn readers never read the post description — they just swipe.

### Slides (primary content)
- Each slide must be self-contained — understandable without reading the post or even the previous slide.
- Include "what is this project?" context on the cover (don't assume the reader knows).
- Use subtitles and insight boxes to carry context that would otherwise be in the post.
- Explain jargon on the slide where it first appears.
- Prefer diagrams, flow charts, and stat boxes over paragraphs. If a slide looks like a wall of text, redesign it.

### Post description (guide, not essay)
- Short narrative arc (6-8 paragraphs, 250-350 words) that walks readers through the slides.
- Each paragraph maps to ~1-2 slides. Reference slides when detail lives there: "(Slide 5 has the breakdown.)"
- Hook → Problem → Solution → Key insight → Mess (brief) → Proud moment → Meta/dogfooding → CTA.
- Keep technical detail light — enough to intrigue, not enough to explain. Slides do the explaining.
- Tone: down to earth, developer journey, honest. Not a press release.

See `docs/marketing/design-system.json` → `shared.contentStrategy` for the full spec.

## Brand Context

Two design brands exist in this workspace — same structural patterns, different palettes:

| | **Personal (Lim Chun Zhe)** | **corvia** |
|--|--|--|
| Use for | LinkedIn posts, developer journal | Product docs, README, brand guide |
| Primary bg | Dark slate `#0f172a` | Navy `#0f1b33` |
| Accents | Multi-color (blue, cyan, green, amber, red, purple) | Gold `#D4A832` / `#E8C44E` |
| Body font | Inter | Inter |
| Brand font | — | Poppins 700/800 |
| Tone | Discovery, journey, honest | Professional, restrained |
| Design file | `docs/marketing/design-system.json` | `repos/corvia/docs/brand/corvia-brand-guide.html` |

When creating content, confirm which brand FIRST. The design system JSON has both palettes under `brands.personal` and `brands.corvia`, with shared layout/typography under `shared`.

## Pre-flight Checklist

Run these BEFORE any visual content work:

```bash
# 1. Start HTTP server from WORKSPACE ROOT (not subdirectory!)
#    Why root: URL paths match repo-relative file paths
#    Trap: starting from subdirectory → 404 on all assets
python3 -m http.server 8099 --bind 127.0.0.1 &
# or: npx http-server /workspaces/corvia-workspace -p 8099 &

# 2. Verify server responds
curl -sf http://localhost:8099/ | head -5

# 3. Ensure Playwright + Chromium are available
cd tools/carousel-render && npm install && npx playwright install chromium
```

**Why these steps matter:** Every one of these caused a retry in production. file:// URLs are blocked by Playwright MCP security policy. Wrong server root causes 404s. Missing Chromium causes browser launch failures.

## Rendering Workflow

### Use standalone Node.js scripts, NOT Playwright MCP browser_run_code

`browser_run_code` runs inside the browser context — no `require()`, no `fs`, no Node.js APIs. Always use a standalone script.

**Shared render tool:** `tools/carousel-render/render.js`

```bash
node tools/carousel-render/render.js \
  --url http://localhost:8099/path/to/carousel.html \
  --outdir docs/marketing/linkedin/YYYY-MM-DD-slug
```

The script includes:
- **Pre-flight checks** — verifies HTTP server is reachable and Chromium is installed
- **Retina rendering** — 2x deviceScaleFactor for high-res output
- **PDF generation** — multi-page PDF with print backgrounds
- **Automated validation** — overflow detection, font size audit, content height budget
- **Pass/fail report** — actionable messages for any issues found

### Before final render: Content Verification (MANDATORY)

Run an **independent agent** using `.agents/skills/content-verification.md` to fact-check all claims. This agent must:
1. Extract every factual claim from post.txt and carousel HTML
2. Verify locally (corvia knowledge base, source code, git history)
3. Verify online (web search, official docs, look for counterexamples)
4. Check fairness (no strawman, no false precision, no overgeneralization)
5. Produce a pass/fail report with specific fixes

**Do not render final PNGs/PDF until verification passes.** Stats without evidence must be qualified ("in my experience", "roughly") or removed.

See `.agents/skills/content-verification.md` for the full verification protocol.

### After rendering, verify visually

Use Playwright MCP `browser_navigate` + `browser_take_screenshot` to spot-check slides, or read the PNG files directly if the AI supports image input.

## LinkedIn Carousel Specs

| Property | Value |
|----------|-------|
| Canvas | 1080 x 1350px (4:5 ratio) |
| deviceScaleFactor | 2 (renders 2160x2700 retina) |
| Padding | 72px all sides |
| Content safe area | 936px wide x ~1110px tall |
| Author tag | absolute, bottom: 48px, left: 72px, 18px |
| Slide number | absolute, bottom: 48px, right: 72px, 18px |

**Content height budget** (must total <= 1350px):
```
72 (top pad) + slide-label (~42) + title (~200) + subtitle (~72)
+ content area + author/number zone (~96) + 72 (bottom pad)
= available content ≈ 796px
```

## Phone Readability

1080px canvas / ~375px phone width = **0.35x scale factor**

| Role | Size | On phone | Notes |
|------|------|----------|-------|
| Cover title | 68px | ~24px | Large, attention-grabbing |
| Slide title | 58px | ~20px | Primary heading |
| Subtitle | 28px | ~10px | Minimum comfortable reading |
| Body | 20-24px | ~7-8px | Limit of readability |
| Secondary | 18px | ~6px | Labels, descriptions |
| Metadata | 15-16px | ~5px | Borderline — non-essential only |

**NEVER go below 15px.** Never adjust all fonts at once for "phone readability" — this kills visual hierarchy. Adjust by role, preserving the ratio between levels.

## Common CSS Pitfalls

### Overflow & clipping
- **Flex children**: always add `min-width: 0` to prevent content overflow
- **Stat values**: never use `white-space: nowrap` — causes horizontal overflow
- **Absolute positioning**: check elements don't overlap with growing content (e.g. stat boxes overlapping author tag)

### Layout choices
- **Grid > flex** for stat boxes at 936px width. `grid-template-columns: 1fr 1fr` is safer than `display: flex` with hardcoded widths
- **Content height**: font-size increase of 4px across all body text adds ~20% total height. Re-check safe area after any font change

### Font hierarchy preservation
- Titles: 58-68px (bold, high contrast)
- Subtitles: 28px (medium weight, secondary color)
- Body: 20-24px (regular weight)
- Labels/secondary: 18px (uppercase/muted)
- Metadata: 15-16px (muted color)

Changing one level without adjusting others breaks the hierarchy. Always maintain at least 1.3x ratio between adjacent levels.

## Slide Templates (Component Registry)

Instead of generating arbitrary HTML, compose slides from canonical templates defined in `docs/marketing/design-system.json` under `shared.slideTemplates`. Each template specifies structure, content constraints, and height budgets.

**Available templates:**

| Template | Use for | Key constraint |
|----------|---------|----------------|
| `cover` | Slide 1 — hook the reader | Max 4 title lines, gradient bg, no cards |
| `setup` | Context — what you were building | 3 card row + 2 info cards |
| `comparison` | Side-by-side broken vs correct | Max 2 pairs, flex children need min-width:0 |
| `category-grid` | 4 categories in 2x2 grid | Optional stat row + quote card |
| `diagram` | Flow diagrams, process comparisons | Vertical flows in 2 columns, no absolute positioning |
| `solution` | Present the fix | Pipeline (2 rows of 3) + code block + summary |
| `before-after` | Workflow before/after | 2 comparison cards + optional related topics |
| `takeaway` | Final slide — lessons + CTA | Max 4 takeaways, closing gradient bg |
| `stat-highlight` | Big numbers that tell a story | Grid 2-col only, never 3-col flex |

**Narrative arc** (recommended sequence):
1. `cover` — hook
2. `setup` — context
3. `comparison` / `category-grid` — the problem
4. `diagram` / `stat-highlight` — the pattern
5. `solution` — the fix
6. `before-after` — the difference
7. `takeaway` — lessons + CTA

6-8 slides is the sweet spot. Read the full template specs in the design system JSON before generating HTML.

**Self-sufficiency checklist (every slide):**
- Does this slide make sense if the reader hasn't read the post description? If not, add context.
- Is there a subtitle or insight box explaining "why this matters"? If not, add one.
- Would a first-time viewer know what the project is? Cover slide must name the project and what it does.
- Is jargon explained in-slide? (e.g., "OpenVINO EP" should be accompanied by "hardware backend" or similar)

## Anti-patterns

| Don't | Why | Do instead |
|-------|-----|-----------|
| Use `file://` URLs with Playwright MCP | Blocked by security policy | Start HTTP server |
| Use `browser_run_code` for rendering | No Node.js APIs in browser context | Standalone Node.js script |
| Adjust all fonts at once | Kills visual hierarchy | Adjust by role, preserve ratios |
| Use `white-space: nowrap` on stat values | Horizontal overflow | Let values wrap or abbreviate |
| Use absolute positioning for growable content | Overlaps when content changes | Use flex/grid flow |
| Skip visual verification | Overflow is invisible in code | Screenshot every slide |
| Start HTTP server from subdirectory | 404 errors on asset paths | Always serve from workspace root |
| Mix corvia and personal brand colors | Confuses brand identity | Check brand context first |

## Validation Checklist (post-render)

1. Each slide PNG exists and is > 10KB
2. Dimensions are 2160x2700 (2x retina of 1080x1350)
3. No text visually overflows its container
4. Font hierarchy is maintained (titles > subtitles > body > labels)
5. Author tag and slide numbers are visible and not overlapped
6. Content doesn't extend into the bottom 96px zone
7. All slides look correct on visual inspection
8. Correct brand palette used (personal vs corvia)

---

## Lessons Learned (queryable)

Real problems encountered during carousel production, structured so AI agents can find specific answers. Each lesson was learned from an actual failure, not theory.

### Q: Why can't Playwright MCP open my HTML file directly?
**Playwright MCP blocks `file://` URLs** (security policy). You must serve files over HTTP. Start a server from the workspace root: `python3 -m http.server 8099 --bind 127.0.0.1`. Starting from the workspace root is critical — if you start from a subdirectory, all URL paths break with 404 errors.

### Q: Why does browser_run_code fail with "require is not defined"?
**`browser_run_code` runs inside the browser**, not Node.js. There's no `require()`, no `fs`, no process. For anything that needs Node.js APIs (rendering, file I/O), write a standalone `.js` script and run it with `node`.

### Q: I made text bigger for phone readability but now the hierarchy looks wrong. What happened?
**You adjusted all fonts at once.** This is the #1 design mistake. Phone readability requires maintaining the *ratio* between levels, not setting a uniform minimum. A 68px title and 28px subtitle have a 2.4x ratio — that's what creates hierarchy. If you bump everything to 24px+, the ratio collapses and nothing stands out. Fix: adjust each role independently, keeping >= 1.3x between adjacent levels.

### Q: My content overflows the slide. How do I debug it?
**Content height budget.** A 1350px slide has ~796px available for content after padding (72px top/bottom), slide label (~42px), title (~200px), subtitle (~72px), and footer zone (~96px). Common causes: too many cards, font sizes too large (4px increase = ~20% more height), too much text per card. Fix: reduce card count, trim text, or use smaller body fonts (20px instead of 24px).

### Q: Flex items are overflowing horizontally. What's wrong?
**Missing `min-width: 0`** on flex children. By default, flex items won't shrink below their content size. Adding `min-width: 0` allows them to shrink. Also avoid `white-space: nowrap` on stat values — it prevents text wrapping and causes overflow.

### Q: Stat boxes overlap the author tag at the bottom of the slide.
**Absolute positioning conflict.** The author tag is absolute-positioned at `bottom: 48px`. If content (stat boxes, cards) grows tall enough, it overlaps the author zone. Fix: use flex layout and ensure total content height stays within the safe area. The render script validates this automatically.

### Q: Elements look aligned in code but the rendered output is misaligned.
**This is the core problem this workflow solves.** CSS is syntactically correct but the visual output is wrong — AI can't detect this from code alone. The fix is the render → screenshot → verify loop. Use `tools/carousel-render/render.js` which validates automatically, or use Playwright MCP to screenshot each slide and visually inspect.

### Q: The render script says "page.evaluate: too many arguments".
**Playwright's `page.evaluate()` only accepts one argument.** Wrap multiple parameters in an object: `page.evaluate(({ a, b }) => ..., { a, b })` instead of `page.evaluate((a, b) => ..., a, b)`.

### Q: How do I know which brand to use for content?
**Personal content** (LinkedIn posts about your developer journey, discoveries, opinions) uses the `personal` brand: dark slate background, multi-color accents, Inter font, discovery tone. **corvia content** (product docs, brand guide, README) uses the `corvia` brand: navy + gold, Poppins for brand name, professional tone. Check `docs/marketing/design-system.json` for both palettes.

### Q: Can I use React/Vite/Slidev instead of plain HTML?
**Plain HTML/CSS is the right choice for LinkedIn carousels.** React/Vite add build complexity without layout benefits. Slidev and Marp are locked to 16:9 and can't produce 4:5 carousels. AI generates HTML/CSS more reliably than JSX component trees. The leverage comes from the design system + this skill file constraining the AI, not from the rendering framework.

### Q: Is there a LinkedIn post about this workflow itself?
**Yes — the "AI blind designer" post** (`docs/marketing/linkedin/2026-03-11-ai-blind-designer/`) covers exactly this discovery: AI can code UI but can't see it, Playwright MCP gives it eyes. The carousel + post text are in that directory. The meta-journey of *building* the render tooling (this skill file, the design system, the validation script) could be a follow-up post — the story of giving AI structure to work within.
