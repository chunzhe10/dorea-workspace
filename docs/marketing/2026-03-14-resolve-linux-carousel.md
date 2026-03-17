# DaVinci Resolve Linux Carousel Implementation Plan

**Goal:** Build an 8-slide LinkedIn carousel telling the story of installing DaVinci Resolve on Linux with Claude as copilot, using a wall-breaker visual narrative with richer CSS illustrations.

**Architecture:** Single HTML file with all 8 slides, CSS-only illustrations (brick walls, chat bubbles, underwater scene), rendered via Playwright to PNG + PDF. Follows established carousel patterns from `dual-gpu-inference-carousel.html`.

**Tech Stack:** HTML, CSS (no JS), Playwright (render), Node.js (render script)

**Spec:** `docs/marketing/2026-03-14-resolve-linux-carousel-design.md`
**Reference carousel:** `docs/marketing/linkedin/2026-03-13-dual-gpu-inference/dual-gpu-inference-carousel.html`
**Design system:** `docs/marketing/design-system.json`

---

## File Structure

| File | Purpose |
|------|---------|
| Create: `docs/marketing/linkedin/2026-03-14-resolve-linux-setup/resolve-linux-carousel.html` | The carousel (all 8 slides + CSS) |
| Create: `docs/marketing/linkedin/2026-03-14-resolve-linux-setup/output/` | Rendered PNGs + PDF |
| Existing: `tools/carousel-render/render.js` | Render + validate script |
| Existing: `docs/marketing/linkedin/2026-03-14-resolve-linux-setup/post.html` | Companion post (already written) |

---

## Content Integrity Rules (MUST follow)

These apply to ALL slide text. Violating any of these is a bug:

- No specific library names (libpango, libglib, etc.)
- No exact error log strings or stack traces
- No technical explanations of WHY Wayland breaks things
- Frame all technical details as discovery: "turns out," "traced back to"
- Reference "DaVinci Resolve" never "DaVinci Resolve Studio"
- Confirmed facts only: GPU memory error message, Wayland root cause, 3 env vars forcing X11, full 4K playback, Claude misdiagnosis, user pushed back

---

## Chunk 1: Base Structure + Slides 1-2

### Task 1: Create HTML skeleton with base CSS

**Files:**
- Create: `docs/marketing/linkedin/2026-03-14-resolve-linux-setup/resolve-linux-carousel.html`

Copy the base structure from `docs/marketing/linkedin/2026-03-13-dual-gpu-inference/dual-gpu-inference-carousel.html` lines 1-120 (everything up to the first slide). This gives you:

- Google Fonts import (Inter, JetBrains Mono, Poppins)
- CSS custom properties (all brand colors)
- `.slide` base (1080x1350, padding 72px, flex column, overflow hidden)
- `@page` and `@media print` rules
- `.slide::before` radial blue glow
- `.slide-number`, `.author-tag`, `.slide-label` positioning
- `.slide-title` (update to 58px per design system — reference carousel used 52px which drifted)
- `.card`, `.terminal`, `.stat-box`, `.insight-box` components
- Highlight color classes

Then add these NEW CSS classes specific to this carousel:

- [ ] **Step 1: Create the HTML file with base CSS**

```css
/* === WALL-BREAKER SPECIFIC STYLES === */

/* Cover title at 68px per design system */
.cover-title {
  font-size: 68px; font-weight: 800; line-height: 1.12;
  letter-spacing: -1.5px; margin-bottom: 24px;
}

/* Correct slide title to 58px (not 52px) */
.slide-title {
  font-size: 58px; font-weight: 800; line-height: 1.15;
  margin-bottom: 32px; letter-spacing: -1px;
}

/* Correct subtitle to 28px per design system (reference used 24px) */
.slide-subtitle {
  font-size: 28px; font-weight: 500; line-height: 1.5;
  color: var(--text-secondary); margin-bottom: 40px;
}

/* Correct slide-label to 15px (reference used 14px, below minimum) */
.slide-label {
  font-size: 15px; font-weight: 700; letter-spacing: 3px;
  text-transform: uppercase; color: var(--accent-blue); margin-bottom: 24px;
}

/* Correct footer elements to 18px per design system */
.author-tag {
  position: absolute; bottom: 48px; left: 72px;
  font-size: 18px; font-weight: 500; color: var(--text-muted);
}
.slide-number {
  position: absolute; bottom: 48px; right: 72px;
  font-size: 18px; font-weight: 600; color: var(--text-muted); letter-spacing: 2px;
}

/* Brick wall pattern */
.brick-wall {
  background:
    repeating-linear-gradient(
      0deg,
      transparent,
      transparent 28px,
      rgba(239, 68, 68, 0.3) 28px,
      rgba(239, 68, 68, 0.3) 30px
    ),
    repeating-linear-gradient(
      90deg,
      transparent,
      transparent 58px,
      rgba(239, 68, 68, 0.3) 58px,
      rgba(239, 68, 68, 0.3) 60px
    );
  background-color: rgba(239, 68, 68, 0.08);
  border: 2px solid rgba(239, 68, 68, 0.3);
  border-radius: 12px;
}

/* Offset alternate brick rows */
.brick-wall::after {
  content: '';
  position: absolute; inset: 0;
  background:
    repeating-linear-gradient(
      90deg,
      transparent,
      transparent 58px,
      rgba(239, 68, 68, 0.2) 58px,
      rgba(239, 68, 68, 0.2) 60px
    );
  background-position: 30px 15px;
  background-size: 60px 60px;
  pointer-events: none;
}

/* Chat bubbles */
.chat-bubble {
  padding: 16px 24px; border-radius: 16px;
  font-size: 20px; font-weight: 500; line-height: 1.4;
  max-width: 400px; position: relative;
}
.chat-bubble.claude {
  background: rgba(59, 130, 246, 0.12);
  border: 1px solid rgba(59, 130, 246, 0.3);
  color: #60a5fa;
}
.chat-bubble.claude::before {
  content: 'Claude'; display: block;
  font-size: 15px; font-weight: 700; letter-spacing: 1px;
  text-transform: uppercase; margin-bottom: 6px;
  color: var(--accent-blue);
}
.chat-bubble.user {
  background: rgba(139, 92, 246, 0.12);
  border: 1px solid rgba(139, 92, 246, 0.3);
  color: #a78bfa;
}
.chat-bubble.user::before {
  content: 'You'; display: block;
  font-size: 15px; font-weight: 700; letter-spacing: 1px;
  text-transform: uppercase; margin-bottom: 6px;
  color: var(--accent-purple);
}

/* Symptom block (rising from foundation) */
.symptom-block {
  background: var(--bg-card);
  border-radius: 16px; padding: 20px 28px;
  display: flex; align-items: center; gap: 16px;
  min-width: 0;
}

/* Foundation slab */
.foundation {
  background: rgba(239, 68, 68, 0.1);
  border: 2px solid rgba(239, 68, 68, 0.35);
  border-radius: 12px; padding: 20px 32px;
  display: flex; justify-content: space-between; align-items: center;
}

/* Film frame sprocket holes */
.film-frame {
  position: relative; border: 3px solid rgba(241, 245, 249, 0.2);
  border-radius: 8px; padding: 40px 60px;
}
.film-frame::before,
.film-frame::after {
  content: '';
  position: absolute; top: 20px; bottom: 20px; width: 24px;
  background: repeating-linear-gradient(
    180deg,
    rgba(241, 245, 249, 0.15) 0px,
    rgba(241, 245, 249, 0.15) 12px,
    transparent 12px,
    transparent 24px
  );
  border-radius: 3px;
}
.film-frame::before { left: 12px; }
.film-frame::after { right: 12px; }

/* Tags */
.tags { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
.tag {
  font-size: 20px; font-weight: 600; padding: 8px 18px;
  border-radius: 8px; background: rgba(59, 130, 246, 0.15);
  color: var(--accent-blue);
}

/* Cover gradient */
.slide.cover {
  background: linear-gradient(160deg, #0f172a 0%, #1a1a3e 50%, #0f172a 100%);
}
/* Closing gradient */
.slide.closing {
  background: linear-gradient(160deg, #0f172a 0%, #1a2332 50%, #0f172a 100%);
}

/* Green glow for breakthroughs */
.green-glow {
  box-shadow: 0 0 60px rgba(16, 185, 129, 0.15), 0 0 120px rgba(16, 185, 129, 0.05);
}

/* Underwater gradient for slide 7 */
.slide.underwater {
  background: linear-gradient(
    180deg,
    #22d3ee 0%,
    #1a8a8a 15%,
    #0d3b66 50%,
    #0a1628 100%
  );
}

/* Bubble particles */
.bubble {
  position: absolute; border-radius: 50%;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Glass card */
.glass-card {
  background: rgba(255, 255, 255, 0.06);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 16px; padding: 24px 28px;
}

/* VRAM bar */
.vram-bar {
  height: 32px; border-radius: 8px; overflow: hidden;
  background: var(--bg-card); border: 1px solid var(--border);
  display: flex;
}
.vram-used {
  background: rgba(239, 68, 68, 0.4); height: 100%;
}
.vram-free {
  background: rgba(16, 185, 129, 0.2); height: 100%; flex: 1;
}
```

- [ ] **Step 2: Verify file loads in browser**

Run:
```bash
cd /workspaces/corvia-workspace && npx http-server -p 8099 -c-1 &
```
Open: `http://localhost:8099/docs/marketing/linkedin/2026-03-14-resolve-linux-setup/resolve-linux-carousel.html`
Expected: blank dark page (no slides yet)

### Task 2: Build Slide 1 (Cover)

**Files:**
- Modify: `docs/marketing/linkedin/2026-03-14-resolve-linux-setup/resolve-linux-carousel.html`

- [ ] **Step 1: Add Slide 1 HTML**

Add after `</style></head><body>`:

```html
<!-- Slide 1: COVER -->
<div class="slide cover">
  <div class="slide-label">A Developer's Discovery</div>
  <div style="font-size: 80px; margin-bottom: 20px;">🎬</div>
  <div class="film-frame">
    <div class="cover-title">
      I Installed <span class="highlight">DaVinci Resolve</span> on Linux.<br>
      Here's Every Wall I Hit.
    </div>
  </div>
  <div class="slide-subtitle" style="margin-top: 28px;">
    Ubuntu 24.04 + Wayland + NVIDIA.<br>With Claude as my copilot.
  </div>
  <div class="tags">
    <span class="tag">Ubuntu 24.04</span>
    <span class="tag" style="background: rgba(245, 158, 11, 0.15); color: var(--accent-amber);">Wayland</span>
    <span class="tag" style="background: rgba(16, 185, 129, 0.15); color: var(--accent-green);">NVIDIA</span>
    <span class="tag" style="background: rgba(139, 92, 246, 0.15); color: var(--accent-purple);">Claude</span>
    <span class="tag" style="background: rgba(239, 68, 68, 0.15); color: var(--accent-red);">DaVinci Resolve</span>
  </div>
  <div class="author-tag">Lim Chun Zhe</div>
  <div class="slide-number">01 / 08</div>
</div>
```

- [ ] **Step 2: Visual check via Playwright**

Use Playwright MCP `browser_navigate` to the URL, then `browser_take_screenshot`. Verify:
- Film frame visible with sprocket holes
- Title is 68px, readable
- Tags visible with distinct colors
- Gradient background visible
- Author tag and slide number positioned correctly

### Task 3: Build Slide 2 (The Setup)

- [ ] **Step 1: Add Slide 2 HTML**

```html
<!-- Slide 2: THE SETUP -->
<div class="slide">
  <div class="slide-label">The Setup</div>
  <div class="slide-title">Linux for Engineering Was Working.<br><span class="highlight">Now: Creative Work.</span></div>

  <div style="display: flex; gap: 16px; align-items: center; margin-bottom: 28px;">
    <!-- Achievement cards -->
    <div class="card" style="flex: 1; min-width: 0; border-top: 3px solid var(--accent-green); padding: 24px;">
      <div style="font-size: 32px; margin-bottom: 8px;">✅</div>
      <div style="font-size: 20px; font-weight: 700; color: var(--accent-green); margin-bottom: 4px;">Low RAM Baseline</div>
      <div style="font-size: 18px; color: var(--text-secondary);">1GB idle vs 5GB</div>
    </div>
    <div class="card" style="flex: 1; min-width: 0; border-top: 3px solid var(--accent-green); padding: 24px;">
      <div style="font-size: 32px; margin-bottom: 8px;">✅</div>
      <div style="font-size: 20px; font-weight: 700; color: var(--accent-green); margin-bottom: 4px;">Native Docker</div>
      <div style="font-size: 18px; color: var(--text-secondary);">No VM layer</div>
    </div>
    <div class="card" style="flex: 1; min-width: 0; border-top: 3px solid var(--accent-green); padding: 24px;">
      <div style="font-size: 32px; margin-bottom: 8px;">✅</div>
      <div style="font-size: 20px; font-weight: 700; color: var(--accent-green); margin-bottom: 4px;">iGPU Assignment</div>
      <div style="font-size: 18px; color: var(--text-secondary);">DRI_PRIME control</div>
    </div>

    <!-- Arrow -->
    <div style="font-size: 28px; color: var(--text-muted);">→</div>

    <!-- Question card -->
    <div class="card" style="flex: 1; min-width: 0; border-top: 3px solid var(--accent-amber); padding: 24px;">
      <div style="font-size: 32px; margin-bottom: 8px;">❓</div>
      <div style="font-size: 20px; font-weight: 700; color: var(--accent-amber); margin-bottom: 4px;">Video Editing?</div>
      <div style="font-size: 18px; color: var(--text-secondary);">Can I edit here?</div>
    </div>
  </div>

  <div style="font-size: 24px; color: var(--text-secondary); line-height: 1.5; max-width: 800px;">
    A few days ago I moved to Linux for engineering. The benefits were clear. Now the real test: can I run a full creative workflow?
  </div>

  <div class="author-tag">Lim Chun Zhe</div>
  <div class="slide-number">02 / 08</div>
</div>
```

- [ ] **Step 2: Visual check**

Playwright screenshot. Verify: 4 cards in a row, green checkmarks on first 3, amber question mark on last, arrow between them, text readable.

- [ ] **Step 3: Commit slides 1-2**

```bash
git add docs/marketing/linkedin/2026-03-14-resolve-linux-setup/resolve-linux-carousel.html
git commit -m "feat(carousel): add resolve-linux carousel skeleton + slides 1-2 (cover, setup)"
```

---

## Chunk 2: Slides 3-4 (Adversary + Claude Steps In)

### Task 4: Build Slide 3 (The Adversary)

- [ ] **Step 1: Add Slide 3 HTML**

```html
<!-- Slide 3: THE ADVERSARY -->
<div class="slide">
  <div class="slide-label" style="color: var(--accent-red);">The Adversary</div>
  <div class="slide-title"><span class="highlight-red">Wayland + Ubuntu 24.04</span></div>
  <div class="slide-subtitle">One Root Cause. Many Symptoms.</div>

  <!-- Symptom blocks rising upward -->
  <div style="display: flex; flex-direction: column; gap: 14px; margin-bottom: 28px; flex: 1;">
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
      <div class="symptom-block" style="border-left: 4px solid var(--accent-red);">
        <span style="font-size: 28px;">💥</span>
        <div>
          <div style="font-size: 22px; font-weight: 700; color: var(--text-primary);">Dependency conflicts</div>
          <div style="font-size: 18px; color: var(--text-secondary);">Bundled vs system libs</div>
        </div>
      </div>
      <div class="symptom-block" style="border-left: 4px solid var(--accent-amber);">
        <span style="font-size: 28px;">🔗</span>
        <div>
          <div style="font-size: 22px; font-weight: 700; color: var(--text-primary);">Library mismatches</div>
          <div style="font-size: 18px; color: var(--text-secondary);">Renamed with t64 suffixes</div>
        </div>
      </div>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
      <div class="symptom-block" style="border-left: 4px solid var(--accent-red);">
        <span style="font-size: 28px;">🖥️</span>
        <div>
          <div style="font-size: 22px; font-weight: 700; color: var(--text-primary);">GPU errors</div>
          <div style="font-size: 18px; color: var(--text-secondary);">Misleading error messages</div>
        </div>
      </div>
      <div class="symptom-block" style="border-left: 4px solid var(--accent-amber);">
        <span style="font-size: 28px;">📦</span>
        <div>
          <div style="font-size: 22px; font-weight: 700; color: var(--text-primary);">Missing packages</div>
          <div style="font-size: 18px; color: var(--text-secondary);">Not in default repos</div>
        </div>
      </div>
    </div>

    <!-- Crack lines (decorative) -->
    <div style="display: flex; justify-content: center; margin: 8px 0;">
      <div style="width: 2px; height: 40px; background: linear-gradient(180deg, rgba(239,68,68,0.4), rgba(239,68,68,0.1));"></div>
    </div>

    <!-- Foundation slab -->
    <div class="foundation">
      <div style="font-size: 22px; font-weight: 800; color: var(--accent-red); letter-spacing: 2px; text-transform: uppercase;">Wayland</div>
      <div style="font-size: 18px; color: var(--text-muted);">Root causes</div>
      <div style="font-size: 22px; font-weight: 800; color: var(--accent-red); letter-spacing: 2px; text-transform: uppercase;">Ubuntu 24.04</div>
    </div>
  </div>

  <div style="font-size: 22px; color: var(--text-secondary); line-height: 1.5;">
    What should be "install and go" turned into a debugging session. Most issues traced back to the same root causes.
  </div>

  <div class="author-tag">Lim Chun Zhe</div>
  <div class="slide-number">03 / 08</div>
</div>
```

- [ ] **Step 2: Visual check**

Verify: 2x2 grid of symptom blocks, crack line connecting to foundation slab, red foundation with "Wayland" and "Ubuntu 24.04" labels.

### Task 5: Build Slide 4 (Claude Steps In)

- [ ] **Step 1: Add Slide 4 HTML**

```html
<!-- Slide 4: CLAUDE STEPS IN -->
<div class="slide">
  <div class="slide-label">Claude Steps In</div>
  <div class="slide-title">Every Wall, <span class="highlight">a Fix</span></div>
  <div class="slide-subtitle">Read. Search. Solve. Repeat.</div>

  <!-- Three mini walls in sequence -->
  <div style="display: flex; gap: 20px; flex: 1; align-items: stretch; margin-bottom: 20px;">

    <!-- Wall 1: mostly intact -->
    <div style="flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 12px;">
      <div class="brick-wall" style="position: relative; flex: 1; display: flex; align-items: center; justify-content: center; min-height: 200px;">
        <div style="font-size: 40px; opacity: 0.6;">🧱</div>
        <!-- Small crack -->
        <div style="position: absolute; bottom: 30%; right: 20%; width: 30px; height: 2px; background: rgba(16,185,129,0.4); transform: rotate(-30deg);"></div>
      </div>
      <div class="chat-bubble claude">Read the error</div>
    </div>

    <!-- Arrow -->
    <div style="display: flex; align-items: center; font-size: 28px; color: var(--text-muted);">→</div>

    <!-- Wall 2: large cracks -->
    <div style="flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 12px;">
      <div class="brick-wall" style="position: relative; flex: 1; display: flex; align-items: center; justify-content: center; min-height: 200px; opacity: 0.7;">
        <div style="font-size: 40px; opacity: 0.4;">🧱</div>
        <!-- Cracks -->
        <div style="position: absolute; top: 20%; left: 15%; width: 50px; height: 2px; background: rgba(16,185,129,0.5); transform: rotate(15deg);"></div>
        <div style="position: absolute; bottom: 25%; right: 10%; width: 40px; height: 2px; background: rgba(16,185,129,0.5); transform: rotate(-20deg);"></div>
        <div style="position: absolute; top: 50%; left: 40%; width: 60px; height: 2px; background: rgba(16,185,129,0.5); transform: rotate(5deg);"></div>
      </div>
      <div class="chat-bubble claude">Searched known issues</div>
    </div>

    <!-- Arrow -->
    <div style="display: flex; align-items: center; font-size: 28px; color: var(--text-muted);">→</div>

    <!-- Wall 3: crumbled -->
    <div style="flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 12px;">
      <div style="position: relative; flex: 1; display: flex; align-items: flex-end; justify-content: center; min-height: 200px;" class="green-glow">
        <!-- Rubble pieces -->
        <div style="display: flex; gap: 6px; align-items: flex-end; padding-bottom: 16px;">
          <div style="width: 30px; height: 18px; background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.2); border-radius: 3px; transform: rotate(-8deg);"></div>
          <div style="width: 24px; height: 14px; background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.15); border-radius: 3px; transform: rotate(12deg);"></div>
          <div style="width: 28px; height: 16px; background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.15); border-radius: 3px; transform: rotate(-5deg);"></div>
          <div style="width: 20px; height: 12px; background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.1); border-radius: 3px; transform: rotate(15deg);"></div>
        </div>
      </div>
      <div class="chat-bubble claude">Wrote the fix</div>
    </div>
  </div>

  <div class="insight-box">
    <p>Each fix revealed the next wall. Wayland kept being the common thread.</p>
  </div>

  <div class="author-tag">Lim Chun Zhe</div>
  <div class="slide-number">04 / 08</div>
</div>
```

- [ ] **Step 2: Visual check**

Verify: 3 walls in a row showing progression (intact → cracked → crumbled), Claude chat bubbles below each, arrows between them, green glow on the crumbled wall, insight box at bottom.

- [ ] **Step 3: Commit slides 3-4**

```bash
git add docs/marketing/linkedin/2026-03-14-resolve-linux-setup/resolve-linux-carousel.html
git commit -m "feat(carousel): add slides 3-4 (adversary, claude steps in)"
```

---

## Chunk 3: Slides 5-6 (The Big One + The Reveal)

### Task 6: Build Slide 5 (The Big One)

- [ ] **Step 1: Add Slide 5 HTML**

```html
<!-- Slide 5: THE BIG ONE -->
<div class="slide">
  <div class="slide-label" style="color: var(--accent-amber);">The Big One</div>
  <div class="slide-title"><span class="highlight-red">"GPU Memory Is Full"</span></div>
  <div class="slide-subtitle">It Wasn't.</div>

  <!-- VRAM bar -->
  <div style="margin-bottom: 12px;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
      <span style="font-size: 16px; color: var(--text-muted);">GPU VRAM Usage</span>
      <span style="font-size: 16px; color: var(--accent-green);">4.8 GB free of 6 GB</span>
    </div>
    <div class="vram-bar">
      <div class="vram-used" style="width: 20%;"></div>
      <div class="vram-free"></div>
    </div>
  </div>

  <!-- Massive wall -->
  <div class="brick-wall" style="position: relative; padding: 40px; display: flex; align-items: center; justify-content: center; min-height: 240px; margin-bottom: 24px;">
    <div style="font-size: 42px; font-weight: 900; color: var(--accent-red); letter-spacing: 3px; text-transform: uppercase; text-align: center;">GPU MEMORY<br>FULL</div>
    <!-- Hairline crack at base -->
    <div style="position: absolute; bottom: 8px; left: 35%; width: 80px; height: 1px; background: rgba(16,185,129,0.3); transform: rotate(2deg);"></div>
  </div>

  <!-- Chat bubbles -->
  <div style="display: flex; flex-direction: column; gap: 16px;">
    <div class="chat-bubble claude" style="align-self: flex-start;">Looks like an OOM issue. Let's build proxies.</div>
    <div class="chat-bubble user" style="align-self: flex-end;">Something doesn't add up.</div>
  </div>

  <div style="font-size: 20px; color: var(--text-muted); text-align: center; margin-top: 20px;">
    We started solving the wrong problem.
  </div>

  <div class="author-tag">Lim Chun Zhe</div>
  <div class="slide-number">05 / 08</div>
</div>
```

- [ ] **Step 2: Visual check**

Verify: VRAM bar mostly green (free), massive wall with "GPU MEMORY FULL" in red, Claude bubble on left suggesting proxies, user bubble on right pushing back, hairline crack at base.

### Task 7: Build Slide 6 (The Reveal)

- [ ] **Step 1: Add Slide 6 HTML**

```html
<!-- Slide 6: THE REVEAL -->
<div class="slide">
  <div class="slide-label" style="color: var(--accent-green);">The Reveal</div>
  <div class="slide-title"><span class="highlight-red">Wayland</span> Again.</div>
  <div class="slide-subtitle">Wall Shattered.</div>

  <!-- Rubble from shattered wall with green glow -->
  <div style="position: relative; display: flex; justify-content: center; align-items: flex-end; height: 100px; margin-bottom: 24px;" class="green-glow">
    <div style="display: flex; gap: 8px; align-items: flex-end;">
      <div style="width: 40px; height: 22px; background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.2); border-radius: 4px; transform: rotate(-12deg);"></div>
      <div style="width: 30px; height: 16px; background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.15); border-radius: 3px; transform: rotate(8deg);"></div>
      <div style="width: 50px; height: 20px; background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.18); border-radius: 4px; transform: rotate(-5deg);"></div>
      <div style="width: 25px; height: 14px; background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.12); border-radius: 3px; transform: rotate(18deg);"></div>
      <div style="width: 35px; height: 18px; background: rgba(239,68,68,0.09); border: 1px solid rgba(239,68,68,0.15); border-radius: 3px; transform: rotate(-15deg);"></div>
    </div>
  </div>

  <!-- Wayland vs X11 comparison -->
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px;">
    <div class="card" style="border-top: 3px solid var(--accent-red); text-align: center; padding: 28px;">
      <div style="font-size: 36px; margin-bottom: 12px;">✕</div>
      <div style="font-size: 28px; font-weight: 800; color: var(--accent-red);">Wayland</div>
      <div style="font-size: 18px; color: var(--text-secondary); margin-top: 8px;">Display server broke Resolve</div>
    </div>
    <div class="card" style="border-top: 3px solid var(--accent-green); text-align: center; padding: 28px;">
      <div style="font-size: 36px; margin-bottom: 12px;">✓</div>
      <div style="font-size: 28px; font-weight: 800; color: var(--accent-green);">X11</div>
      <div style="font-size: 18px; color: var(--text-secondary); margin-top: 8px;">Force X11, everything works</div>
    </div>
  </div>

  <!-- Terminal fix -->
  <div class="terminal" style="margin-bottom: 20px;">
    <span class="comment"># Force X11 instead of Wayland</span><br>
    <span class="prompt">$</span> <span class="cmd">3 environment variables</span> <span class="highlight-val">+ /opt/resolve/bin/resolve</span>
  </div>

  <!-- Stat -->
  <div class="stat-box" style="text-align: center; border: 1px solid rgba(16,185,129,0.3); background: rgba(16,185,129,0.06);">
    <div style="font-size: 48px; font-weight: 800; color: var(--accent-green);">Full 4K Playback</div>
    <div style="font-size: 18px; color: var(--text-secondary); margin-top: 4px;">No proxies needed</div>
  </div>

  <div class="insight-box" style="margin-top: 16px;">
    <p>Turns out it wasn't a memory problem. It was a Wayland compatibility issue.</p>
  </div>

  <div class="author-tag">Lim Chun Zhe</div>
  <div class="slide-number">06 / 08</div>
</div>
```

- [ ] **Step 2: Visual check**

Verify: rubble with green glow, Wayland (red X) vs X11 (green check) comparison, terminal block with generic fix, "Full 4K Playback" stat in green, insight box.

- [ ] **Step 3: Commit slides 5-6**

```bash
git add docs/marketing/linkedin/2026-03-14-resolve-linux-setup/resolve-linux-carousel.html
git commit -m "feat(carousel): add slides 5-6 (the big one, the reveal)"
```

---

## Chunk 4: Slides 7-8 (What's Next + Takeaways)

### Task 8: Build Slide 7 (What's Next — Underwater)

- [ ] **Step 1: Add Slide 7 HTML**

```html
<!-- Slide 7: WHAT'S NEXT -->
<div class="slide underwater">
  <div class="slide-label" style="color: var(--accent-cyan);">What's Next</div>
  <div class="slide-title">Diving Trip. <span class="highlight-cyan">Resolve MCP.</span></div>
  <div class="slide-subtitle">Pieces Coming Together.</div>

  <!-- Bubble particles -->
  <div class="bubble" style="width: 8px; height: 8px; top: 180px; left: 120px; opacity: 0.12;"></div>
  <div class="bubble" style="width: 12px; height: 12px; top: 320px; right: 150px; opacity: 0.08;"></div>
  <div class="bubble" style="width: 6px; height: 6px; top: 450px; left: 300px; opacity: 0.15;"></div>
  <div class="bubble" style="width: 10px; height: 10px; top: 550px; right: 250px; opacity: 0.1;"></div>
  <div class="bubble" style="width: 5px; height: 5px; top: 700px; left: 200px; opacity: 0.18;"></div>
  <div class="bubble" style="width: 9px; height: 9px; top: 800px; right: 180px; opacity: 0.09;"></div>
  <div class="bubble" style="width: 7px; height: 7px; top: 650px; left: 500px; opacity: 0.14;"></div>
  <div class="bubble" style="width: 11px; height: 11px; top: 400px; left: 700px; opacity: 0.07;"></div>
  <div class="bubble" style="width: 4px; height: 4px; top: 900px; right: 400px; opacity: 0.2;"></div>

  <!-- Diving mask icon (simple geometric) -->
  <div style="display: flex; justify-content: center; margin: 20px 0;">
    <div style="position: relative;">
      <!-- Mask oval -->
      <div style="width: 100px; height: 60px; border: 3px solid rgba(6, 182, 212, 0.4); border-radius: 50%; background: rgba(6, 182, 212, 0.08);"></div>
      <!-- Snorkel -->
      <div style="position: absolute; top: -20px; right: -8px; width: 12px; height: 30px; border: 3px solid rgba(6, 182, 212, 0.4); border-radius: 6px 6px 0 0; background: transparent;"></div>
    </div>
  </div>

  <!-- Camera cards (glass-morphism) -->
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 28px;">
    <div class="glass-card" style="text-align: center;">
      <div style="font-size: 32px; margin-bottom: 8px;">📷</div>
      <div style="font-size: 22px; font-weight: 700; color: var(--text-primary);">DJI Action 4</div>
      <div style="font-size: 16px; color: var(--text-secondary);">Underwater action cam</div>
    </div>
    <div class="glass-card" style="text-align: center;">
      <div style="font-size: 32px; margin-bottom: 8px;">🔮</div>
      <div style="font-size: 22px; font-weight: 700; color: var(--text-primary);">Insta360 X5</div>
      <div style="font-size: 16px; color: var(--text-secondary);">360° underwater footage</div>
    </div>
  </div>

  <!-- Teaser pipeline -->
  <div style="display: flex; align-items: center; justify-content: center; gap: 16px; margin-bottom: 24px; opacity: 0.6;">
    <span style="font-size: 20px; color: var(--accent-cyan); font-weight: 600;">footage</span>
    <span style="font-size: 18px; color: var(--text-muted);">- - →</span>
    <span style="font-size: 20px; color: var(--accent-cyan); font-weight: 600;">AI</span>
    <span style="font-size: 18px; color: var(--text-muted);">- - →</span>
    <span style="font-size: 20px; color: var(--accent-cyan); font-weight: 600;">timeline</span>
  </div>

  <div style="font-size: 22px; color: var(--text-secondary); line-height: 1.5; text-align: center;">
    I love editing, but the repetitive parts eat time I'd rather spend diving.<br>
    Found a <span style="color: var(--accent-cyan); font-weight: 600;">DaVinci Resolve MCP</span>. Still brainstorming.
  </div>

  <div class="author-tag">Lim Chun Zhe</div>
  <div class="slide-number">07 / 08</div>
</div>
```

- [ ] **Step 2: Visual check**

Verify: underwater gradient (dark at bottom, teal/cyan at top), bubble particles scattered, diving mask icon centered, glass-morphism camera cards, dotted pipeline teaser faded, text readable against gradient.

### Task 9: Build Slide 8 (Takeaways)

- [ ] **Step 1: Add Slide 8 HTML**

```html
<!-- Slide 8: TAKEAWAYS -->
<div class="slide closing">
  <div class="slide-label">Key Takeaways</div>
  <div class="slide-title">What I Learned Setting Up <span class="highlight">Resolve</span> on Linux</div>

  <div style="flex: 1; display: flex; flex-direction: column; justify-content: center; gap: 0;">

    <!-- Takeaway 1 -->
    <div style="display: flex; gap: 20px; align-items: flex-start; padding: 24px 0;">
      <div style="font-size: 32px; flex-shrink: 0;">🔍</div>
      <div>
        <div style="font-size: 24px; font-weight: 700; color: var(--text-primary); margin-bottom: 6px;">Error messages lie.</div>
        <div style="font-size: 24px; font-weight: 500; color: var(--text-secondary); line-height: 1.5;">The GPU error pointed at memory. The real cause was the display server. Always read the logs.</div>
      </div>
    </div>

    <div style="height: 1px; background: var(--border); margin: 0 40px;"></div>

    <!-- Takeaway 2 -->
    <div style="display: flex; gap: 20px; align-items: flex-start; padding: 24px 0;">
      <div style="font-size: 32px; flex-shrink: 0;">🤝</div>
      <div>
        <div style="font-size: 24px; font-weight: 700; color: var(--text-primary); margin-bottom: 6px;">AI collaboration > AI answers.</div>
        <div style="font-size: 24px; font-weight: 500; color: var(--text-secondary); line-height: 1.5;">Claude got it wrong first. I steered us back. Neither would have solved it alone.</div>
      </div>
    </div>

    <div style="height: 1px; background: var(--border); margin: 0 40px;"></div>

    <!-- Takeaway 3 -->
    <div style="display: flex; gap: 20px; align-items: flex-start; padding: 24px 0;">
      <div style="font-size: 32px; flex-shrink: 0;">⚙️</div>
      <div>
        <div style="font-size: 24px; font-weight: 700; color: var(--text-primary); margin-bottom: 6px;">The engineering stack becomes the creative stack.</div>
        <div style="font-size: 24px; font-weight: 500; color: var(--text-secondary); line-height: 1.5;">Same Linux setup. Same hardware. Now running a full video editing workflow.</div>
      </div>
    </div>
  </div>

  <div style="height: 1px; background: var(--border); margin: 16px 0;"></div>

  <div style="text-align: center; padding: 8px 0;">
    <div style="font-size: 24px; font-weight: 600; color: var(--text-secondary); margin-bottom: 8px;">
      Anyone else using AI to streamline their video editing workflow?
    </div>
    <div style="font-size: 20px; color: var(--text-muted);">
      Curious what people have landed on.
    </div>
  </div>

  <div class="author-tag">Lim Chun Zhe</div>
  <div class="slide-number">08 / 08</div>
</div>
```

- [ ] **Step 2: Visual check**

Verify: closing gradient, 3 takeaways with emoji icons and dividers, CTA question at bottom, author tag and 08/08.

- [ ] **Step 3: Commit slides 7-8**

```bash
git add docs/marketing/linkedin/2026-03-14-resolve-linux-setup/resolve-linux-carousel.html
git commit -m "feat(carousel): add slides 7-8 (diving tease, takeaways)"
```

---

## Chunk 5: Render + Validate

### Task 10: Render PNGs and PDF

- [ ] **Step 1: Start HTTP server if not running**

```bash
cd /workspaces/corvia-workspace
npx http-server -p 8099 -c-1 &
```

- [ ] **Step 2: Run render script**

```bash
node tools/carousel-render/render.js \
  --url http://localhost:8099/docs/marketing/linkedin/2026-03-14-resolve-linux-setup/resolve-linux-carousel.html \
  --outdir docs/marketing/linkedin/2026-03-14-resolve-linux-setup/output
```

Expected: 8 PNG files (slide-01.png through slide-08.png) + carousel.pdf + validation report.

- [ ] **Step 3: Check validation output**

Verify ALL pass:
- All 8 PNGs exist and >10KB
- No text overflow on any slide
- No fonts below 15px minimum
- Content fits within safe area height

- [ ] **Step 4: Visual review of rendered PNGs**

Use Playwright or Read tool to inspect each rendered PNG. Check:
- Slide 1: film frame, gradient, tags readable
- Slide 4: brick walls visible, progression clear
- Slide 5: VRAM bar, chat bubbles distinguishable
- Slide 6: rubble, Wayland/X11 comparison, green stat
- Slide 7: underwater gradient renders, bubbles visible, glass cards readable

- [ ] **Step 5: Fix any issues found and re-render**

If validation fails or visual issues found, fix the HTML and re-run render.

- [ ] **Step 6: Final commit**

```bash
git add docs/marketing/linkedin/2026-03-14-resolve-linux-setup/
git commit -m "feat(carousel): render resolve-linux carousel PNGs + PDF"
```
