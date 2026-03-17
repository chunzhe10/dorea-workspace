# DaVinci Resolve on Linux — LinkedIn Carousel Design

**Date:** 2026-03-14
**Type:** LinkedIn carousel (8 slides, 1080x1350, 4:5 ratio)
**Visual direction:** Wall-breaker narrative with richer CSS illustrations
**Post:** `docs/marketing/linkedin/2026-03-14-resolve-linux-setup/post.html`
**Design system:** `docs/marketing/design-system.json`

## Story Arc

First-time DaVinci Resolve installation on Ubuntu 24.04 with Claude as copilot. Wayland + Ubuntu 24.04 as the central adversary causing cascading issues. Claude helps through each wall, gets the GPU error wrong, user steers to the real answer. Teases a diving trip and DaVinci Resolve MCP for future video editing workflow.

## Content Integrity Rules

- No specific library names (libpango, libglib, etc.) — user is unsure which they hit
- No exact error log strings — user unsure if they saw them or AI generated them
- No technical explanations of WHY Wayland breaks things (texture sharing, interop mechanisms)
- Frame all technical details as discovery: "turns out," "traced back to"
- Confirmed facts only: GPU memory error message, Wayland as root cause, 3 env vars forcing X11, full 4K playback, Claude initial misdiagnosis, user pushed back
- Reference DaVinci Resolve without "Studio" naming

## Typography (from design system)

- Cover title: 68px, weight 800, line-height 1.12
- Slide title: 58px, weight 800, line-height 1.15
- Subtitle: 28px, weight 500, line-height 1.5
- Body: 20-24px, weight 500, line-height 1.5
- Tags: 20px
- Labels: 18px, uppercase, muted color
- Terminal/code: 16-20px JetBrains Mono
- Minimum readable: 15px (never go below)
- All flex children MUST have `min-width: 0` to prevent overflow

## Visual Signature

- **Brick walls** (red/amber) that crack and crumble as problems are solved
- **Claude chat bubbles** (blue) showing AI actions
- **User chat bubbles** (purple) showing human judgment
- **Green glow/checkmarks** on breakthrough moments
- **Cracked red foundation** for Wayland/Ubuntu adversary
- **Underwater gradient** (deep navy → teal → cyan) for diving tease
- **Film frame** motif on cover (not clapperboard — simpler, more feasible in CSS)
- **Closing gradient** on slide 8: `#0f172a → #1a2332 → #0f172a` (closingSlide from design system)
- **Radial blue glow** `::before` pseudo-element on each slide (top-right corner, per existing carousels)
- **Slide labels** uppercase label above title on every slide (e.g., "The Setup," "The Adversary")
- **Slide numbers** in `0N / 08` format, bottom-right

## Slide Specifications

### Slide 1: COVER

**Title:** "I Installed DaVinci Resolve on Linux. Here's Every Wall I Hit."
**Subtitle:** "Ubuntu 24.04 + Wayland + NVIDIA. With Claude as my copilot."
**Tags:** Ubuntu 24.04, Wayland, NVIDIA, Claude, DaVinci Resolve
**Slide label:** "A Developer's Discovery"
**Layout:** cover template — gradient background (`#0f172a → #1a1a3e → #0f172a`)
**Visual:** CSS film frame element with sprocket holes along left/right edges (repeating radial-gradient circles). Title inside the frame. Emoji anchor: 🎬 (80px).
**Footer:** Author tag left, `01 / 08` right.

### Slide 2: THE SETUP

**Title:** "Linux for Engineering Was Working. Now: Creative Work."
**Layout:** Three "achievement" cards in a row + arrow + question card
**Visual:**
- 3 cards with green checkmarks: "Low RAM baseline" / "Native Docker" / "iGPU assignment"
- Each card: icon (32px emoji), title (20px bold green), brief description (18px secondary)
- Right arrow (→) leading to a fourth card with amber border and question mark icon
- Fourth card: "Video editing?" with "Can I edit here?" subtitle
**Slide label:** "The Setup"
**Text below cards:** "A few days ago I moved to Linux for engineering. The benefits were clear. Now the real test."
**Insight box:** None — clean setup slide.

### Slide 3: THE ADVERSARY

**Slide label:** "The Adversary"
**Title:** "Wayland + Ubuntu 24.04"
**Subtitle:** "One Root Cause. Many Symptoms."
**Layout:** Foundation + rising blocks diagram
**Visual:**
- Bottom: wide red foundation slab (full width, 80px tall) labeled "Wayland" on left, "Ubuntu 24.04" on right. Cracked texture (CSS border/gradient tricks for crack lines).
- Rising from foundation: 3-4 generic symptom blocks stacked/staggered upward:
  - "Dependency conflicts" (red border)
  - "Library mismatches" (amber border)
  - "GPU errors" (red border)
  - Optional: "Missing packages" (amber border)
- Crack lines radiating from foundation upward through blocks
- Each block: 16px border-radius, colored left border, icon + label
**Text below:** "What should be 'install and go' turned into a debugging session. Most issues traced back to the same root causes."

### Slide 4: CLAUDE STEPS IN

**Slide label:** "Claude Steps In"
**Title:** "Every Wall, a Fix"
**Subtitle:** "Read. Search. Solve. Repeat."
**Layout:** Three mini walls in horizontal sequence showing progression
**Visual:**
- Three brick wall illustrations side by side (flex row, equal width):
  - Wall 1: mostly intact, small crack. Blue Claude bubble: "Read the error"
  - Wall 2: large cracks, chunks falling. Blue Claude bubble: "Searched known issues"
  - Wall 3: mostly crumbled, rubble at base. Blue Claude bubble: "Wrote the fix"
- Green glow behind each crumbled section (CSS box-shadow or gradient)
- Arrows between walls showing progression (→)
- Each wall: CSS brick pattern using repeating-linear-gradient
**Insight box (green):** "Each fix revealed the next wall. Wayland kept being the common thread."

### Slide 5: THE BIG ONE

**Slide label:** "The Big One"
**Title:** "GPU Memory Is Full"
**Subtitle:** "It Wasn't."
**Layout:** Large wall + two chat bubbles + VRAM indicator
**Visual:**
- Massive wall (taller and wider than slide 4's walls), "GPU MEMORY FULL" in large red text across it
- Optional: simplified VRAM bar at top showing mostly empty (green fill ~20%, indicating plenty free)
- Two chat bubbles below/beside the wall:
  - Blue (Claude): "Looks like an OOM issue. Let's build proxies."
  - Purple (You): "Something doesn't add up."
- A hairline crack forming at the wall's base — tension, not yet resolved
**Text:** Minimal — the visual tells the story. Small caption: "We started solving the wrong problem."

### Slide 6: THE REVEAL

**Slide label:** "The Reveal"
**Title:** "Wayland Again."
**Subtitle:** "Wall Shattered."
**Layout:** Crumbled wall + before/after + terminal fix
**Visual:**
- Top: the wall from slide 5 in rubble. Scattered brick pieces (CSS positioned blocks, rotated, faded). Green glow radiating from behind where the wall was.
- Middle: simple two-column comparison:
  - Left card (red border, X icon): "Wayland"
  - Right card (green border, checkmark icon): "X11"
- Bottom: small terminal block showing the fix concept generically:
  ```
  # Force X11 instead of Wayland
  3 environment variables + /opt/resolve/bin/resolve
  ```
- Green stat box: "Full 4K Playback" in large text
**Insight box (green):** "Turns out it wasn't a memory problem. It was a Wayland compatibility issue."

### Slide 7: WHAT'S NEXT

**Slide label:** "What's Next"
**Title:** "Diving Trip. Resolve MCP."
**Subtitle:** "Pieces Coming Together."
**Layout:** Full underwater scene with floating elements
**Visual:**
- Background: underwater gradient (bottom: #0a1628 deep navy, middle: #0d3b66 deep blue, top: #1a8a8a teal, hints of #22d3ee cyan at very top)
- CSS bubble particles: 8-12 small circles (4-12px) with white/cyan, low opacity (0.1-0.2), scattered across slide
- Diving mask icon: simple geometric shape (oval + snorkel rectangle) in dark navy, centered lower-third. Do NOT attempt a full diver silhouette — too complex for pure CSS.
- Two floating cards (semi-transparent, glass-morphism style):
  - "DJI Action 4" with camera icon
  - "Insta360 X5" with 360 camera icon
- Bottom: subtle dotted-line pipeline, partially faded: "footage → AI → timeline" in muted teal text with dashed connectors
**Text:** "I love editing, but the repetitive parts eat time I'd rather spend diving. Found a DaVinci Resolve MCP. Still brainstorming."

### Slide 8: TAKEAWAYS

**Slide label:** "Key Takeaways"
**Title:** "What I Learned Setting Up Resolve on Linux"
**Layout:** takeaway template — 3 items with icons, dividers, CTA
**Visual:** Closing gradient matching cover (dark navy → deep purple)
**Takeaways:**
1. Icon: magnifying glass or eye emoji. **"Error messages lie."** "The GPU error pointed at memory. The real cause was the display server. Always read the logs."
2. Icon: handshake or arrows emoji. **"AI collaboration > AI answers."** "Claude got it wrong first. I steered us back. Neither would have solved it alone."
3. Icon: tools or gear emoji. **"The engineering stack becomes the creative stack."** "Same Linux setup. Same hardware. Now running a full video editing workflow."
**CTA:** "Anyone else using AI to streamline their video editing workflow?"
**Footer:** Author tag, slide 8/8.

## Technical Notes

- Canvas: 1080x1350px, 2x deviceScaleFactor for retina
- Font stack: Inter (body), JetBrains Mono (terminal blocks), Poppins (brand name only)
- Min font: 15px (phone readability at 0.35x scale)
- All flex children MUST have `min-width: 0` to prevent overflow
- Stat boxes use `grid-template-columns: 1fr 1fr`, never flex
- Brick wall texture: `repeating-linear-gradient` with alternating offsets
- Bubble particles: absolute positioned divs with `border-radius: 50%`, low opacity
- Underwater gradient: `linear-gradient` with 4+ color stops
- Glass-morphism cards: `backdrop-filter: blur(10px)`, semi-transparent background
- Diving mask icon: simple geometric CSS (oval + rectangle), NOT a full diver silhouette
- Film frame: sprocket holes via `repeating-radial-gradient`, NOT a clapperboard
- All layouts use flex/grid, no absolute positioning except footer and decorative elements
- Each slide has `::before` pseudo-element with radial blue glow (top-right corner)
- Render via: `node tools/carousel-render/render.js`
