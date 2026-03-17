# Content Verification Skill

Independent fact-checking and fairness review for all generated marketing content. This skill MUST be run by a separate agent before any content is published or rendered as final.

## When to Run

**Mandatory trigger:** After generating post text or carousel HTML, before final render/publish. This is not optional — all content passes through verification.

**Who runs it:** An independent agent (subagent) with no context from the content creation session. Fresh eyes prevent confirmation bias.

## Verification Process

### Step 1: Extract Claims

Read the content (post.txt + carousel HTML) and extract every factual claim into a structured list:

```
CLAIM: [exact quote or paraphrase]
TYPE: [stat | comparison | technical | experience | generalization]
VERIFIABLE: [yes | partially | no — opinion]
```

### Step 2: Local Verification

For each verifiable claim:

1. **Check corvia knowledge base** — `corvia_search` for prior entries that support or contradict the claim
2. **Check source code** — if the claim is about how a tool/feature works, read the actual code or config
3. **Check project history** — git log, existing docs, conversation transcripts for evidence
4. **Check the numbers** — if a stat is quoted, find the source data. If no source exists, flag it as "estimated" or "anecdotal"

### Step 3: Online Verification

For each claim that references external tools, patterns, or industry trends:

1. **Web search** for the specific claim (e.g., "Playwright MCP file:// URL blocked")
2. **Check official docs** for tool capabilities (e.g., Playwright MCP GitHub repo, Anthropic docs)
3. **Search for counterexamples** — actively look for evidence that contradicts the claim
4. **Check competitor fairness** — if comparing tools, verify the comparison is accurate and current

### Step 4: Fairness Review

Evaluate the overall content for:

| Check | What to look for |
|-------|-----------------|
| **Strawman** | Does it unfairly represent a tool, method, or competing approach? |
| **Cherry-picking** | Does it only show cases where the approach works, ignoring failures? |
| **Survivorship bias** | Does it present the successful outcome without the failed attempts? |
| **Overgeneralization** | Does "it worked for me" become "it works for everyone"? |
| **False precision** | Are rough estimates presented as exact data? (e.g., "80%" without measurement) |
| **Recency bias** | Are claims about tools based on current versions, not outdated info? |
| **Attribution** | Are ideas, tools, and approaches properly credited? |
| **Tone balance** | Does it acknowledge limitations and trade-offs, not just benefits? |

### Step 5: Generate Report

Output a structured verification report:

```
## Verification Report: [content title]
Date: YYYY-MM-DD
Reviewer: [agent-id]

### Claims Verified
- CLAIM: "..." → VERIFIED (source: ...)
- CLAIM: "..." → VERIFIED (source: ...)

### Claims Flagged
- CLAIM: "..." → UNVERIFIED — no source data found. Recommend: label as anecdotal or remove
- CLAIM: "..." → EXAGGERATED — actual evidence suggests X, not Y. Recommend: soften language
- CLAIM: "..." → OUTDATED — this was true for version X but changed in version Y

### Fairness Issues
- [issue description and recommended fix]

### Verdict
[PASS | PASS WITH CHANGES | NEEDS REVISION]
[Summary of required changes before publishing]
```

## Fairness Standards

### What "fair" means for this content

1. **Honest about your own experience** — "I found that..." not "Everyone knows that..."
2. **Specific, not sweeping** — "In my carousel work, 3 out of 4 attempts had overflow issues" not "AI always fails at layout"
3. **Acknowledge what AI does well** — if critiquing a limitation, also credit what it does right
4. **No strawman comparisons** — compare the actual current state, not a caricatured version
5. **Stats need sources** — either measured data, cited research, or explicitly labeled "estimated from my experience"
6. **Tool claims are verifiable** — if you say "Playwright MCP blocks file:// URLs", that should be checkable in docs or code
7. **Alternatives acknowledged** — if presenting one solution, note that others exist ("Windows has per-app GPU settings too")
8. **Proportional claims** — "the difference was immediate" is fine for personal experience; "this changes everything" needs evidence

### Red flags that must be fixed before publishing

- Unqualified superlatives: "best", "only way", "always", "never" (without evidence)
- Fake precision: specific percentages without measurement methodology
- Misleading before/after: cherry-picked examples that don't represent typical results
- Omitting known limitations of the recommended approach
- Claiming credit for community discoveries without attribution

## Applying to Existing Content

### Example: "AI Blind Designer" carousel claims to verify

| Claim | Type | How to verify |
|-------|------|---------------|
| "It writes clean code" | experience | Check actual AI output in git history |
| "Text overflows containers. Diagrams overlap." | experience | Verify with actual screenshots from development |
| "3-5x more iterations" | stat | Count actual iteration rounds from conversation transcripts |
| "80% of fixes break something else" | stat | Count fix-break cycles from actual sessions |
| "0 self-caught errors" | stat | Verify AI never caught its own visual errors without screenshot |
| "Playwright MCP lets Claude Code launch a browser" | technical | Check Playwright MCP docs and actual .mcp.json config |
| "file:// URLs blocked by security policy" | technical | Verify in Playwright MCP source/docs |
| "The difference was immediate" | experience | Acceptable as personal experience if qualified |
| Windows has per-app GPU settings | technical | Verify in Windows docs (linux-discovery post) |

### Stats qualification guide

| If your evidence is... | Then label it... | Example |
|------------------------|-----------------|---------|
| Measured data from logs/transcripts | State as fact with method | "Across 12 iterations, 9 introduced new layout issues" |
| Rough count from memory | "roughly" or "around" | "Roughly 3-5x more back-and-forth" |
| General impression | "felt like" or "in my experience" | "It felt like most fixes broke something else" |
| No evidence at all | Remove or reframe | Don't use a specific number |

## Integration with Content Workflow

In the visual content skill (`.agents/skills/visual-content-playwright.md`), the workflow becomes:

1. Generate post text + carousel HTML
2. **→ Run content-verification agent (this skill) ←**
3. Address any flagged claims
4. Render final PNGs + PDF
5. Publish

The verification step is non-negotiable. Ship honest content.
