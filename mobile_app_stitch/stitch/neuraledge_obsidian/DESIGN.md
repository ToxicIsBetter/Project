# Design System Strategy: The Quantitative Minimalist

## 1. Overview & Creative North Star
The Creative North Star for this design system is **"The Digital Architect."** 

In the high-stakes world of institutional crypto trading, clarity is the ultimate luxury. This system moves away from the "neon-glow" clichés of retail Web3. Instead, it adopts the authoritative, data-dense DNA of legacy terminals (Bloomberg/Reuters) and evolves them through a lens of modern, high-end editorial design. 

We break the "standard dashboard" mold by prioritizing **Information Density over Decoration.** We utilize intentional asymmetry, where technical data is balanced by expansive, focused typography. By leaning into a "Technical Brutalism" refined by premium finishes, we create an environment that feels like a precision instrument—unapologetically professional, cold, and efficient.

---

## 2. Colors: Tonal Architecture
The palette is a calculated study in deep obsidian and technical blues. We do not use color for "vibes"; we use it for **Semantic Signal.**

### The "No-Line" Rule
Standard UI relies on lines to separate data. In this system, **1px solid borders for sectioning are prohibited.** Boundaries must be defined through background color shifts. 
- Use `surface_container_low` for the main workspace.
- Use `surface_container_highest` for active widgets.
- The eye should navigate through shifts in "mass" and tone, not by following "boxes."

### Surface Hierarchy & Nesting
Treat the UI as a series of physical layers.
- **Level 0 (Background):** `surface_dim` (#10141a) - The base canvas.
- **Level 1 (Sub-sections):** `surface_container` (#1c2026) - Used for grouping logic.
- **Level 2 (Active Cards):** `surface_container_high` (#262a31) - Floating data modules.
- **Level 3 (Interaction):** `surface_container_highest` (#31353c) - Focused/hovered elements.

### The "Glass & Gradient" Rule
To add "soul" to the terminal, main CTA elements or AI-driven insights should use **Glassmorphism.** Use semi-transparent versions of `primary_container` (#58a6ff) with a `backdrop-filter: blur(12px)`. Apply a subtle linear gradient (from `primary` to `primary_container`) on active buttons to mimic the slight sheen of a high-end physical console.

---

## 3. Typography: Technical Authority
We utilize a dual-font strategy to balance legibility with a "machine-processed" aesthetic.

*   **Display & Headlines (Space Grotesk):** This font provides the "Institutional" character. Its wide stance and geometric quirks feel intentional and architectural. Use `display-lg` for portfolio totals and `headline-sm` for section headers.
*   **Data & Body (Inter):** Inter is our workhorse. At small scales, its high x-height ensures that 10pt trade data is readable. 
*   **The Technical Edge:** For ticker symbols and timestamps, use `label-sm` with `letter-spacing: 0.05em` to evoke a monospaced, terminal-like precision without sacrificing the balance of a sans-serif.

---

## 4. Elevation & Depth: Tonal Layering
We reject the "drop shadow" of 2010s design. Depth is earned through light and opacity, not black ink.

*   **The Layering Principle:** Instead of shadows, stack `surface_container_lowest` objects on top of `surface_container_low` regions. The 2-4% difference in hex value provides enough "lift" for the eye to distinguish priority.
*   **Ambient Shadows:** When a modal or pop-over *must* float, use a shadow with a blur of `40px` and an opacity of `6%`. The color must be sampled from `on_surface` (#dfe2eb), creating a "glow" of light rather than a dark void.
*   **The "Ghost Border" Fallback:** If a layout feels too amorphous, use a "Ghost Border": `outline_variant` (#414752) at **15% opacity**. This creates a suggestion of a container that disappears upon focus.

---

## 5. Components: The Precision Set

### Buttons
*   **Primary:** Solid `primary_container` (#58a6ff) with `on_primary_fixed` text. No rounded corners beyond `md` (0.375rem).
*   **Secondary:** Ghost style. No background, `outline_variant` ghost border, `primary` text.
*   **Tertiary:** Text-only, uppercase `label-md` with +10% letter spacing.

### Data Inputs
*   **The Terminal Input:** Dark backgrounds (`surface_container_lowest`). Focus state is indicated by a 2px left-accent bar of `primary` (#a2c9ff), rather than a full border glow.

### Cards & Lists
*   **The Divider Forfeit:** Forbid the use of horizontal lines between list items. Use the **Spacing Scale** (specifically `1.5` for tight data or `3` for editorial content) to create separation.
*   **Signal Indicators:** Use `secondary` (#67df70) for "Long/Positive" and `tertiary_container` (#ff7b70) for "Short/Negative." These should be small, high-density chips or simple 2px underlines.

### AI Insight Modules (Unique Component)
*   These modules should use the **Glassmorphism** rule. A `surface_container_high` background at 60% opacity with a `primary` subtle inner-glow to signify "Neural" activity.

---

## 6. Do's and Don'ts

### Do
*   **Do** prioritize a high information-to-ink ratio.
*   **Do** use `spaceGrotesk` for all numerical currency values to give them "weight."
*   **Do** use the `0.5` spacing unit (0.1rem) for micro-alignments in data tables.
*   **Do** allow elements to overlap slightly (e.g., a tooltip slightly covering a chart edge) to create a sense of depth and density.

### Don't
*   **Don't** use 100% white (#FFFFFF). All "white" text must use `on_surface` (#dfe2eb) to reduce eye strain in dark mode.
*   **Don't** use large corner radii. Stick to `sm` (0.125rem) or `md` (0.375rem) to maintain a professional, rigid feel.
*   **Don't** use standard "Success Green" (#00FF00). Only use the calibrated `secondary` (#67df70) to ensure it sits correctly against the deep blue background.
*   **Don't** use "centered" layouts. Institutional tools are "Left-Heavy" or "Grid-Locked" for maximum efficiency.