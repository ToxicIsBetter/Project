# Technical Specification: The Algorithmic Architect

## 1. Overview & Creative North Star
The visual identity of this design system is rooted in the "Algorithmic Architect" – a philosophy that merges the brutalist efficiency of institutional finance (Bloomberg) with the ethereal, fluid intelligence of Web3. 

While typical dashboards rely on rigid grids and heavy borders to separate data, this system utilizes **Tonal Depth** and **Intentional Asymmetry**. We move away from the "template" look by treating the interface as a living, breathing command center where information density is balanced by sophisticated layered surfaces. We prioritize "High-Information Editorial," where data is not just displayed but curated through a hierarchy that feels both authoritative and futuristic.

---

## 2. Colors & Surface Logic
The palette is a sophisticated dark-mode spectrum designed to reduce eye strain during long-duration trading while highlighting critical AI-driven signals.

### The "No-Line" Rule
Standard UI relies on `1px` borders to define sections. In this system, **solid 100% opaque borders are prohibited.** Boundaries must be defined through background color shifts. Use `surface-container-low` (#181c22) for secondary modules and `surface-container-high` (#262a31) for active interaction zones. This creates a "seamless" look where the eye navigates via value changes rather than structural lines.

### Surface Hierarchy & Nesting
Treat the UI as a physical stack of semi-polished materials:
- **Base Layer:** `surface` (#10141a) — The foundation.
- **Structural Modules:** `surface-container` (#1c2026) — Main dashboard cards.
- **Nested Insights:** `surface-container-highest` (#31353c) — For deep-dive data or focused AI analysis within a card.

### The "Glass & Gradient" Rule
To inject a "Web3" soul into the terminal:
- **Glassmorphism:** For floating overlays or context menus, use `surface-variant` (#31353c) at 60% opacity with a `20px` backdrop-blur. 
- **Signature Gradients:** Main Action buttons or "AI-Active" states should utilize a linear gradient from `primary` (#a2c9ff) to `primary_container` (#58a6ff) at 135 degrees. This adds a "lithic" glow that flat colors cannot achieve.

---

## 3. Typography
The typography strategy creates a tension between the tech-forward **Space Grotesk** and the hyper-functional **Inter**.

- **Display & Headlines (Space Grotesk):** Used for high-level metrics and brand-defining moments. Its geometric quirks signal a modern, "Neural" edge.
- **Data & Body (Inter):** Used for all institutional data, trading pairs, and terminal logs. Inter provides the "Bloomberg-grade" legibility required for high-density environments.
- **Labeling:** Utilize `label-sm` (0.6875rem) in `on_surface_variant` (#c0c7d4) for metadata. This keeps the interface dense but legible, mimicking a professional ticker-tape.

---

## 4. Elevation & Depth
Depth is achieved through **Tonal Layering** rather than drop shadows.

- **The Layering Principle:** To lift a card, do not add a shadow. Instead, transition from `surface-container-low` to `surface-container-high`. The relative contrast provides all the necessary visual affordance.
- **Ambient Shadows:** For "floating" elements like Tooltips, use an extra-diffused shadow: `0px 24px 48px rgba(0, 0, 0, 0.4)`. The shadow must never be pure black; it should feel like an occlusion of the ambient dark background.
- **The "Ghost Border":** If a container requires a boundary (e.g., in high-density data tables), use a "Ghost Border": `outline-variant` (#414752) at **15% opacity**. This provides a whisper of structure without breaking the seamless aesthetic.

---

## 5. Components

### Cards & Modules
*   **Style:** No internal dividers. Use `spacing-6` (1.3rem) to separate content sections vertically. 
*   **Interaction:** On hover, a card should shift from `surface-container` to `surface-container-high`.

### Buttons
*   **Primary:** Gradient-filled (`primary` to `primary_container`). Border-radius: `md` (0.375rem).
*   **Secondary:** Ghost style. No background, but a `Ghost Border` (15% opacity `outline-variant`).
*   **Tertiary:** Text-only, using `primary` (#a2c9ff) for the label.

### Input Fields
*   **Structure:** Background should be `surface-container-lowest` (#0a0e14) to create an "inset" effect, making the field feel carved out of the terminal.
*   **State:** On focus, the `Ghost Border` increases to 100% opacity of `surface_tint` (#a2c9ff).

### Chips & Badges
*   **Positive (Profit):** Background: `secondary_container` (#27a640) at 20% opacity. Text: `secondary` (#67df70).
*   **Warning (Risk):** Background: `error_container` (#93000a) at 20% opacity. Text: `error` (#ffb4ab).

### Data Visualizations
*   **The "Neural" Glow:** Charts should use the `primary` (#a2c9ff) token for trend lines, applying a subtle glow effect (drop-shadow with `primary` color at low opacity) to simulate an emissive terminal screen.

---

## 6. Do's and Don'ts

### Do
- **Do** embrace high information density. Institutional users prefer seeing more data at once over "generous" white space.
- **Do** use `Space Grotesk` for all numerical values that represent "AI Insights" to differentiate them from "Market Data."
- **Do** use `spacing-1` and `spacing-2` for micro-adjustments in data tables to keep the "Terminal" feel.

### Don't
- **Don't** use standard #000000 shadows; they look "muddy" on our deep #10141a background.
- **Don't** use 100% opaque lines to separate list items. Use a background color shift or `0.1rem` of vertical space.
- **Don't** use standard "Success Green." Only use the `secondary` (#67df70) token to maintain the specific brand tonal balance.
- **Don't** use rounded corners larger than `xl` (0.75rem). This system is about precision, not "friendliness."