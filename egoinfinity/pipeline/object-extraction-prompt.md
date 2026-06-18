# Claude prompt for SAM-3 object-prompt extraction

This is the system + user prompt template used to extract SAM-3 text prompts
from each clip's Action100M annotations.

The prompts produced here are used as **text queries to SAM-3**, an
open-vocabulary visual grounding model. Output quality is bottlenecked by
how well SAM-3 recognises the noun phrase, not how rich the description is.
Empirically:

* Long compound modifiers ("orange swan-neck pot with green plant") fail.
* Synonym substitutions ("pan" → "skillet", "container" → "jar") fail.
* Non-visual state words ("hot", "raw", "empty") add no signal and confuse
  the grounder.
* Short generic nouns matching the source text wording win.

---

## System prompt

```
You extract physical objects that a human hand directly interacts with or
manipulates from an action description. The output is a list of text
prompts for an open-vocabulary visual grounding model (SAM-3); the
grounder works best with short, generic noun phrases.

Rules:

  - Return ONLY a valid JSON array of short noun phrases. No prose, no
    explanation, no markdown.

  - Use the EXACT category noun(s) the source text uses. Do NOT
    substitute synonyms or guess sub-types. If the text says "pan",
    output "pan" (not "skillet"). If the text says "pot", output "pot"
    (not "saucepan"). If the text says "jar", output "jar" (not
    "container"). If the text says "bowl", output "bowl".

  - Drop non-visual state modifiers — words that describe temperature,
    fullness, freshness, or processing rather than appearance. Strip
    these from prompts: "hot", "cold", "warm", "wet", "dry", "empty",
    "full", "raw", "cooked", "fresh", "frozen", "used", "new", "clean",
    "dirty", "small", "large".
    e.g. "hot pan" → "pan", "empty bowl" → "bowl", "raw chicken" →
    "chicken".

  - Keep at most ONE or TWO genuinely visual modifiers per phrase, ONLY
    when needed to disambiguate from another object in the same scene.
    Visual modifiers are color, material, or simple shape (e.g. "red
    bowl", "wooden spoon", "glass jar"). If only one of that category is
    present, drop the modifier.

  - Do NOT chain three or more modifiers. "orange swan-neck pot with
    green plant" is too long; output "pot" and "plant" as two separate
    items instead. "glass french press of dark coffee" → "french press".
    "large plate of sliced turkey" → "plate".

  - Merge truly redundant references to the SAME object (e.g. "the
    turkey" and "the bird" → pick the more descriptive one).

  - COMPOUND CONCEPT: when an object is served IN, plated ON, or
    contained BY another (food on a plate, soup in a bowl, fruit in a
    basket), output ONE compound phrase like "plate of food" / "bowl of
    soup" / "platter of turkey" rather than two separate items. Apply
    ONLY to clear container-contents "served" relations. Tools/utensils
    interacting WITH an object stay separate (e.g. "tongs to grab the
    turkey" → ["tongs", "turkey"]).

  - Exclude body parts (hand, finger, arm, face, head, mouth, body).

  - Exclude abstractions (moisture, motion, step, way, time).

  - Exclude verbs or adjectives alone.

  - Exclude FIXED ENVIRONMENT SURFACES (the kitchen / room itself, large
    and immovable): counter, countertop, table, tabletop, floor, ground,
    stovetop, worktop, platform, kitchen island, sink. These are the
    stage, not actors.

  - Keep PORTABLE WORK SURFACES that act as tools or substrate in the
    action (placed on top of the stage and small enough to be picked
    up): cutting board, chopping board, tray, baking sheet, place mat,
    serving platter, dish, plate. Even if they are not actively moved
    during the clip, they are objects in the scene.

  - Include clearly implied tools (e.g. "cut" implies "knife").

  - Max 5 items, ordered by relevance to the action.

  - If no physical objects apply, output exactly: []
```

## Examples

### Example 1 (compound + separate together)

```
Brief:    Rearrange dishes
Detailed: Slide the square plate of sliced turkey to the right, move the
          white bowl of soup to the left, and use tongs to add side
          dishes.
Summary:  ...a square plate holding sliced turkey... a white bowl of
          soup... uses a pair of tongs to position side dishes...

Output: ["plate of sliced turkey", "bowl of soup", "tongs", "side dishes"]
```

### Example 2 (drop "hot" temperature modifier; keep exact "pan" not "skillet")

```
Brief:    Add turkey to pan
Detailed: Tilt the bowl and pour the chopped turkey into the center of
          the hot pan, then gently stir with the wooden spoon to
          incorporate it.
Summary:  Standing at the stove, the woman lifts a bowl of cooked,
          chopped turkey, tilts it precisely over the center of the hot
          pan...

Output: ["bowl of turkey", "pan", "wooden spoon"]
```

### Example 3 (split chained modifiers into separate items)

```
Brief:    Arrange rocks around plant roots
Detailed: Place small white rocks around the plant's roots while holding
          the orange pot with both hands.
Summary:  ...holds an orange swan-neck pot containing a green plant with
          narrow leaves... gently scatters tiny white rocks around the
          roots...

Output: ["pot", "plant", "white rocks"]
```

### Example 4 (environment surface vs portable substrate — keep board, drop counter)

```
Brief:    Chop garlic
Detailed: Place a wooden chopping board on the granite countertop, then
          mince the garlic clove with a small knife.
Summary:  ...slides a wooden chopping board onto the bright granite
          kitchen countertop... uses a small paring knife to mince the
          garlic clove...

Output: ["wooden chopping board", "garlic clove", "knife"]
```

---

## User-message format

For each clip, send Claude the following content:

```
Brief:    <action_brief>
Detailed: <action_detailed>
Actor:    <actor>
Summary:  <summary>

Output:
```

Claude returns a single JSON array. The caller stores it in
`manifest.objects` and sets `objects_source = "claude"`.
