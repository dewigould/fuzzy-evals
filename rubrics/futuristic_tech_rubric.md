# Rubric for Evaluating Answers to Futuristic Technology Roadmapping Questions

## Instructions for Graders

You are evaluating answers to technology roadmapping questions. Each question asks the respondent to map out a plausible 15-year path to an ambitious technological outcome, assuming AI-accelerated R&D. These are not prediction exercises — they are tests of whether the respondent can think rigorously about technological development under deep uncertainty while committing to specific, defensible claims.

Here are some failure modes to watch for:

- **Buzzword roadmaps**: stringing together plausible-sounding milestones with no mechanistic reasoning connecting them
- **Wikipedia recaps**: summarizing the current state of the art without projecting forward
- **Uniform optimism**: treating every bottleneck as equally solvable without distinguishing hard problems from very hard problems
- **Timeline inflation**: assigning dates to milestones without justifying why *that* date and not five years earlier or later
- **Hedging as structure**: organizing the answer around uncertainty ("it depends on funding, breakthroughs, and political will") rather than around substantive claims

**Calibration guidance:**
- 0–2: Non-engagement or surface-level recaps of the field dressed up as a roadmap
- 3–4: Competent overview — hits the obvious milestones but doesn't reveal deeper understanding
- 5: Strong — demonstrates real understanding of the bottlenecks and makes defensible choices about sequencing
- 6: Excellent — identifies non-obvious dependencies, shows where the hard problems *actually* are, surprises you
- 7: Outstanding — the roadmap itself is an argument; reading it changes how you think about the feasibility question
- 8: Exceptional — could seed a serious research agenda or strategy document

---

## Criteria

### 1. Scientific Grounding

Does the answer demonstrate real understanding of the underlying science and engineering, or does it gesture at plausible-sounding steps?

This criterion catches the failure mode of "sounds right to a layperson but would make a domain expert wince." The best answers show the respondent understands *why* specific problems are hard — not just *that* they are hard.

| Score | Description |
|-------|-------------|
| 0 | No meaningful scientific content; purely speculative hand-waving |
| 1 | Name-drops relevant fields or techniques but mischaracterizes or oversimplifies them |
| 2 | Gets the basics right but stays at a textbook level; no sense that the respondent understands frontier challenges |
| 3 | Demonstrates working knowledge of the current state of the art; correctly identifies the major open problems |
| 4 | Shows understanding of *why* specific problems are hard — identifies the physical, chemical, or biological constraints at play |
| 5 | Engages with the science at a level of detail that constrains the roadmap; the physics/biology/engineering *shapes* the timeline rather than decorating it |
| 6 | Identifies specific technical parameters or thresholds that matter (e.g., critical current densities, channel counts, error rates) and reasons about them |
| 7 | Demonstrates understanding of the field deep enough to distinguish between problems that are hard-but-tractable and problems that may require fundamentally new approaches |
| 8 | The scientific reasoning itself is a contribution — identifies constraints or opportunities that a knowledgeable reader might not have connected |

---

### 2. Bottleneck Identification and Differentiation

Does the answer distinguish between different *types* of obstacles, or does it treat all challenges as generic "breakthroughs needed"?

Not all bottlenecks are alike. Some are scientific (we don't know if it's possible), some are engineering (we know it's possible but can't build it yet), some are scaling (it works in the lab but not at volume), some are economic or regulatory. A good roadmap tells you which is which and explains why it matters.

| Score | Description |
|-------|-------------|
| 0 | No bottleneck analysis; just a sequence of milestones with no obstacles acknowledged |
| 1 | Acknowledges that challenges exist but treats them as a homogeneous blob ("significant technical hurdles remain") |
| 2 | Lists bottlenecks but doesn't differentiate their nature or relative difficulty |
| 3 | Distinguishes between at least two types of bottleneck (e.g., scientific unknowns vs. engineering challenges) |
| 4 | Classifies bottlenecks meaningfully and explains why different types require different approaches and timelines |
| 5 | Identifies which bottlenecks are serial (must be solved before others) vs. parallel (can be worked on simultaneously), shaping the roadmap's structure |
| 6 | Identifies a bottleneck that is non-obvious — one that most roadmaps for this technology would miss or underweight |
| 7 | The bottleneck analysis reveals the *structure* of the problem: what's actually rate-limiting, what's coupled, and what can be decoupled |
| 8 | Bottleneck analysis is sophisticated enough to identify potential "shortcut" paths — ways the problem could become easier than expected if a specific sub-problem yields |

---

### 3. Timeline Plausibility and Justification

Are the milestones sequenced and dated in a way that reflects real reasoning about development timelines, or are dates assigned arbitrarily?

This criterion penalizes two failure modes equally: unjustified optimism (everything converges perfectly by 2040) and unjustified vagueness (milestones are described but never dated). A good roadmap makes you believe the respondent actually thought about *how long things take and why.*

| Score | Description |
|-------|-------------|
| 0 | No timeline structure; milestones are listed without temporal ordering or dates |
| 1 | Dates are assigned but feel arbitrary — no reasoning connects the timeline to the difficulty of each step |
| 2 | Timeline has some logic but is either uniformly optimistic or uniformly hedged; all milestones feel equally spaced |
| 3 | Timeline reflects basic reasoning about sequencing — prerequisites come before dependents, harder steps take longer |
| 4 | Timeline is justified with reference to analogous historical development timelines or known development rates |
| 5 | Timeline explicitly accounts for the AI-acceleration assumption in a nuanced way — identifying which steps AI compresses and which it doesn't |
| 6 | Timeline reasoning identifies critical path dependencies and explains what would need to go right (and how likely that is) for the overall schedule to hold |
| 7 | The timeline includes explicit "if/then" branching — alternate paths or fallback approaches depending on which breakthroughs materialize first |
| 8 | Timeline reasoning is itself illuminating about the nature of technology development — the sequencing argument teaches you something about how breakthroughs propagate |

---

### 4. Scaling and Deployment Realism

Does the answer address the gap between "works in principle" and "deployed at the specified scale," or does it hand-wave the last mile?

Many technology roadmaps are actually roadmaps to a lab demo, not to deployment. This criterion specifically rewards attention to manufacturing, cost curves, supply chains, regulation, infrastructure, and all the unglamorous work that separates a breakthrough from an impact.

| Score | Description |
|-------|-------------|
| 0 | No consideration of scaling or deployment; the roadmap ends at "breakthrough achieved" |
| 1 | Brief mention that scaling will be needed, with no specifics |
| 2 | Acknowledges specific scaling challenges (manufacturing, cost, regulation) but doesn't integrate them into the roadmap |
| 3 | Includes scaling milestones as distinct phases in the roadmap with at least some reasoning about what they require |
| 4 | Addresses cost trajectories, manufacturing constraints, or supply chain requirements with enough specificity to be falsifiable |
| 5 | Integrates scaling considerations throughout the roadmap rather than appending them at the end — recognizes that deployment constraints shape R&D priorities |
| 6 | Identifies specific scaling bottlenecks that are distinct from the core scientific challenges and explains why they might be independently rate-limiting |
| 7 | Demonstrates understanding of how technology deployment actually works — learning curves, pilot-to-production transitions, regulatory timelines, infrastructure buildout rates |
| 8 | The scaling analysis is sophisticated enough to distinguish this technology's deployment dynamics from generic "technology maturation" — it reflects the specific economics and logistics of the domain |

---

### 5. Integration and Interconnection Awareness

Does the answer recognize that this technology doesn't develop in isolation, or does it treat the roadmap as a single-thread story?

Real technology development is embedded in an ecosystem. Advances in adjacent fields enable or constrain progress. Competing approaches interact. The same breakthrough might serve multiple applications, changing incentive structures. A great roadmap is aware of this web.

| Score | Description |
|-------|-------------|
| 0 | Treats the technology as developing in a vacuum; no reference to adjacent fields or enabling technologies |
| 1 | Mentions that other fields are relevant but doesn't specify how |
| 2 | Identifies one or two adjacent technologies or fields that matter, with brief explanation |
| 3 | Shows how progress in specific adjacent fields (AI, materials science, manufacturing, etc.) enables key milestones |
| 4 | Identifies bidirectional dependencies — how this technology both depends on and enables progress in other areas |
| 5 | Maps the broader ecosystem well enough to identify where convergence of multiple advances creates windows of opportunity |
| 6 | Identifies a non-obvious cross-domain dependency or enabling technology that most roadmaps would miss |
| 7 | The ecosystem mapping reveals structural features of the problem — e.g., that progress is bottlenecked on a field that has different incentive structures or development rates |
| 8 | The interconnection analysis is itself a contribution — it reveals how the feasibility of this technology is coupled to broader technological trajectories in a way that reframes the question |

---

### 6. Epistemic Honesty Under Ambition

Does the answer maintain intellectual honesty while still being ambitious, or does it either retreat into safe vagueness or bulldoze through uncertainties?

These questions explicitly ask for ambition ("assume AI-accelerated R&D has meaningfully compressed timelines"). But ambition without honesty is just hype. The best answers are simultaneously bold in their claims and precise about where those claims rest on shaky ground. This criterion rewards the difficult balance of saying "here's how it could work" without pretending the hard parts are easy.

| Score | Description |
|-------|-------------|
| 0 | Either pure hype (ignores all difficulties) or pure hedging (refuses to commit to anything); no useful calibration |
| 1 | Acknowledges uncertainty in a generic way ("many challenges remain") without localizing it to specific claims |
| 2 | Distinguishes between more and less certain parts of the roadmap, but the distinction feels cursory |
| 3 | Clearly marks which milestones are extensions of current trends vs. which require genuine breakthroughs, and treats them differently |
| 4 | Assigns rough confidence levels or probability ranges to key milestones, with reasoning behind them |
| 5 | Identifies the specific assumptions that, if wrong, would invalidate the roadmap — the load-bearing uncertainties |
| 6 | Handles the "AI-acceleration" assumption with sophistication — doesn't treat it as a magic wand but reasons about which specific bottlenecks AI can and cannot help with |
| 7 | The uncertainty analysis is productive rather than defensive — it identifies what we'd need to learn earliest to de-risk the roadmap, effectively prioritizing research |
| 8 | Masterful calibration — simultaneously ambitious and honest in a way that makes the roadmap more useful precisely because you know exactly where to push back on it |

---

## Output Format

Return your evaluation as a JSON object with no additional commentary, explanation, or text outside the JSON. Use the following structure:

```json
{
  "scientific_grounding": 0,
  "bottleneck_identification": 0,
  "timeline_plausibility": 0,
  "scaling_deployment_realism": 0,
  "integration_awareness": 0,
  "epistemic_honesty": 0,
  "total": 0
}
```

Replace each `0` with the appropriate score (0–8). The `total` field should be the sum of all six individual scores (maximum 48).

**Do not include any text before or after the JSON object.**