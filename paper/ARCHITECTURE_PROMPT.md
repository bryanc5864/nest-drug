# NEST-DRUG Architecture Diagram Prompt

## For Image Generation (ChatGPT/DALL-E, Midjourney, etc.)

### Style Requirements
- Publication-quality scientific figure for ICLR 2026 conference paper
- Clean, professional flowchart with two main panels
- Colors: Blue (#4472C4) for MPNN, Green (#27AE60) for L1, Teal (#17A2B8) for L2, Purple (#9B59B6) for L3, Orange (#E8871E) for FiLM highlight, Red (#C0392B) for prediction
- Rounded rectangle boxes with subtle shadows
- Black arrows showing data flow
- Mathematical notation in LaTeX style
- White/light gray background
- Legend at bottom
- Dimensions: 11" × 5.5" (wide format)

---

### Panel (a): Molecular Encoding Pipeline

**Left to right flow:**

1. **SMILES Input** (light gray box)
   - Label: "SMILES"
   - Subtext: "CC(=O)Oc1ccccc1C(=O)O"
   - Small monospace font for the SMILES string

2. **Arrow →**

3. **Molecular Graph** (white box with molecule visualization)
   - Label: "Molecular Graph"
   - Inside: Simple 2D molecule structure (benzene ring with substituents)
   - Show nodes (atoms as circles) connected by edges (bonds as lines)

4. **Arrow →**

5. **Feature Extraction** (light blue box)
   - Label: "Node/Edge Features"
   - Two sub-boxes inside:
     - "Atom Features" (70-dim): element, charge, hybridization, aromaticity, H-count
     - "Bond Features" (9-dim): type, conjugation, ring, stereo
   - Small icons representing feature vectors

6. **Arrow →**

7. **MPNN Block** (BLUE, prominent)
   - Label: "MPNN (L0)"
   - Subtext: "6 layers, GRU"
   - Inside show stacked layers: Layer 1, Layer 2, ..., Layer 6
   - Show message passing visual: nodes exchanging information
   - Annotation: "GRU updates"

8. **Arrow →**

9. **Output Embedding** (light gray)
   - Label: "h_mol"
   - Subtext: "512-dim"
   - Show as a vertical bar/vector representation

**Below the main flow (detail boxes):**

- **Message Passing Equation Box** (light green)
  - Title: "Message Passing"
  - Equation: m_v^(t) = Σ_{u∈N(v)} M(h_u, e_uv)
  - Dashed arrow connecting to MPNN block

- **GRU Update Equation Box** (light blue)
  - Title: "GRU Update"
  - Equation: h_v^(t+1) = GRU(h_v^(t), m_v^(t))
  - Dashed arrow connecting to MPNN block

---

### Panel (b): Hierarchical Context Modulation

**Background:** Light orange/yellow tinted panel to highlight this is the novel contribution

**Top section - Context Embeddings:**

1. **L1: Program** (GREEN box)
   - Label: "L1: Program"
   - Subtext: "128-dim"
   - Show embedding table visual (stacked colored rows)
   - Examples below: "EGFR, DRD2, BACE1..."
   - Represents target protein identity

2. **L2: Assay** (TEAL box)
   - Label: "L2: Assay"
   - Subtext: "64-dim"
   - Show embedding table visual
   - Examples below: "IC50, Ki, EC50..."
   - Represents assay type

3. **L3: Round** (PURPLE box)
   - Label: "L3: Round"
   - Subtext: "32-dim"
   - Show embedding table visual
   - Examples below: "Round 1, 2, 3..."
   - Represents temporal context

4. **Concatenation Symbol** [ || ]
   - Three arrows from L1, L2, L3 converging
   - Label: "224-dim"
   - Shows embeddings being concatenated

**Middle section - FiLM Modulation (ORANGE, prominent):**

Large orange box containing:

1. **Title:** "FiLM Modulation"

2. **γ MLP** (lighter orange sub-box)
   - Label: "γ MLP"
   - Subtext: "224→512"
   - Produces scale parameters

3. **β MLP** (lighter orange sub-box)
   - Label: "β MLP"
   - Subtext: "224→512"
   - Produces shift parameters

4. **Operations:**
   - Circle with ⊙ (element-wise multiply): γ output × h_mol
   - Circle with + (addition): result + β output

5. **h_mol input arrow** coming from Panel (a)

6. **Output: h_mod** (white box inside FiLM)
   - Label: "h_mod"
   - Subtext: "512-dim"

7. **Equation below FiLM box:**
   - h_mod = γ(c) ⊙ h_mol + β(c)

**Bottom section - Prediction:**

1. **MLP Head** (RED box)
   - Label: "MLP Head"
   - Show layer stack: 512 → 256 → 128 → 1
   - Annotation: "×N tasks" (multi-task)
   - ReLU activations, dropout

2. **Arrow →**

3. **Output ŷ** (light gray)
   - Label: "ŷ"
   - Represents predicted activity

---

### Legend (bottom of figure)

Horizontal row of color-coded boxes with labels:
- Light gray: "Input"
- Blue: "L0: MPNN"
- Green: "L1: Program"
- Teal: "L2: Assay"
- Purple: "L3: Round"
- Orange: "FiLM"
- Red: "Prediction"

---

### Key Visual Elements to Include

1. **Data flow arrows:** Solid black arrows showing the forward pass
2. **Dashed arrows:** Connect detail boxes to main components
3. **Embedding tables:** Visualize as stacked horizontal bars
4. **Layer stacks:** Show neural network layers as stacked rectangles
5. **Mathematical notation:** Use proper LaTeX-style subscripts and Greek letters
6. **Molecule visualization:** Simple 2D structure with atoms and bonds
7. **Dimension annotations:** Show tensor dimensions at each stage

---

### Text to Include

- All dimension annotations (70-dim, 9-dim, 512-dim, 128-dim, 64-dim, 32-dim, 224-dim)
- Component names (SMILES, Graph, MPNN, FiLM, MLP Head)
- Mathematical equations for message passing and FiLM
- Example context values (EGFR, IC50, Round 1)
- Panel labels: (a) Molecular Encoding, (b) Hierarchical Context Modulation
