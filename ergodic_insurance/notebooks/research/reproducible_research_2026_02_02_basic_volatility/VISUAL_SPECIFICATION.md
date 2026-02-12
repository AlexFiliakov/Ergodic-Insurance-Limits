# Visual Specification for Publication Charts

Consistent visual standards for all figures in the ergodic insurance research paper.
Charts target 300 DPI, full-color reproduction, and colorblind accessibility.

---

## Color Palette

Based on the Okabe-Ito palette (widely recognized as colorblind-safe).

| Role | Name | Hex | Usage |
|------|------|-----|-------|
| Primary / Insured | Blue | `#0072B2` | Insured trajectories, primary data series |
| Secondary / Uninsured | Orange | `#E69F00` | Uninsured/no-insurance trajectories |
| Positive / Growth | Green | `#009E73` | Growth lift, insurance recovery, positive outcomes |
| Negative / Ruin | Vermillion | `#D55E00` | Ruin events, retained losses, negative outcomes |
| Tertiary | Purple | `#CC79A7` | Third data category, supplementary series |
| Light accent | Sky blue | `#56B4E9` | Light fills, secondary insured series |
| Highlight | Yellow | `#F0E442` | Sparse annotations, callout backgrounds |
| Reference | Gray | `#999999` | Reference lines, no-insurance baseline, grid |
| Text | Dark | `#333333` | Axis labels, annotations, emphasis text |

### Deductible Color Mapping

When plotting multiple deductible levels on the same chart:

| Deductible | Color | Hex |
|------------|-------|-----|
| $0 (Guaranteed Cost) | Blue | `#0072B2` |
| $100K | Sky blue | `#56B4E9` |
| $250K | Orange | `#E69F00` |
| $500K | Vermillion | `#D55E00` |
| No Insurance | Gray | `#999999` |

### Fill Opacity Rules

- **Fan chart bands** (wide range, e.g., P5-P95): `alpha = 0.10`
- **Fan chart bands** (mid range, e.g., P25-P75): `alpha = 0.20`
- **Fan chart bands** (narrow range, e.g., P40-P60): `alpha = 0.35`
- **Area fills** (e.g., distribution KDEs): `alpha = 0.25`
- **Bar chart fills**: `alpha = 0.85`
- **Spaghetti line overlays**: `alpha = 0.05`

---

## Typography

### Font Family

Use **serif** fonts to match the LaTeX paper body (Computer Modern):

```python
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['CMU Serif', 'Computer Modern', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'
```

### Font Sizes

| Element | Size (pt) | Weight |
|---------|-----------|--------|
| Figure title (suptitle) | 13 | Bold |
| Panel/subplot title | 11 | Regular |
| Axis labels | 10 | Regular |
| Tick labels | 9 | Regular |
| Legend text | 9 | Regular |
| Annotations | 9 | Regular |
| Table headers | 10 | Bold |
| Table cells | 9 | Regular |

---

## Line Styles

### Line Widths

| Purpose | Width (pt) |
|---------|-----------|
| Emphasis / primary data line | 2.5 |
| Standard data line | 1.5 |
| Reference / baseline (zero line, threshold) | 1.0 |
| Grid lines | 0.5 |
| Spaghetti / background path lines | 0.3 |

### Dash Patterns

| Style | Usage |
|-------|-------|
| Solid (`-`) | Insured data, primary series, optimal values |
| Dashed (`--`) | Uninsured/no-insurance data, secondary series |
| Dotted (`:`) | Reference lines (ruin threshold, zero baseline) |
| Dash-dot (`-.`) | Tertiary comparison, confidence bounds |

### Insured vs. Uninsured Convention

- **Insured**: Solid line, blue color family
- **Uninsured**: Dashed line, orange color family

This convention is maintained across ALL charts for instant recognition.

---

## Axes and Grid

### Spines

- **Show**: Bottom and left spines only
- **Hide**: Top and right spines
- **Color**: `#333333`
- **Width**: 0.8 pt

```python
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
```

### Grid

- **Visibility**: On by default, rendered behind data
- **Style**: Dashed (`--`)
- **Color**: `#CCCCCC`
- **Alpha**: 0.3
- **Width**: 0.5 pt
- **Major grid only** (no minor grid)

```python
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
```

---

## Figure Sizing

Targets for the LaTeX paper:

| Layout | Width (in) | Typical Height (in) |
|--------|-----------|-------------------|
| Single column | 3.5 | 2.5-3.0 |
| 1.5 column | 5.5 | 3.5-4.5 |
| Double column / full width | 7.0 | 4.0-5.5 |
| Double column, 2-panel | 7.0 | 3.0-3.5 |
| Heatmap / table figure | 7.0 | 4.0-5.0 |

All figures at **300 DPI** for print quality.

```python
plt.rcParams['figure.dpi'] = 150       # Screen display
plt.rcParams['savefig.dpi'] = 300       # Saved files
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
```

---

## Annotations and Callouts

- Use `ax.annotate()` with an arrow from callout to data point
- Arrow style: `arrowprops=dict(arrowstyle='->', color='#333333', lw=0.8)`
- Text box: light yellow background (`#F0E442`, alpha=0.7), thin border (`#333333`, lw=0.5)
- Keep annotations sparse; use only where the reader benefits from explicit labeling

---

## Chart-Specific Guidelines

### Fan Charts
- Use 3 nested bands: P5-P95 (lightest), P25-P75 (medium), P40-P60 (darkest)
- Bold median line (2.5 pt)
- Log scale on y-axis for wealth trajectories
- Horizontal dotted line at ruin threshold

### Heatmaps
- Use `matplotlib.colors.ListedColormap` with the deductible palette
- White text on dark cells, dark text on light cells
- Annotate each cell with the numeric value
- Grid lines between cells (white, 2pt)

### Bar Charts
- Moderate bar width (0.6-0.8 for single, 0.35 for grouped)
- Thin edge line (`edgecolor='white'`, lw=0.5) for separation
- Horizontal reference line at zero (dotted gray)

### Distribution Plots
- KDE or histogram with 80-100 bins
- Fill under curve with alpha=0.25
- Vertical line at mean (solid) and median (dashed)
- Annotate key statistics inline

### Tables as Figures
- Use `ax.table()` or manual text placement
- Alternate row shading (white / `#F5F5F5`)
- Bold optimal values, gray suboptimal
- Right-align numeric columns

---

## Output Format

- **Format**: PNG (300 DPI) for paper drafts; PDF for final submission
- **Naming**: `output/publication/{chart_name}.png`
- **Background**: White (`#FFFFFF`)
- **No figure border** around saved files

---

## Quick-Reference: Matplotlib rcParams

```python
RCPARAMS = {
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#CCCCCC',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
}
```
