"""Print the Pipeline Provenance map: every table and figure label in the two
manuscript files, the file it is typeset from, and the script that writes that file.

Run from the repository root:  python3 experiments/provenance_map.py [--check]
With --check, exits non-zero if any label has no known generator.
"""
import os
import re
import sys

MANUSCRIPT_FILES = ["springer_project/TabStress.tex", "springer_project/TabStress_Supplementary.tex"]

GENERATORS = {
    "tab_inventory_main.tex": ["experiments/build_revision_tables.py"],
    "tab_accounting.tex": ["experiments/build_revision_tables.py"],
    "tab_combined.tex": ["experiments/analyze_revision.py"],
    "tab_gaussian_principal.tex": ["experiments/build_revision_tables.py"],
    "tab_categorical.tex": ["experiments/build_revision_tables.py"],
    "tab_ablation.tex": ["experiments/build_manuscript_tables.py"],
    "tab_sensitivity.tex": ["experiments/analyze_sensitivity.py", "experiments/build_revision_tables.py"],
    "tab_sensitivity_extra.tex": ["experiments/analyze_sensitivity.py"],
    "tab_protocol.tex": ["experiments/build_revision_tables.py"],
    "tab_tabicl_configs.tex": ["experiments/build_revision_tables.py"],
    "tab_foundation.tex": ["experiments/analyze_foundation.py"],
    "tab_search_space.tex": ["experiments/build_revision_tables.py"],
    "tab_protocol_b_tests.tex": ["experiments/build_revision_tables.py"],
    "tab_pol_ordering.tex": ["experiments/run_pol_ordering_sensitivity.py"],
    "tab_cleaning_counts.tex": ["experiments/dataset_cleaning_counts.py"],
    "tab_wilcoxon.tex": ["experiments/build_manuscript_tables.py"],
    "tab_trajectories.tex": ["experiments/build_manuscript_tables.py"],
    "tab_seeds.tex": ["experiments/build_manuscript_tables.py"],
    "tab_ranks.tex": ["experiments/build_manuscript_tables.py"],
    "tab_native.tex": ["experiments/build_manuscript_tables.py"],
    "tab_inventory.tex": ["experiments/build_manuscript_tables.py"],
    "tab_friedman.tex": ["experiments/build_manuscript_tables.py"],
    "tab_endurance.tex": ["experiments/build_manuscript_tables.py"],
    "tab_budget.tex": ["experiments/build_manuscript_tables.py"],
    "tab_accuracy_grids.tex": ["experiments/build_manuscript_tables.py"],
    "atlas_primary.tex": ["experiments/build_manuscript_tables.py"],
    "methodology_flowchart.tex": ["hand-typed TikZ (sections/methodology_flowchart.tex)"],
    "fig_curves.png": ["experiments/build_revision_figures.py"],
    "fig_overall.png": ["experiments/build_revision_figures.py"],
    "fig_ranking.png": ["experiments/build_revision_figures.py"],
    "fig_knee.png": ["experiments/build_revision_figures.py"],
    "curves_": ["experiments/build_primary_figures.py"],
    "cd_": ["experiments/build_primary_figures.py"],
    "macro_primary.png": ["experiments/build_primary_figures.py"],
}

INLINE_RE = re.compile(r"%% ---- begin inlined (\S+) ----")
END_RE = re.compile(r"%% ---- end inlined (\S+) ----")
LABEL_RE = re.compile(r"\\label\{((?:tab|fig):[^}]+)\}")
ENV_RE = re.compile(r"\\begin\{(table\*?|figure\*?)\}(.*?)\\end\{\1\}", re.S)
GRAPHIC_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def source_of_label(text, position, env_spans):
    for start, end, graphic in env_spans:
        if start <= position < end and graphic:
            return graphic
    inline = None
    for match in INLINE_RE.finditer(text[:position]):
        inline = match.group(1)
        closing = END_RE.search(text, match.end())
        if closing and closing.start() > position:
            return inline
    return "hand-typed"


def labels_in(path):
    text = open(path, encoding="utf-8").read()
    env_spans = []
    for match in ENV_RE.finditer(text):
        graphic = GRAPHIC_RE.search(match.group(2))
        env_spans.append((match.start(), match.end(), graphic.group(1) if graphic else None))
    rows = []
    for match in LABEL_RE.finditer(text):
        source = source_of_label(text, match.start(), env_spans)
        rows.append((os.path.basename(path), match.group(1), source))
    return rows


def generator_for(source):
    base = os.path.basename(source)
    if base in GENERATORS:
        return ", ".join(GENERATORS[base])
    for prefix in ("curves_", "cd_"):
        if base.startswith(prefix):
            return ", ".join(GENERATORS[prefix])
    if source == "hand-typed":
        return "hand-typed in the .tex file (no generator)"
    return "UNMAPPED"


def main():
    rows = []
    for path in MANUSCRIPT_FILES:
        rows.extend(labels_in(path))
    print("| Manuscript file | Label | Typeset from | Generating script |")
    print("|---|---|---|---|")
    unmapped = 0
    for filename, label, source in rows:
        generator = generator_for(source)
        unmapped += generator == "UNMAPPED"
        print(f"| {filename} | `{label}` | `{source}` | {generator} |")
    if "--check" in sys.argv and unmapped:
        print(f"UNMAPPED labels: {unmapped}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
