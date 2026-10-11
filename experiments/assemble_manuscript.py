"""Builds the single-file Springer submission springer_project/TabStress.tex and its
bibliography TabStress.bib. Springer forbids \\input, so every included file is inlined."""

import os
import re

ROOT = "springer_project"
SOURCE = os.path.join(ROOT, "build", "TabStress_main.tex")
TARGET = os.path.join(ROOT, "TabStress.tex")
SUPP_SOURCE = os.path.join(ROOT, "build", "TabStress_supp_main.tex")
SUPP_TARGET = os.path.join(ROOT, "TabStress_Supplementary.tex")
SEMINAL_BIB = "paper/references.bib"
CORPUS_BIB = "literature/papers_2025_2026.bib"
REVISED_BIB = "springer_project/revision/references_revised_seminal.bib"  # verified corrections take precedence
BIB_TARGET = os.path.join(ROOT, "TabStress.bib")
INPUT_RE = re.compile(r"^\s*\\input\{([^}]+)\}\s*$")
IF_RE = re.compile(r"^%%IF(NOT)?\s+(\S+)\s*$")
CITE_RE = re.compile(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]*)\}")


def block_open(key):
    """Truth value of an %%IF key against the analysis outputs (missing key or file = false)."""
    import json
    src, _, path = key.partition(".")
    if src not in VALUE_SOURCES or not os.path.exists(VALUE_SOURCES[src]):
        return False
    try:
        return bool(lookup(json.load(open(VALUE_SOURCES[src])), path))
    except (KeyError, IndexError, ValueError):
        return False


def inline(path, depth=0, dropped=None):
    """Inline every \\input and resolve %%IF ... %%ENDIF blocks on the way, so that the inputs of
    a dropped block are never read."""
    if depth > 5:
        raise RecursionError(f"\\input nesting too deep at {path}")
    dropped = [] if dropped is None else dropped
    out, stack = [], []
    with open(path) as fh:
        for line in fh:
            m = IF_RE.match(line.strip())
            if m:
                ok = block_open(m.group(2))
                if m.group(1):          # %%IFNOT: fallback text, used only while the data are missing
                    ok = not ok
                    if ok:
                        dropped.append(m.group(2))
                elif not ok:
                    dropped.append(m.group(2))
                stack.append(ok)
                continue
            if line.strip() == "%%ENDIF":
                if not stack:
                    raise SystemExit(f"%%ENDIF without %%IF in {path}")
                stack.pop()
                continue
            if not all(stack):
                continue
            m = INPUT_RE.match(line)
            if not m:
                out.append(line)
                continue
            child = m.group(1)
            child = child if child.endswith(".tex") else child + ".tex"
            child_path = os.path.normpath(os.path.join(ROOT, child))
            out.append(f"%% ---- begin inlined {child} ----\n")
            out.append(inline(child_path, depth + 1, dropped))
            out.append(f"%% ---- end inlined {child} ----\n")
    if stack:
        raise SystemExit(f"unbalanced %%IF/%%ENDIF in {path}")
    return "".join(out)


def entries(path):
    out = {}
    for chunk in re.split(r"(?=^@)", open(path).read(), flags=re.M):
        m = re.match(r"@\w+\{([^,]+),", chunk.strip())
        if m:
            out.setdefault(m.group(1).strip(), chunk.strip())
    return out


def corpus_authors_to_bibtex(entry):
    """The corpus bib separates full names ("First Last, First Last") by commas, which
    BibTeX would read as "Last, First". Seminal entries are already in BibTeX form and
    must never pass through this function: "Herbold, Steffen" is one author."""
    def repl(m):
        names = [n.strip() for n in m.group(1).split(",") if n.strip()]
        return "author={" + " and ".join(names) + "}"
    return re.sub(r"author=\{([^}]*)\}", repl, entry, count=1)


def cited_keys(tex):
    body = re.sub(r"(?m)(?<!\\)%.*$", "", tex)
    return sorted({k.strip() for g in CITE_RE.findall(body) for k in g.split(",") if k.strip()})


def process_corpus_entry(entry):
    entry = corpus_authors_to_bibtex(entry)
    if not re.search(r"\bdoi\s*=", entry, re.IGNORECASE):
        m_url = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", entry)
        if m_url:
            doi_val = f"10.48550/arXiv.{m_url.group(1)}"
            entry = re.sub(r"(\n\s*\}\s*)$", rf",\n  doi={{{doi_val}}}\1", entry)
    return entry


def build_bib(tex):
    """SQGASpr.bib holds exactly the cited entries: seminal sources first, then the
    post-2024 corpus. A cited key found in neither stops the build."""
    seminal, corpus = entries(SEMINAL_BIB), entries(CORPUS_BIB)
    revised = entries(REVISED_BIB) if os.path.exists(REVISED_BIB) else {}
    out, missing = [], []
    for key in cited_keys(tex):
        if key in revised:
            # sn-mathphys.bst typesets url fields outside \url, where an underscore breaks the build
            out.append("\n".join(l for l in revised[key].splitlines() if not (l.strip().startswith("url") and ("_" in l or len(l) > 90))))
        elif key in seminal:
            out.append(seminal[key])
        elif key in corpus:
            out.append(process_corpus_entry(corpus[key]))
        else:
            missing.append(key)
    if missing:
        raise SystemExit(f"cited keys absent from {SEMINAL_BIB} and {CORPUS_BIB}: {missing}")
    with open(BIB_TARGET, "w") as fh:
        fh.write("\n\n".join(out) + "\n")
    return len(out)


TOKEN_RE = re.compile(r"@@\{([^|}]+)\|([^}]+)\}@@")
VALUE_SOURCES = {
    "orig": "results/analysis/audit/audit_numbers.json",
    "prim": "results/analysis/audit_primary/audit_numbers.json",
    "extra": "results/analysis/audit_primary/extras.json",
    "ops": "results/analysis/audit_primary/operator_audit.json",
    "fix": "results/analysis/audit_primary/fix_extras.json",
    "rev": "results/analysis/revision/revision.json",
    "found": "results/analysis/revision/foundation.json",
    "sens": "results/analysis/revision/sensitivity_extra.json",
}


def lookup(data, path):
    cur = data
    for part in path.split("."):
        part = part.replace("~", "|")  # analysis keys such as "mrr|symmetric_label" are written mrr~symmetric_label
        if cur is None:
            raise KeyError(path)
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif part in cur:
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


def fmt_value(v, spec):
    from decimal import ROUND_HALF_UP, Decimal
    if spec == "int":
        return f"{int(round(float(v))):,}".replace(",", "{,}")
    if spec == "r2":
        return str(Decimal(repr(float(v))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    if spec.startswith("pct"):
        return f"{100 * float(v):.{int(spec[3:])}f}"
    if spec == "p":               # p-values: scientific below 0.001, three decimals otherwise
        return fmt_value(v, "sci") if float(v) < 1e-3 else f"{float(v):.3f}"
    if spec == "sci":
        m, e = f"{float(v):.1e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    if spec == "str":
        return str(v)
    if spec in ("count", "Count"):  # a count that may be zero: "none of the ..." rather than "zero of the ..."
        word = "none" if int(v) == 0 else fmt_value(v, "word")
        return word.capitalize() if spec == "Count" else word
    if spec == "Word":
        return fmt_value(v, "word").capitalize()
    if spec == "condlist":
        names = {"symmetric_label": "symmetric label noise", "class_conditional": "class-conditional label noise",
                 "gaussian_feature": "Gaussian feature noise", "mcar_imputed": "imputed masking",
                 "mcar_native": "native masking"}
        items = [names.get(x, x) for x in v]
        if not items:
            return "none"
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    if spec == "word":
        words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                 "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                 "nineteen", "twenty"]
        return words[int(v)]
    if spec == "seeds":           # [42] -> "seed 42", [42, 43] -> "seeds 42 and 43", [42, 43, 44] -> "seeds 42, 43 and 44"
        items = [str(int(x)) for x in v]
        if len(items) == 1:
            return f"seed {items[0]}"
        return "seeds " + ", ".join(items[:-1]) + " and " + items[-1]
    if spec == "seedcount":       # 1 -> "one seed", 3 -> "three seeds"
        return f"{fmt_value(v, 'word')} seed" + ("" if int(v) == 1 else "s")
    if spec == "ord":
        return ["", "first", "second", "third", "fourth", "fifth", "sixth"][int(v)]
    if spec == "name":
        return {"cart": "CART", "randomforest": "RF", "xgboost": "XGBoost", "lightgbm": "LightGBM",
                "catboost": "CatBoost", "mlp": "MLP", "tabicl": "TabICL"}.get(str(v), str(v))
    return format(float(v), spec)


def substitute(tex):
    """Replace @@{source.path|format}@@ by the computed value; a missing key stops the build."""
    import json
    data = {k: json.load(open(p)) for k, p in VALUE_SOURCES.items() if os.path.exists(p)}

    def repl(m):
        key, spec = m.group(1).strip(), m.group(2).strip()
        src, _, path = key.partition(".")
        if src not in data:
            raise SystemExit(f"value source '{src}' not available for token {m.group(0)}")
        try:
            return fmt_value(lookup(data[src], path), spec)
        except (KeyError, IndexError, ValueError) as exc:
            raise SystemExit(f"cannot resolve token {m.group(0)}: {exc!r}")
    return TOKEN_RE.sub(repl, tex)


def assemble(source, target, dropped):
    flat = substitute(inline(source, dropped=dropped))
    with open(target, "w") as fh:
        fh.write(flat)
    leftover = [l for l in flat.splitlines() if INPUT_RE.match(l)]
    print(f"{target}: {flat.count(chr(10))} lines, unresolved inputs: {len(leftover)}")
    return flat


def main():
    """Builds the main article and the supplementary material; both share one bibliography file."""
    dropped = []
    flat = assemble(SOURCE, TARGET, dropped)
    supp = assemble(SUPP_SOURCE, SUPP_TARGET, dropped) if os.path.exists(SUPP_SOURCE) else ""
    with open(os.path.join(ROOT, "build", "dropped_blocks.txt"), "w") as fh:
        fh.write("\n".join(dropped) + ("\n" if dropped else ""))
    if dropped:
        print(f"PROVISIONAL BUILD: dropped blocks for {sorted(set(dropped))}")
    n_bib = build_bib(flat + "\n" + supp)
    print(f"{BIB_TARGET}: {n_bib} cited entries")


if __name__ == "__main__":
    main()
