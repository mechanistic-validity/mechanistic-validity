"""Generate the Criterion Audit pages from the NeurIPS submission's own sections.

The site drifted from the paper once already. The fix is not to copy the tables by hand
a second time but to render them from the file the submission compiles:

    mechanistic-validity-NEW2/paper/generated_neurips/<claim>_section.tex

That directory differs from paper/generated/ for all sixteen claims, and only
generated_neurips is the submitted version. Nothing here is authored; if the audit record
changes, rerun this.

    python3 scripts/gen_audit_pages.py
"""
import json, re, sys
from pathlib import Path

SRC = Path.home() / "Documents/GitHub/mechanistic-validity-NEW2/paper/generated_neurips"
OUT = Path(__file__).resolve().parents[1] / "src/content/docs/framework/audits"
# Citation key -> display label and URL, derived from the submission bibliography
# (mechanistic-validity-NEW2/paper/references.bib). Committed rather than rebuilt so a
# rerun renders the same links; keys with no DOI, arXiv id or URL render unlinked.
CITES = json.load(open(Path(__file__).with_name("citemap.json")))

# The contested criterion judgments, verbatim from the submission's sensitivity-analysis
# table. Only the claims listed there carry one.
CONTESTED = {
 "workspace": [("I6", "Inconclusive or Partially confirmed",
   "Whether a crossing over layer bands meets a criterion asking for two mechanisms")],
 "greater_than": [("E4", "Disconfirmed or Untested",
   "Whether the later cross-model test was run and failed, or reports a different circuit")],
 "othello": [("M5", "Partially confirmed or Untested",
   "Whether one instrument's known-positive licenses a sensitivity verdict for another"),
   ("I6", "Untested or Not applicable",
   "Whether a design supplying one mechanism and one behavior leaves double dissociation unattempted or unaskable")],
 "induction_broad": [("I5", "Disconfirmed or Inconclusive",
   "Whether a rival named and never excluded is a failure or an absence")],
 "gender": [("I6", "Untested or Not applicable",
   "Whether a design supplying one mechanism and one behavior leaves double dissociation unattempted or unaskable")],
 "knowledge_neurons": [("I6", "Untested or Not applicable",
   "Whether a design supplying one mechanism and one behavior leaves double dissociation unattempted or unaskable")],
 "probing": [("I6", "Untested or Not applicable",
   "Whether a design supplying one mechanism and one behavior leaves double dissociation unattempted or unaskable")],
}

TITLES = {
 "ioi": ("IOI Circuit", "examples-ioi"), "induction": ("Induction Heads", "examples-induction-heads"),
 "induction_broad": ("Induction Heads (General ICL)", "examples-induction-heads-icl"),
 "copy_suppression": ("Copy Suppression", "examples-copy-suppression"),
 "greater_than": ("Greater-Than Circuit", "examples-greater-than"),
 "grokking": ("Grokking / Modular Addition", "examples-grokking"),
 "docstring": ("Docstring Circuit", "examples-docstring"),
 "gender": ("Gender Bias Circuits", "examples-gender-bias"),
 "othello": ("Othello Board State", "examples-othello"),
 "probing": ("Probing Classifiers", "examples-probing"),
 "sae": ("SAE Features", "examples-sae-features"),
 "successor_heads": ("Successor Heads", "examples-successor-heads"),
 "superposition": ("Superposition", "examples-superposition"),
 "knowledge_neurons": ("Knowledge Neurons", "examples-knowledge-neurons"),
 "workspace": ("Global Workspace", "examples-global-workspace"),
 "refusal": ("Refusal Direction", "examples-refusal-direction"),
}

def detex(s):
    def _cite(m):
        out = []
        for k in m.group(1).split(","):
            e = CITES.get(k.strip())
            if not e:
                out.append(k.strip())
            elif e.get("url"):
                out.append(f"[{e['label']}]({e['url']})")
            else:
                out.append(e["label"])
        return "(" + "; ".join(out) + ")"
    s = re.sub(r"\\cite[a-z]*\{([^}]*)\}", _cite, s)
    s = re.sub(r"\\textbf\{(.*?)\}", r"**\1**", s)
    s = re.sub(r"\\emph\{(.*?)\}|\\textit\{(.*?)\}", lambda m: "*" + (m.group(1) or m.group(2)) + "*", s)
    s = re.sub(r"\\underline\{\\hspace\{[^}]*\}\}", "\\\\_\\\\_\\\\_", s)
    s = s.replace("$\\to$", "→").replace("\\ldots", "…").replace("---", "—").replace("--", "–")
    s = s.replace("``", "\u201c").replace("''", "\u201d").replace("\\%", "%").replace("\\&", "&")
    s = s.replace("\\S", "§").replace("\\times", "×").replace("~", " ")
    s = re.sub(r"\\multicolumn\{\d+\}\{[^{}]*\}\{(.*?)\}", r"\1", s)
    for pat, ch in ((r'\\"\{?\\?i\}?', "ï"), (r'\\"\{?o\}?', "ö"), (r'\\"\{?u\}?', "ü"),
                    (r"\\'\{?e\}?", "é"), (r"\\'\{?a\}?", "á"), (r"\\`\{?e\}?", "è"),
                    (r"\\o\b", "ø"), (r"\\H\s*\{?o\}?", "ő")):
        s = re.sub(pat, ch, s)
    s = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", s)     # any leftover one-arg macro
    s = re.sub(r"\\[a-zA-Z]+", "", s)                     # bare macros
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()

def rows_of(block):
    """Split a tabular body into logical rows on unescaped \\\\."""
    out = []
    for raw in re.split(r"\\\\(?!\w)", block):
        raw = re.sub(r"\\(midrule|toprule|bottomrule|addlinespace(\[[^\]]*\])?|endfirsthead|endhead)", "", raw)
        if raw.strip():
            out.append(raw)
    return out

def convert(stem):
    tex = (SRC / f"{stem}_section.tex").read_text()
    title, twin = TITLES[stem]
    md = [f'---\ntitle: "{title}"\ndescription: "Criterion audit of the {title.lower()} claim, '
          f'as submitted."\n---\n', f"# {title}\n"]

    for lbl, tag in (("Source", "Source"), ("Description", "Description")):
        m = re.search(r"\\textbf\{%s\.\}(.*?)(?=\n\n|\\vspace|\\clearpage)" % tag, tex, re.S)
        if m:
            md.append(f"**{lbl}.** {detex(m.group(1))}\n")

    # The 36-criterion table is a longtable whose caption sits inside the environment;
    # the other two are table floats wrapping a tabular. Find the environment, then pull
    # the caption and the row body out of it, rather than assuming an order.
    blocks = re.findall(r"\\begin\{(longtable|tabular)\}(?:\[[^\]]*\])?\{[^\n]*\n(.*?)\\end\{\1\}", tex, re.S)
    caps = re.findall(r"\\caption\{(.*?)\}\s*(?:\\\\|\n)", tex, re.S)
    RANK = {"Readings of the claim": 0, "Verdict": 1, "36-criterion audit": 2}
    rendered = []
    for i, (env, body) in enumerate(blocks):
        cap = detex(caps[i]) if i < len(caps) else ""
        low = cap.lower()
        heading = ("36-criterion audit" if "36-criterion" in low
                   else "Readings of the claim" if "readings" in low
                   else "Verdict" if "verdict" in low else f"Table {i+1}")
        body = re.sub(r"\\caption\{.*?\}\s*\\\\", "", body, flags=re.S)
        body = re.sub(r"\\label\{[^}]*\}", "", body)
        chunk = [f"## {heading}\n"]
        if cap:
            chunk.append(f"{cap}\n")
        cells = []
        for r in rows_of(body):
            # A \multicolumn row is a section label (Construct Validity, Total, Verdict).
            # Its column spec carries nested braces, so match the whole row rather than
            # trying to rewrite the macro in place.
            mc = re.match(r"\s*\\multicolumn\{(\d+)\}\{.*?\}\{(.*)\}\s*$", r, re.S)
            if mc:
                label = detex(mc.group(2))
                if label:
                    # Some section rows already carry their own bold (Total:, Verdict:).
                    cells.append([label if "**" in label else "**" + label + "**"])
                continue
            parts = [detex(c) for c in r.split("&")]
            if any(p for p in parts):
                cells.append(parts)
        if not cells:
            continue
        cells = [c for j, c in enumerate(cells) if j == 0 or c != cells[0]]
        width = max(len(c) for c in cells)
        header = cells[0] + [""] * (width - len(cells[0]))
        chunk.append("| " + " | ".join(h or " " for h in header) + " |")
        chunk.append("|" + "---|" * width)
        for c in cells[1:]:
            c = c + [""] * (width - len(c))
            chunk.append("| " + " | ".join(x.replace("|", "\\|") or " " for x in c) + " |")
        chunk.append("")
        rendered.append((RANK.get(heading, 9), chunk))
    if stem in CONTESTED:
        sens = ["## Sensitivity\n",
                "Criterion judgments this audit record leaves contested: each carries an argument on "
                "both sides, and the status shipped is the one the record settled on."
                + (" At Partially confirmed this claim reaches Validated.\n" if stem == "workspace" else "\n"),
                "| Criterion | In tension | What would settle it |", "|---|---|---|"]
        for cid, tension, settle in CONTESTED[stem]:
            sens.append(f"| {cid} | {tension} | {settle} |")
        sens.append("")
        rendered.append((2.5, sens))
    for _, chunk in sorted(rendered, key=lambda x: x[0]):
        md.extend(chunk)

    # A heading rather than a bare footer, so it appears in the page's table of contents
    # and a reader can see there is something below to scroll to.
    md.append(f"\n## Exploratory Lens Analysis\n\nAn exploratory reading of this claim through "
              f"the framework's five lenses, written for this site and not part of the paper, is at "
              f"[{title} — exploratory lens analysis](/mechanistic-validity/framework/examples/{twin}/).\n")
    return "\n".join(md)

OUT.mkdir(parents=True, exist_ok=True)
n = 0
for stem in TITLES:
    if not (SRC / f"{stem}_section.tex").exists():
        print("  missing:", stem); continue
    (OUT / f"{stem}.md").write_text(convert(stem)); n += 1
print(f"{n} audit pages written to {OUT}")
