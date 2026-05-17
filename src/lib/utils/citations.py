"""Citation instances for the exhaustive task catalog.

Every citation has a verifiable URL and exact table/figure/section location.
The Citation class is duplicated from task_reference_baselines.py so this
package can be used standalone.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    paper: str
    venue: str
    arxiv_id: str
    location: str

    @property
    def pdf_url(self) -> str:
        if not self.arxiv_id:
            return ""
        return f"https://arxiv.org/pdf/{self.arxiv_id}"

    @property
    def html_url(self) -> str:
        if not self.arxiv_id:
            return ""
        return f"https://arxiv.org/html/{self.arxiv_id}"

    def __repr__(self) -> str:
        return f"{self.paper} ({self.venue}) [{self.location}] {self.pdf_url}"


# ---------------------------------------------------------------------------
# Published circuit papers
# ---------------------------------------------------------------------------

WANG_IOI = Citation(
    paper="Wang et al., Interpretability in the Wild: IOI in GPT-2 Small",
    venue="ICLR 2023",
    arxiv_id="2211.00593v1",
    location="Section 2 (task) + Section 4 (faithfulness)",
)

HANNA_GT = Citation(
    paper="Hanna et al., How does GPT-2 compute greater-than over tokens?",
    venue="NeurIPS 2023",
    arxiv_id="2305.00586v2",
    location="Section 2 (quantitative eval) + Section 3.2 (circuit eval)",
)

LAZO_SVA = Citation(
    paper="Lazo et al., Identifying a Circuit for Verb Conjugation in GPT-2",
    venue="arXiv 2025",
    arxiv_id="2506.22105v1",
    location="Table 3 (Section 4.2) + Figure 1",
)

OLSSON_INDUCTION = Citation(
    paper="Olsson et al., In-context Learning and Induction Heads",
    venue="arXiv 2022",
    arxiv_id="2209.11895",
    location="Section 2 (mechanism) + Table 5 Appendix (GPT-2 Small head list)",
)

MATHWIN_GENDER = Citation(
    paper="Mathwin, Preliminary Circuit for Gendered Pronouns in GPT-2 Small",
    venue="MATS Hackathon 2023 (unpublished)",
    arxiv_id="",
    location="hackathon report; ACDC-discovered circuit >= full model",
)

GARCIA_CARRASCO_ACRONYM = Citation(
    paper="Garcia-Carrasco et al., How does GPT-2 Predict Acronyms?",
    venue="AISTATS 2024",
    arxiv_id="2405.04156",
    location="Table 1 (8-head circuit, 3 role groups)",
)

MCDOUGALL_COPY_SUPP = Citation(
    paper="McDougall et al., Copy Suppression: Comprehensively Understanding an Attention Head",
    venue="arXiv 2023",
    arxiv_id="2310.04625",
    location="Section 2 (L10H7 characterization, 76.9% of impact explained)",
)

CONMY_ACDC = Citation(
    paper="Conmy et al., Towards Automated Circuit Discovery for Mechanistic Interpretability",
    venue="NeurIPS 2023",
    arxiv_id="2304.14997v3",
    location="Table 2 (Appendix D, AUC) + Abstract (68/32k edges)",
)

# ---------------------------------------------------------------------------
# Roadmap papers — tasks with published circuits or partial characterizations
# ---------------------------------------------------------------------------

MERULLO_COLORED = Citation(
    paper="Merullo et al., Circuit Component Reuse Across Tasks in Transformers",
    venue="arXiv 2023",
    arxiv_id="2310.08744",
    location="Section 4.1 (colored objects task, 78% IOI circuit overlap)",
)

GEVA_FACTUAL = Citation(
    paper="Geva et al., Dissecting Recall of Factual Associations in Auto-Regressive LMs",
    venue="EMNLP 2023",
    arxiv_id="2304.14767",
    location="Section 3 (3-step: subject enrichment, relation propagation, attribute extraction)",
)

TIGGES_SENTIMENT = Citation(
    paper="Tigges et al., Linear Representations of Sentiment in Large Language Models",
    venue="arXiv 2023",
    arxiv_id="2310.15154",
    location="Section 3 (linear probe + causal direction, 76% accuracy mediated)",
)

FRIEDMAN_BRACKETS = Citation(
    paper="Friedman et al., Learning Transformer Programs",
    venue="NeurIPS 2023",
    arxiv_id="2306.01128",
    location="Section 4.2 (bracket matching, structured prediction tasks)",
)

HEIMERSHEIM_DOCSTRING = Citation(
    paper="Heimersheim & Janiak, A Circuit for Python Docstrings in a 4-Layer Attention-Only Transformer",
    venue="Alignment Forum 2023",
    arxiv_id="",
    location="https://www.alignmentforum.org/posts/u6KXXmKFbXfWzoAXn/",
)

NANDA_MODULAR = Citation(
    paper="Nanda et al., Progress Measures for Grokking via Mechanistic Interpretability",
    venue="ICLR 2023",
    arxiv_id="2301.05217",
    location="Section 3 (modular arithmetic circuit, 1-layer transformer)",
)

KIM_ENTITY = Citation(
    paper="Kim & Schuster, Entity Tracking in Language Models",
    venue="ACL 2023",
    arxiv_id="2305.02363",
    location="Section 3 (entity state tracking probes)",
)

LINZEN_SVA = Citation(
    paper="Linzen et al., Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies",
    venue="TACL 2016",
    arxiv_id="1611.01368",
    location="Section 2 (SVA with PP attractors)",
)

WILCOX_FILLER = Citation(
    paper="Wilcox et al., Using Computational Models to Test Syntactic Learnability",
    venue="Linguistic Inquiry 2022",
    arxiv_id="1811.01329",
    location="Section 3 (filler-gap dependencies in neural LMs)",
)

WARSTADT_BLIMP = Citation(
    paper="Warstadt et al., BLiMP: The Benchmark of Linguistic Minimal Pairs",
    venue="TACL 2020",
    arxiv_id="1912.00582v4",
    location="Table 3 (category-level); GitHub supplementary for per-paradigm",
)

MILLER_A_AN = Citation(
    paper="Miller & Neo, A Mechanistic Understanding of the A/An Prediction",
    venue="arXiv 2023",
    arxiv_id="2309.04105",
    location="Section 3 (single neuron L1N2631 in GPT-2 Large)",
)

# ---------------------------------------------------------------------------
# Sentinel for tasks discovered in this project
# ---------------------------------------------------------------------------

THIS_WORK = Citation(
    paper="Tower 2026, Weight-Space Circuit Discovery in GPT-2",
    venue="unpublished",
    arxiv_id="",
    location="MIB/weight_circuit/experiments/v2_second_investigation/",
)
