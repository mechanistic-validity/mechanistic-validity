# Email draft — David Jensen (send yourself; schedule ~12pm tomorrow)

To: jensen@cs.umass.edu   ← verify address
Subject: Draft framework on validity for mechanistic-interpretability claims — would value your read

---

Hi David,

Apologies for the slow follow-up — I finally have something concrete to show you, and it's squarely in your wheelhouse.

I've been developing a framework I'm calling Mechanistic Validity: it ports the validity-theory tradition (construct → measurement → internal → external → interpretive, in dependency order) onto *causal mechanistic claims* in interpretability — the "this circuit implements this computation" kind of claim — rather than onto benchmark scores. The core argument is that a claim caps at the tier its weakest validity dimension supports. I use it to audit sixteen published interpretability claims (IOI, induction heads, SAE features, steering vectors, and Anthropic's recent "global workspace" result), showing the framework discriminates without new experiments.

Your work on evaluation methodology and causal models of systems is exactly the lens I keep reaching for, so I'd really value your read before I commit to the validation experiments. Three questions where your input would actually change what I do:

1. **Severity.** I want preregistration and independent corroboration to gate the top tiers *without* a crude "no-prereg, no-Triangulated" rule. I've grounded it in Mayo's severe testing plus the evidence-family structure — does that hold up, or is there a cleaner formulation?
2. **Self-validation.** I plan to calibrate the framework against a ground-truth toy circuit (grokking modular addition, using the Clock-vs-Pizza dual mechanism so I can test whether the tiers *discriminate* between two known mechanisms). Is that the right validation target, or is there a better one?
3. Whether the paper's prescriptive turn — proposing preregistration and corroboration as standards for the field — overreaches.

Draft attached: the structure is complete and the argument is all there; prose is still filling in, and anything unwritten is clearly marked in red so you can see the skeleton for what it is. Even quick reactions to those three questions would help a lot, and I'd be glad to talk if that's easier.

Thanks,
Elliot
