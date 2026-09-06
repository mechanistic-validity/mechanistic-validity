import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightClientMermaid from '@pasqal-io/starlight-client-mermaid';
import { execSync } from 'child_process';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://mechanistic-validity.github.io',
  base: '/mechanistic-validity',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    {
      name: 'build-backlinks',
      hooks: {
        'astro:build:start': () => execSync('node scripts/build-backlinks.mjs'),
        'astro:server:start': () => execSync('node scripts/build-backlinks.mjs'),
      },
    },
    starlight({
      title: 'Mechanistic Validity',
      description: 'A framework for evaluating mechanistic claims in neural networks',
      plugins: [starlightClientMermaid()],
      customCss: ['./src/styles/custom.css'],
      expressiveCode: {
        themes: ['github-dark', 'github-light'],
        styleOverrides: {
          borderRadius: '0.375rem',
        },
      },
      components: {
        Pagination: './src/components/PageFooter.astro',
        PageTitle: './src/components/PageTitle.astro',
      },
      sidebar: [
        { label: 'Home', link: '/' },
        {
          label: 'Mechanistic Validity Framework',
          collapsed: false,
          items: [
            { label: 'Framework Overview', link: '/framework/' },
            {
              label: 'Theoretical Foundations',
              collapsed: true,
              items: [
                { label: 'Philosophy of Science', link: '/framework/lenses/core/philosophy-of-science' },
                { label: 'Psychometrics', link: '/framework/lenses/core/measurement-theory' },
                { label: 'Neuroscience', link: '/framework/lenses/core/neuroscience' },
                { label: 'Pharmacology', link: '/framework/lenses/core/pharmacology' },
                { label: 'Genetics', link: '/framework/lenses/supporting/genetics' },
                { label: 'Mechanistic Interpretability', link: '/framework/lenses/core/mechanistic-interpretability' },
              ],
            },
            {
              label: '1. Description Modes',
              collapsed: true,
              items: [
                { label: 'Overview', link: '/framework/description-modes/' },
                { label: 'Computational', link: '/framework/modes/computational' },
                { label: 'Algorithmic', link: '/framework/modes/algorithmic' },
                { label: 'Representational', link: '/framework/modes/representational' },
                { label: 'Implementational (functional)', link: '/framework/modes/implementational-functional' },
                { label: 'Implementational (connectomic)', link: '/framework/modes/implementational-connectomic' },
                { label: 'Implementational (statistical)', link: '/framework/modes/implementational-activation' },
                { label: 'Implementational (topographic)', link: '/framework/modes/implementational-topographic' },
              ],
            },
            {
              label: '2. Evidence Families',
              collapsed: true,
              items: [
                { label: 'Overview', link: '/framework/evidence-families/' },
                { label: 'Weights', link: '/framework/evidence-families/weights' },
                { label: 'Activations', link: '/framework/evidence-families/activations' },
                { label: 'Behavior', link: '/framework/evidence-families/behavior' },
                { label: 'Training', link: '/framework/evidence-families/training' },
              ],
            },
            { label: '3. Metrics', link: '/framework/metrics/' },
            {
              label: '4. Criteria',
              collapsed: true,
              items: [
                { label: 'Overview', link: '/framework/criteria/' },
                {
                  label: 'Construct (C1–C6)',
                  collapsed: true,
                  items: [
                    { label: 'C1 Falsifiability', link: '/framework/criteria/construct/falsifiability' },
                    { label: 'C2 Structural plausibility', link: '/framework/criteria/construct/structural-plausibility' },
                    { label: 'C3 Convergent validity', link: '/framework/criteria/construct/convergent-validity' },
                    { label: 'C4 Discriminant validity', link: '/framework/criteria/construct/discriminant-validity' },
                    { label: 'C5 Nomological validity', link: '/framework/criteria/construct/nomological-validity' },
                    { label: 'C6 Complementation validity', link: '/framework/criteria/construct/complementation-validity' },
                  ],
                },
                {
                  label: 'Measurement (M1–M7)',
                  collapsed: true,
                  items: [
                    { label: 'M1 Reliability', link: '/framework/criteria/measurement/reliability' },
                    { label: 'M2 Baseline separation', link: '/framework/criteria/measurement/baseline-separation' },
                    { label: 'M3 Stability', link: '/framework/criteria/measurement/stability' },
                    { label: 'M4 Calibration', link: '/framework/criteria/measurement/calibration' },
                    { label: 'M5 Sensitivity', link: '/framework/criteria/measurement/sensitivity' },
                    { label: 'M6 Invariance', link: '/framework/criteria/measurement/invariance' },
                    { label: 'M7 Selection correction', link: '/framework/criteria/measurement/selection-correction' },
                  ],
                },
                {
                  label: 'Internal (I1–I12)',
                  collapsed: true,
                  items: [
                    { label: 'I1 Necessity', link: '/framework/criteria/internal/necessity' },
                    { label: 'I2 Sufficiency', link: '/framework/criteria/internal/sufficiency' },
                    { label: 'I3 Minimality', link: '/framework/criteria/internal/minimality' },
                    { label: 'I4 Specificity', link: '/framework/criteria/internal/specificity' },
                    { label: 'I5 Rival mechanism exclusion', link: '/framework/criteria/internal/rival-mechanism-exclusion' },
                    { label: 'I6 Double dissociation', link: '/framework/criteria/internal/double-dissociation' },
                    { label: 'I7 Confound control', link: '/framework/criteria/internal/confound-control' },
                    { label: 'I8 Confounding sensitivity', link: '/framework/criteria/internal/confounding-sensitivity' },
                    { label: 'I9 Epistatic interaction', link: '/framework/criteria/internal/epistatic-interaction' },
                    { label: 'I10 Rescue reversibility', link: '/framework/criteria/internal/rescue-reversibility' },
                    { label: 'I11 Onset coupling', link: '/framework/criteria/internal/onset-coupling' },
                    { label: 'I12 Offset coupling', link: '/framework/criteria/internal/offset-coupling' },
                  ],
                },
                {
                  label: 'External (E1–E6)',
                  collapsed: true,
                  items: [
                    { label: 'E1 Intervention reach', link: '/framework/criteria/external/intervention-reach' },
                    { label: 'E2 Prompt generalization', link: '/framework/criteria/external/prompt-generalization' },
                    { label: 'E3 Cross-task generalization', link: '/framework/criteria/external/cross-task-generalization' },
                    { label: 'E4 Cross-model recurrence', link: '/framework/criteria/external/cross-model-recurrence' },
                    { label: 'E5 Graded response', link: '/framework/criteria/external/graded-response' },
                    { label: 'E6 Novel prediction', link: '/framework/criteria/external/novel-prediction' },
                  ],
                },
                {
                  label: 'Interpretive (V1–V5)',
                  collapsed: true,
                  items: [
                    { label: 'V1 Level declaration', link: '/framework/criteria/interpretive/level-declaration' },
                    { label: 'V2 Level-evidence match', link: '/framework/criteria/interpretive/level-evidence-match' },
                    { label: 'V3 Alternative level', link: '/framework/criteria/interpretive/alternative-level' },
                    { label: 'V4 Unlicensed labeling', link: '/framework/criteria/interpretive/unlicensed-labeling' },
                    { label: 'V5 Scope declaration', link: '/framework/criteria/interpretive/scope-declaration' },
                  ],
                },
              ],
            },
            {
              label: '5. Validity Types',
              collapsed: true,
              items: [
                { label: 'Overview', link: '/framework/validity-types/' },
                { label: 'Construct Validity', link: '/framework/validity-types/construct' },
                { label: 'Measurement Validity', link: '/framework/validity-types/measurement' },
                { label: 'Internal Validity', link: '/framework/validity-types/internal' },
                { label: 'External Validity', link: '/framework/validity-types/external' },
                { label: 'Interpretive Validity', link: '/framework/validity-types/interpretive' },
              ],
            },
            {
              label: '6. Verdicts',
              collapsed: true,
              items: [
                { label: 'Overview', link: '/framework/verdicts/' },
                { label: 'Tier 1: Proposed', link: '/framework/verdicts/proposed' },
                { label: 'Tier 2: Causally Suggestive', link: '/framework/verdicts/causally-suggestive' },
                { label: 'Tier 3: Mechanistically Supported', link: '/framework/verdicts/mechanistically-supported' },
                { label: 'Tier 4: Triangulated', link: '/framework/verdicts/triangulated' },
                { label: 'Tier 5: Validated', link: '/framework/verdicts/validated' },
                { label: 'Underdetermined', link: '/framework/verdicts/underdetermined' },
                { label: 'Insufficient', link: '/framework/verdicts/insufficient' },
                { label: 'Disconfirmed', link: '/framework/verdicts/disconfirmed' },
              ],
            },
          ],
        },
        {
          label: 'Case Studies',
          collapsed: false,
          items: [
            { label: 'Overview', link: '/framework/examples/' },
            { label: 'Induction Heads', link: '/framework/examples/examples-induction-heads' },
            { label: 'IOI Circuit', link: '/framework/examples/examples-ioi' },
            { label: 'Greater-Than', link: '/framework/examples/examples-greater-than' },
            { label: 'Copy Suppression', link: '/framework/examples/examples-copy-suppression' },
            { label: 'Successor Heads', link: '/framework/examples/examples-successor-heads' },
            { label: 'Modular Addition', link: '/framework/examples/examples-grokking' },
            { label: 'Refusal Direction', link: '/framework/examples/examples-refusal-direction' },
            { label: 'Superposition', link: '/framework/examples/examples-superposition' },
            { label: 'Global Workspace', link: '/framework/examples/examples-global-workspace' },
            { label: 'Docstring Circuit', link: '/framework/examples/examples-docstring' },
            { label: 'Gender Bias Circuits', link: '/framework/examples/examples-gender-bias' },
            { label: 'Othello World Model', link: '/framework/examples/examples-othello' },
            { label: 'SAE Features', link: '/framework/examples/examples-sae-features' },
            { label: 'Probing Classifiers', link: '/framework/examples/examples-probing' },
            { label: 'Induction Heads (General ICL)', link: '/framework/examples/examples-induction-heads-icl' },
            { label: 'Knowledge Neurons', link: '/framework/examples/examples-knowledge-neurons' },
          ],
        },
      ],
    }),
  ],
});
