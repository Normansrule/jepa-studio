// Help disclosures: every control label marked data-help="key" gets an info button that opens
// a short explanation with a link to the matching docs section on GitHub.
// Pattern: disclosure button (aria-expanded + aria-controls); Esc closes and returns focus to
// the button; clicking elsewhere closes it; only one is open at a time. Text via textContent.
import { el } from './dom.js';

const REPO = 'https://github.com/Normansrule/jepa-studio/blob/main/docs/';

// Text written from docs/data.md, docs/equations.md, docs/hardware.md and docs/web-demo.md.
// Anchors are GitHub's heading slugs for those files.
export const HELP = {
  'shapes-n': {
    text: 'How many images of the built-in Shapes dataset to draw: ten classes (circle, ring, stripes…) with random colour, position, size, rotation and noise. Labels are hidden from pretraining and only used by Evaluate.',
    doc: 'data.md#synthetic-shapes', where: 'data.md, Synthetic Shapes',
  },
  'data-seed': {
    text: 'Seeds the Shapes generator and is saved as "seed" in the config. Image i with seed s is identical in the browser and in Python, so a run can be reproduced on desktop.',
    doc: 'data.md#synthetic-shapes', where: 'data.md, Synthetic Shapes',
  },
  'aug-jitter': {
    text: 'Colour jitter strength s. With probability 0.8, brightness and contrast are scaled by a random factor in [1 − s, 1 + s] and saturation by one in [1 − s/2, 1 + s/2].',
    doc: 'data.md#2-views-augmentations', where: 'data.md §2, Views',
  },
  'aug-gray': {
    text: 'Probability that a view is turned grayscale, so the encoder cannot rely on colour alone to make views agree.',
    doc: 'data.md#2-views-augmentations', where: 'data.md §2, Views',
  },
  'aug-flip': {
    text: 'Probability of a horizontal flip, drawn separately for every global and local view.',
    doc: 'data.md#2-views-augmentations', where: 'data.md §2, Views',
  },
  'aug-mask': {
    text: 'Fraction of each view covered with 4-pixel blocks filled with mid-grey (0.5). 0 turns masking off; for time series it zeroes a span of that fraction of the window.',
    doc: 'data.md#2-views-augmentations', where: 'data.md §2, Views',
  },
  'series-window': {
    text: 'Length of the windows views are cut from. A global view is a random 80–100% sub-window resampled to this length; a local view is a 30–60% sub-window resampled to half of it (at least 8).',
    doc: 'data.md#2-views-augmentations', where: 'data.md §2, Views',
  },
  'train-preset': {
    text: 'A preset sets steps, batch size and SIGReg directions together, and puts λ and the learning rate back to their defaults. Change any of them and the preset shows "Custom".',
    doc: 'hardware.md#2-automatic-decisions', where: 'hardware.md §2, Automatic decisions',
  },
  'train-steps': {
    text: 'Optimizer steps to run. The learning rate warms up linearly over the first 10% of them, then decays along a cosine to 0.001 × its peak.',
    doc: 'hardware.md#5-learning-rate-schedule', where: 'hardware.md §5, Learning-rate schedule',
  },
  'train-lambda': {
    text: 'Weight of SIGReg in L = (1 − λ) L_pred + λ L_SIGReg, between 0 and 1. Higher λ pushes the embedding toward an isotropic Gaussian faster, at the cost of the prediction term.',
    doc: 'equations.md#1-the-lejepa-objective', where: 'equations.md §1, The LeJEPA objective',
  },
  'train-lr': {
    text: 'Peak AdamW learning rate, reached after the warm-up and then cosine-decayed to 0.001 × this value by the last step.',
    doc: 'hardware.md#5-learning-rate-schedule', where: 'hardware.md §5, Learning-rate schedule',
  },
  'train-slices': {
    text: 'Each step SIGReg projects every view\'s batch onto this many fresh random unit directions and tests each 1-D projection against N(0, 1) with the Epps–Pulley statistic. More directions give a less noisy signal and cost more time.',
    doc: 'equations.md#2-sigreg-random-slices--the-eppspulley-test', where: 'equations.md §2, SIGReg',
  },
  'train-batch': {
    web: {
      text: 'Images per step. Auto times one step at batch 32, 64, 128 and 256 and keeps the largest that stays under 150 ms; 32 is the floor because SIGReg needs enough samples per direction.',
      doc: 'web-demo.md#batch-size-and-the-timing-probe', where: 'web-demo.md, Batch size and the timing probe',
    },
    desktop: {
      text: 'Images per step. Auto runs one real step at 16, 32, 64… and keeps the largest size that stays under hardware.max_memory_frac (85%) of memory, capped by the hardware mode.',
      doc: 'hardware.md#2-automatic-decisions', where: 'hardware.md §2, Automatic decisions',
    },
  },
  'train-gpu': {
    text: 'Runs the first layer\'s forward pass and weight gradient as WebGPU compute shaders, but only when a start-up check finds them correct and faster than JavaScript.',
    doc: 'web-demo.md#webgpu', where: 'web-demo.md, WebGPU',
  },
  'eval-frac': {
    text: '20% of the items are held out for scoring. This share of the rest has its labels used: it trains the linear probe and is the k-NN memory. Pretraining never sees labels.',
    doc: 'equations.md#4-evaluation-linear-probe-k-nn-effective-rank-collapse-detector', where: 'equations.md §4, Evaluation',
  },
  'world-env': {
    text: 'The small simulated task the world model was trained on. The planner sees only embeddings of rendered frames, never the true state.',
    doc: 'equations.md#5-planning-in-a-latent-world-model-cost-and-cem', where: 'equations.md §5, Planning',
  },
  'world-start': {
    text: 'Seeds the environment\'s starting state; the same seed gives the same start every time.',
    doc: 'equations.md#5-planning-in-a-latent-world-model-cost-and-cem', where: 'equations.md §5, Planning',
  },
  'world-goal': {
    text: 'Seeds the goal state. Its frame is encoded once, and the planner minimizes the distance between imagined embeddings and that goal embedding.',
    doc: 'equations.md#5-planning-in-a-latent-world-model-cost-and-cem', where: 'equations.md §5, Planning',
  },
  'world-steps': {
    text: 'Most actions the bot executes. It replans after every action (receding horizon) and stops early when the goal is reached.',
    doc: 'equations.md#5-planning-in-a-latent-world-model-cost-and-cem', where: 'equations.md §5, Planning',
  },
  'world-samples': {
    text: 'Action sequences the Cross-Entropy Method samples per iteration. The lowest-cost few (the elites) refit the Gaussian the next iteration samples from.',
    doc: 'equations.md#5-planning-in-a-latent-world-model-cost-and-cem', where: 'equations.md §5, Planning',
  },
};

let open = null;   // {btn, panel}

function close({ focus = false } = {}) {
  if (!open) return;
  open.btn.setAttribute('aria-expanded', 'false');
  open.panel.hidden = true;
  if (focus) open.btn.focus();
  open = null;
}

function infoIcon() {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('aria-hidden', 'true');
  const c = document.createElementNS(ns, 'circle');
  c.setAttribute('cx', '12'); c.setAttribute('cy', '12'); c.setAttribute('r', '9.5');
  c.setAttribute('fill', 'none'); c.setAttribute('stroke', 'currentColor'); c.setAttribute('stroke-width', '1.6');
  const p = document.createElementNS(ns, 'path');
  p.setAttribute('d', 'M12 10.5v6M12 7.4v.01');
  p.setAttribute('fill', 'none'); p.setAttribute('stroke', 'currentColor'); p.setAttribute('stroke-width', '2.2'); p.setAttribute('stroke-linecap', 'round');
  svg.append(c, p);
  return svg;
}

/** Add an info button + disclosure panel to every [data-help] label/legend under root. */
export function initHelp(root, tier = 'web') {
  for (const lab of root.querySelectorAll('[data-help]')) {
    const key = lab.dataset.help;
    let h = HELP[key];
    if (!h) continue;
    if (h.web || h.desktop) h = h[tier] || h.web;
    const name = lab.dataset.helpName || lab.textContent.trim();
    const pid = `help-${key}`;
    const btn = el('button', { class: 'info-btn', attrs: { type: 'button', 'aria-expanded': 'false', 'aria-controls': pid, 'aria-label': `About ${name}`, 'data-help-for': key } }, infoIcon());
    const panel = el('div', { class: 'help-pop', attrs: { id: pid, hidden: true } },
      el('p', { text: h.text }),
      el('a', { text: `Read more: ${h.where}`, attrs: { href: REPO + h.doc, target: '_blank', rel: 'noopener noreferrer' } }));
    if (lab.classList.contains('label-row')) {      // an existing row (e.g. a switch): after its label
      if (lab.firstElementChild) lab.firstElementChild.after(btn); else lab.append(btn);
      lab.after(panel);
    } else {
      const row = el('div', { class: 'label-row' });
      lab.replaceWith(row);
      row.append(lab, btn);
      row.after(panel);
    }
    btn.addEventListener('click', () => {
      if (open && open.btn === btn) { close(); return; }
      close();
      btn.setAttribute('aria-expanded', 'true');
      panel.hidden = false;
      open = { btn, panel };
    });
  }
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || !open) return;
    const inside = open.panel.contains(document.activeElement) || open.btn === document.activeElement;
    close({ focus: inside });
    if (inside) e.stopPropagation();
  });
  document.addEventListener('pointerdown', (e) => {
    if (open && !open.panel.contains(e.target) && !open.btn.contains(e.target)) close();
  });
}
