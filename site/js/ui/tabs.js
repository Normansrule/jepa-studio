// Accessible tabs (WAI-ARIA tabs pattern): arrow keys / Home / End move focus and select,
// the URL hash (#train, #inspect ...) remembers the tab, and onChange fires on every switch.
export function setupTabs(tablist, { onChange = () => {} } = {}) {
  const tabs = [...tablist.querySelectorAll('[role="tab"]')];
  const panels = tabs.map((t) => document.getElementById(t.getAttribute('aria-controls')));

  function select(i, { focus = false, updateHash = true } = {}) {
    tabs.forEach((t, j) => {
      const on = i === j;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      panels[j].hidden = !on;
    });
    if (focus) tabs[i].focus();
    if (updateHash) history.replaceState(null, '', `#${tabs[i].dataset.tab}`);
    onChange(tabs[i].dataset.tab);
  }

  tabs.forEach((t, i) => {
    t.addEventListener('click', () => select(i));
    t.addEventListener('keydown', (e) => {
      let j = null;
      if (e.key === 'ArrowRight') j = (i + 1) % tabs.length;
      else if (e.key === 'ArrowLeft') j = (i - 1 + tabs.length) % tabs.length;
      else if (e.key === 'Home') j = 0;
      else if (e.key === 'End') j = tabs.length - 1;
      if (j !== null) { e.preventDefault(); select(j, { focus: true }); }
    });
  });

  const fromHash = () => {
    const name = location.hash.slice(1);
    const i = tabs.findIndex((t) => t.dataset.tab === name);
    return i >= 0 ? i : 0;
  };
  window.addEventListener('hashchange', () => select(fromHash(), { updateHash: false }));
  select(fromHash(), { updateHash: false });
  return {
    select: (name) => { const i = tabs.findIndex((t) => t.dataset.tab === name); if (i >= 0) select(i); },
    current: () => tabs.find((t) => t.getAttribute('aria-selected') === 'true').dataset.tab,
  };
}
