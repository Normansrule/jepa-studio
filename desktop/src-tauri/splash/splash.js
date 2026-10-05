// Setup/splash window. Reads the shell's status once (backend_status) and then follows the
// "setup://progress" events. Text only goes through textContent.
'use strict';

(function () {
  const $ = (id) => document.getElementById(id);
  const T = window.__TAURI__;

  function render(s) {
    if (!s) return;
    $('headline').textContent = s.message || '';
    const bar = document.querySelector('.bar');
    if (typeof s.progress === 'number') {
      const pct = Math.max(0, Math.min(100, Math.round(s.progress * 100)));
      bar.classList.remove('indeterminate');
      $('fill').style.width = pct + '%';
      bar.setAttribute('aria-valuenow', String(pct));
    } else if (s.phase !== 'error') {
      bar.classList.add('indeterminate');
      bar.removeAttribute('aria-valuenow');
    }
    if (typeof s.line === 'string' && s.line) $('line').textContent = s.line;
    if (s.phase === 'error') {
      bar.classList.remove('indeterminate');
      bar.hidden = true;
      $('line').hidden = true;
      $('note').hidden = true;
      $('headline').hidden = true;
      $('error').hidden = false;
      $('error-text').textContent = s.error || s.message || 'unknown error';
      $('log-path').textContent = s.logDir || '';
    }
  }

  if (!T || !T.core || !T.event) {
    render({ phase: 'error', message: 'Setup failed', error: 'The desktop bridge is not available in this window.' });
    return;
  }
  T.event.listen('setup://progress', (e) => render(e.payload)).catch(() => {});
  T.core.invoke('backend_status').then(render).catch(() => {});
})();
