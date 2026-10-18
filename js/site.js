// James C. Mott · shared site script.
// Each block checks for its own elements, so one file serves every page.

// ── Expand / collapse (CV milestones, FAQ) ──
document.querySelectorAll('[data-toggle]').forEach(btn => {
  const panel = document.getElementById(btn.getAttribute('aria-controls'));
  if (!panel) return;
  btn.addEventListener('click', () => {
    const open = btn.getAttribute('aria-expanded') === 'true';
    btn.setAttribute('aria-expanded', String(!open));
    panel.style.maxHeight = open ? '0px' : panel.scrollHeight + 'px';
    const label = btn.querySelector('.label');
    if (label && btn.dataset.openLabel) {
      label.textContent = open ? btn.dataset.closedLabel : btn.dataset.openLabel;
    }
  });
});

// ── Talks: slide viewer ──
// Adding a new talk: append one object to TALKS.
//   slideDir: folder containing slide-01.jpg, slide-02.jpg, ...
//   slideCount: number of slides in that folder
const TALKS = [
  {
    id: 'knox-2025',
    slideDir: 'talks/knox-2025/slides',
    slideCount: 23
  }
];

function padSlide(n, total) {
  const width = total >= 100 ? 3 : total >= 10 ? 2 : 1;
  return String(n).padStart(width, '0');
}

TALKS.forEach(talk => {
  const viewer = document.querySelector(`[data-talk="${talk.id}"]`);
  if (!viewer) return;
  const stage = viewer.querySelector('.stage');
  const counter = viewer.querySelector('.viewer-current');
  const prev = viewer.querySelector('[data-dir="-1"]');
  const next = viewer.querySelector('[data-dir="1"]');

  // Only slide 1 loads eagerly; neighbours load as you navigate.
  const imgs = Array.from({ length: talk.slideCount }, (_, i) => {
    const img = document.createElement('img');
    const src = `${talk.slideDir}/slide-${padSlide(i + 1, talk.slideCount)}.jpg`;
    if (i === 0) img.src = src; else img.dataset.src = src;
    img.alt = `Slide ${i + 1} of ${talk.slideCount}`;
    stage.appendChild(img);
    return img;
  });

  function load(i) {
    const img = imgs[i];
    if (img && img.dataset.src) { img.src = img.dataset.src; delete img.dataset.src; }
  }

  let idx = 0;
  function show(i) {
    idx = Math.max(0, Math.min(talk.slideCount - 1, i));
    load(idx); load(idx - 1); load(idx + 1);
    imgs.forEach((img, k) => img.classList.toggle('active', k === idx));
    counter.textContent = idx + 1;
    prev.disabled = idx === 0;
    next.disabled = idx === talk.slideCount - 1;
  }

  prev.addEventListener('click', () => show(idx - 1));
  next.addEventListener('click', () => show(idx + 1));
  viewer.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft') { e.preventDefault(); show(idx - 1); }
    if (e.key === 'ArrowRight') { e.preventDefault(); show(idx + 1); }
  });
  show(0);
});

// ── Ask James chat ──
// The system prompt lives on the Cloudflare Worker, not in this file.
(function initChat() {
  const log = document.getElementById('chatLog');
  const input = document.getElementById('chatInput');
  const send = document.getElementById('chatSend');
  if (!log || !input || !send) return;

  const CHAT_PROXY_URL = 'https://ask-james.jimmott52.workers.dev';
  const messages = [];
  let busy = false;

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
  }

  function addMessage(who, html) {
    const row = document.createElement('div');
    row.className = 'msg ' + who;
    const av = document.createElement('div');
    av.className = 'msg-av';
    av.textContent = who === 'james' ? 'J' : 'You';
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML = html;
    row.append(av, bubble);
    log.appendChild(row);
    log.scrollTop = log.scrollHeight;
    return bubble;
  }

  async function sendMessage() {
    const text = input.value.trim();
    if (!text || busy) return;
    input.value = '';
    input.style.height = 'auto';
    busy = true;
    send.disabled = true;

    addMessage('user', escapeHtml(text));
    messages.push({ role: 'user', content: text });
    const bubble = addMessage('james', 'Thinking ·');
    bubble.classList.add('pending');

    try {
      const res = await fetch(CHAT_PROXY_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages })
      });
      bubble.classList.remove('pending');
      if (!res.ok) {
        const err = await res.text();
        bubble.innerHTML = escapeHtml('Something went wrong (' + res.status + '). ' + err.slice(0, 160));
      } else {
        const data = await res.json();
        const reply = data.content && data.content[0] && data.content[0].text ? data.content[0].text : '';
        bubble.innerHTML = reply ? escapeHtml(reply) : 'No response received. Please try again.';
        if (reply) messages.push({ role: 'assistant', content: reply });
      }
    } catch (err) {
      bubble.classList.remove('pending');
      bubble.innerHTML = escapeHtml('Could not reach the assistant. ' + err.message);
    }
    log.scrollTop = log.scrollHeight;
    busy = false;
    send.disabled = false;
    input.focus();
  }

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  });
  send.addEventListener('click', sendMessage);
  document.querySelectorAll('[data-prompt]').forEach(b => {
    b.addEventListener('click', () => { input.value = b.dataset.prompt; sendMessage(); });
  });
})();
