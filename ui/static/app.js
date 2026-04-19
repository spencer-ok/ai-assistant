const statusEl = document.getElementById('status');
const speechEl = document.getElementById('speech');
const transcriptEl = document.getElementById('transcript');
const textbox = document.getElementById('textbox');
let lastLen = 0;

const STATUS_LABELS = { idle: 'Ready', listening: 'Listening', thinking: 'Processing', speaking: 'Speaking' };
const STATUS_EMOJI = { idle: '\u{1F60A}', listening: '\u{1F3A4}', thinking: '\u{1F4AD}', speaking: '\u{1F50A}' };

function poll() {
  fetch('/state').then(r => r.json()).then(s => {
    document.body.className = s.status;
    statusEl.textContent = STATUS_LABELS[s.status] || s.status;
    const emojiEl = document.getElementById('state-emoji');
    if (emojiEl) emojiEl.textContent = STATUS_EMOJI[s.status] || '';
    speechEl.textContent = s.current_speech;
    const mb = document.getElementById('mutebtn');
    const mode = s.listen_mode || 'always';
    mb.classList.remove('muted', 'wake');
    if (mode === 'muted') { mb.classList.add('muted'); mb.innerHTML = '&#x1F507;'; mb.title = 'Muted — click for Always On'; }
    else if (mode === 'name') { mb.classList.add('wake'); mb.innerHTML = '&#x1F4AC;'; mb.title = 'Name Activated — click for Muted'; }
    else { mb.innerHTML = '&#x1F3A4;'; mb.title = 'Always Listening — click for Name Activated'; }
    if (s.transcript.length !== lastLen) {
      transcriptEl.innerHTML = s.transcript.map(m =>
        `<div class="msg ${m.role}"><span class="label">${m.role === 'user' ? 'You' : 'Rosie'}:</span> ${m.text}</div>`
      ).join('');
      transcriptEl.scrollTop = transcriptEl.scrollHeight;
      lastLen = s.transcript.length;
    }
  }).catch(() => {});
}
setInterval(poll, 200);

function updateClock() {
  const now = new Date();
  const h = now.getHours() % 12 || 12;
  const m = String(now.getMinutes()).padStart(2, '0');
  const ampm = now.getHours() >= 12 ? 'PM' : 'AM';
  document.getElementById('clock').textContent = `${h}:${m} ${ampm}`;
  const days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
  const months = ['January','February','March','April','May','June','July','August','September','October','November','December'];
  document.getElementById('clock-date').textContent = `${days[now.getDay()]}, ${months[now.getMonth()]} ${now.getDate()}, ${now.getFullYear()}`;
}
updateClock();
setInterval(updateClock, 10000);

document.getElementById('sendbtn').onclick = () => {
  const text = textbox.value.trim();
  if (!text) return;
  fetch('/send', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text}) });
  textbox.value = '';
};
textbox.addEventListener('keydown', e => { if (e.key === 'Enter') document.getElementById('sendbtn').click(); });
document.getElementById('stopbtn').onclick = () => { fetch('/stop', { method: 'POST' }); };
document.getElementById('mutebtn').onclick = () => { fetch('/mute', { method: 'POST' }); };
document.getElementById('exitbtn').onclick = () => { fetch('/shutdown', { method: 'POST' }); };

// Caregiver panel
const cgPanel = document.getElementById('caregiver');
document.getElementById('cg-toggle').onclick = () => { cgPanel.classList.toggle('open'); if (cgPanel.classList.contains('open')) loadCaregiverMessages(); };

function loadCaregiverMessages() {
  fetch('/caregiver').then(r => r.json()).then(d => {
    document.getElementById('cg-messages').innerHTML = d.messages.map(m =>
      `<div class="cg-msg"><span class="cg-from">${m.from}:</span> ${m.text} <span class="cg-time">${m.time}</span></div>`
    ).join('') || '<div style="color:#555">No notes yet</div>';
  });
}
loadCaregiverMessages();
setInterval(loadCaregiverMessages, 5000);

fetch('/caregivers').then(r => r.json()).then(d => {
  const select = document.getElementById('cg-from-select');
  const input = document.getElementById('cg-from');
  if (d.caregivers && d.caregivers.length > 0) {
    d.caregivers.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.name;
      opt.textContent = `${c.name} (${c.relationship})`;
      select.appendChild(opt);
    });
    select.style.display = 'block';
    input.style.display = 'none';
  }
});

document.getElementById('cg-sendbtn').onclick = sendNote;
document.getElementById('cg-text').addEventListener('keydown', e => { if (e.key === 'Enter') sendNote(); });

function sendNote() {
  const text = document.getElementById('cg-text').value.trim();
  const select = document.getElementById('cg-from-select');
  const input = document.getElementById('cg-from');
  const from = select.style.display !== 'none' ? select.value : input.value.trim() || 'Caregiver';
  if (!text) return;
  fetch('/caregiver', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text, from}) })
    .then(() => { document.getElementById('cg-text').value = ''; cgPanel.classList.remove('open'); })
    .catch(e => console.error('Note send failed:', e));
};

const recordBtn = document.getElementById('cg-recordbtn');
recordBtn.onclick = () => {
  const from = document.getElementById('cg-from').value.trim() || 'Caregiver';
  fetch('/mute', { method: 'POST' }).then(() => {
    recordBtn.textContent = 'Listening...';
    recordBtn.style.background = '#e74c3c'; recordBtn.style.color = '#fff';
    setTimeout(() => {
      fetch('/caregiver/record', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({from}) })
        .then(r => r.json())
        .then(d => {
          recordBtn.textContent = 'Record Note';
          recordBtn.style.background = ''; recordBtn.style.color = '';
          fetch('/mute', { method: 'POST' });
          if (d.text) { loadCaregiverMessages(); cgPanel.classList.remove('open'); }
          else alert('Nothing heard - try again');
        })
        .catch(() => { recordBtn.textContent = 'Record Note'; recordBtn.style.background = ''; recordBtn.style.color = ''; fetch('/mute', { method: 'POST' }); });
    }, 1000);
  });
};
