const chat = document.querySelector('#chat');
const form = document.querySelector('#form');
const input = document.querySelector('#message');

function add(text, cls, sources = []) {
  const bubble = document.createElement('div');
  bubble.className = cls;
  const body = document.createElement('div');
  body.textContent = text;
  bubble.append(body);
  if (sources.length) {
    const evidence = document.createElement('div');
    evidence.className = 'sources';
    evidence.textContent = `Sources: ${sources.join(', ')}`;
    bubble.append(evidence);
  }
  chat.append(bubble);
  chat.scrollTop = chat.scrollHeight;
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  add(message, 'user');
  input.value = '';
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  });
  const result = await response.json();
  add(result.reply, 'bot', result.sources);
});

document.querySelector('#reset').onclick = async () => {
  await fetch('/api/reset', { method: 'POST' });
  chat.innerHTML = '';
  add("Hello! I'm ZentroMart's AI assistant. How can I help?", 'bot');
};

async function sendAudio(blob, filename = 'voice-message.wav') {
  const audioUrl = URL.createObjectURL(blob);
  const metadataAudio = new Audio();
  const duration = await new Promise(resolve => {
    metadataAudio.onloadedmetadata = () => resolve(metadataAudio.duration);
    metadataAudio.onerror = () => resolve(null);
    metadataAudio.src = audioUrl;
  });
  URL.revokeObjectURL(audioUrl);
  if (duration !== null && duration > 60) {
    add('Audio must be no longer than 60 seconds. Please trim the recording and try again.', 'bot');
    return;
  }
  const formData = new FormData();
  formData.append('audio', blob, filename);
  const language = document.querySelector('#stt-language').value;
  const replyLanguage = document.querySelector('#tts-language').value;
  const voice = document.querySelector('#tts-voice').value;
  const response = await fetch(`/api/voice?language=${encodeURIComponent(language)}&reply_language=${encodeURIComponent(replyLanguage)}&voice=${encodeURIComponent(voice)}`, { method: 'POST', body: formData });
  const result = await response.json();
  if (!response.ok) {
    add(result.detail || 'Voice request failed.', 'bot');
    return;
  }
  add(`You said: ${result.transcript}`, 'user');
  add(result.reply, 'bot', result.sources);
  new Audio(result.audio_url).play().catch(() => {});
}

document.querySelector('#audio-file').onchange = event => {
  if (event.target.files[0]) sendAudio(event.target.files[0], event.target.files[0].name);
};

let recorder;
let chunks = [];
let recordingTimer;
document.querySelector('#record').onclick = async event => {
  if (recorder?.state === 'recording') {
    recorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = chunk => chunks.push(chunk.data);
    recorder.onstop = async () => {
      clearTimeout(recordingTimer);
      stream.getTracks().forEach(track => track.stop());
      const recording = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
      await sendAudio(recording, 'voice-message.webm');
      event.target.textContent = 'Start voice message';
    };
    recorder.start();
    event.target.textContent = 'Stop and send';
    recordingTimer = setTimeout(() => {
      if (recorder.state === 'recording') {
        add('Recording stopped at the 60-second limit.', 'bot');
        recorder.stop();
      }
    }, 60000);
  } catch {
    add('Microphone access was not available. You can upload an audio file instead.', 'bot');
  }
};
