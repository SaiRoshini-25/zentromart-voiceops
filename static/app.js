const chat = document.querySelector('#chat');
const form = document.querySelector('#form');
const input = document.querySelector('#message');
const sendButton = form.querySelector('button[type="submit"]');
const recordButton = document.querySelector('#record');
const audioFile = document.querySelector('#audio-file');
const liveStatus = document.querySelector('#live-status');
const welcomeMessage = "Hello! I'm ZentroMart's AI assistant. How can I help?";
let activeSpeechAudio = null;
const quickActionsMarkup = `
  <div class="quick-actions" aria-label="Suggested questions">
    <span class="suggestion-label">POPULAR TOPICS</span>
    <button type="button" data-message="Where is my order?"><span aria-hidden="true">↗</span> Track an order</button>
    <button type="button" data-message="What is your return policy?"><span aria-hidden="true">↩</span> Returns</button>
    <button type="button" data-message="How do refunds work?"><span aria-hidden="true">＄</span> Refunds</button>
  </div>`;

function add(text, role, sources = [], audioUrl = null) {
  stopSpeechPlayback();
  const message = document.createElement('div');
  message.className = `message ${role}`;

  if (role === 'bot') {
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.setAttribute('aria-hidden', 'true');
    avatar.textContent = 'Z';
    message.append(avatar);
  }

  const content = document.createElement('div');
  content.className = 'message-content';
  const label = document.createElement('span');
  label.className = 'message-label';
  label.textContent = role === 'user' ? 'You' : 'ZentroMart assistant';
  const time = document.createElement('time');
  time.textContent = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(new Date());
  label.append(time);

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.textContent = text;
  content.append(label, bubble);

  if (role === 'bot' && sources.length) {
    const sourceList = document.createElement('div');
    sourceList.className = 'sources';
    sourceList.setAttribute('aria-label', 'Answer sources');
    for (const source of sources) {
      const chip = document.createElement('span');
      chip.className = 'source-chip';
      chip.textContent = source;
      sourceList.append(chip);
    }
    content.append(sourceList);
  }

  if (audioUrl) {
    const audio = document.createElement('audio');
    audio.className = 'message-audio';
    audio.controls = true;
    audio.preload = 'auto';
    audio.autoplay = true;
    audio.src = audioUrl;
    audio.setAttribute('aria-label', 'Play spoken reply');
    audio.addEventListener('play', () => {
      if (activeSpeechAudio && activeSpeechAudio !== audio) {
        activeSpeechAudio.pause();
        activeSpeechAudio.currentTime = 0;
      }
      activeSpeechAudio = audio;
    });
    audio.addEventListener('ended', () => {
      if (activeSpeechAudio === audio) activeSpeechAudio = null;
    });
    content.append(audio);
    activeSpeechAudio = audio;
  }

  message.append(content);
  chat.append(message);
  chat.scrollTop = chat.scrollHeight;
  if (audioUrl) {
    const audio = message.querySelector('audio');
    audio.play().catch(() => {
      const note = document.createElement('p');
      note.className = 'audio-playback-note';
      note.textContent = 'Your browser blocked automatic playback. Use the play control to hear this reply.';
      content.append(note);
    });
  }
  return message;
}

function stopSpeechPlayback() {
  chat.querySelectorAll('audio.message-audio').forEach(audio => {
    audio.pause();
    audio.currentTime = 0;
  });
  activeSpeechAudio = null;
}

function setBusy(busy) {
  sendButton.disabled = busy;
  input.disabled = busy;
  liveStatus.textContent = busy ? 'ZentroMart assistant is responding.' : '';
}

async function readResponse(response) {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

async function sendMessage(message) {
  const cleanMessage = message.trim();
  if (!cleanMessage || sendButton.disabled) return;

  add(cleanMessage, 'user');
  input.value = '';
  setBusy(true);
  const typing = add('Thinking', 'bot');
  typing.classList.add('typing');
  typing.querySelector('.message-bubble').innerHTML = 'Thinking<span class="typing-dots" aria-hidden="true">•••</span>';

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: cleanMessage })
    });
    const result = await readResponse(response);
    typing.remove();
    if (!response.ok) {
      add(result.detail || 'I could not send your message. Please try again.', 'bot');
      return;
    }
    add(result.reply || 'I could not prepare a reply. Please try again.', 'bot', result.sources || []);
  } catch {
    typing.remove();
    add('I could not connect just now. Please check your connection and try again.', 'bot');
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener('submit', event => {
  event.preventDefault();
  sendMessage(input.value);
});

document.addEventListener('click', event => {
  const button = event.target.closest('[data-message]');
  if (button) sendMessage(button.dataset.message || '');
});

document.querySelector('#reset').addEventListener('click', async () => {
  const resetButton = document.querySelector('#reset');
  resetButton.disabled = true;
  stopSpeechPlayback();
  try {
    const response = await fetch('/api/reset', { method: 'POST' });
    const result = await readResponse(response);
    if (!response.ok) {
      add(result.detail || 'I could not start a new conversation. Please try again.', 'bot');
      return;
    }
    chat.innerHTML = `
      <div class="message bot">
        <div class="message-avatar" aria-hidden="true">Z</div>
        <div class="message-content">
          <span class="message-label">ZentroMart assistant <time>now</time></span>
          <div class="message-bubble">${welcomeMessage}</div>
        </div>
      </div>${quickActionsMarkup}`;
    chat.scrollTop = 0;
  } catch {
    add('I could not connect just now. Please try starting a new conversation again.', 'bot');
  } finally {
    resetButton.disabled = false;
  }
});

async function sendAudio(blob, filename = 'voice-message.wav') {
  stopSpeechPlayback();
  if (!blob.size) {
    add('I did not receive any audio. Please record again or choose another file.', 'bot');
    return;
  }
  if (blob.size > 20 * 1024 * 1024) {
    add('Audio files must be smaller than 20 MB. Please choose a shorter recording.', 'bot');
    return;
  }

  const previewUrl = URL.createObjectURL(blob);
  const metadataAudio = new Audio();
  const duration = await new Promise(resolve => {
    const timeout = setTimeout(() => resolve(null), 2500);
    metadataAudio.onloadedmetadata = () => {
      clearTimeout(timeout);
      resolve(metadataAudio.duration);
    };
    metadataAudio.onerror = () => {
      clearTimeout(timeout);
      resolve(null);
    };
    metadataAudio.src = previewUrl;
  });
  URL.revokeObjectURL(previewUrl);
  if (duration !== null && duration > 60) {
    add('Audio must be no longer than 60 seconds. Please trim the recording and try again.', 'bot');
    return;
  }

  const formData = new FormData();
  formData.append('audio', blob, filename);
  const params = new URLSearchParams({
    language: document.querySelector('#stt-language').value,
    reply_language: document.querySelector('#tts-language').value,
    voice: document.querySelector('#tts-voice').value
  });
  recordButton.disabled = true;
  liveStatus.textContent = 'Uploading your voice message.';

  try {
    const response = await fetch(`/api/voice?${params}`, { method: 'POST', body: formData });
    const result = await readResponse(response);
    if (!response.ok) {
      add(result.detail || 'Voice request failed. Please try again or upload another audio file.', 'bot');
      return;
    }
    if (result.transcript) add(`You said: ${result.transcript}`, 'user');
    add(result.reply || 'I could not prepare a reply. Please try again.', 'bot', result.sources || [], result.audio_url || null);
  } catch {
    add('I could not connect to voice support. Please try again or send a text message.', 'bot');
  } finally {
    recordButton.disabled = false;
    liveStatus.textContent = '';
  }
}

audioFile.addEventListener('change', async event => {
  const file = event.target.files?.[0];
  if (file) {
    stopSpeechPlayback();
    await sendAudio(file, file.name);
  }
  event.target.value = '';
});

let recorder;
let recordingStream;
let chunks = [];
let recordingTimer;

function setRecording(isRecording) {
  recordButton.classList.toggle('is-recording', isRecording);
  recordButton.setAttribute('aria-label', isRecording ? 'Stop and send voice message' : 'Start voice message');
  recordButton.innerHTML = isRecording
    ? '<span class="record-icon" aria-hidden="true">●</span><span>Stop &amp; send</span>'
    : '<span class="record-icon" aria-hidden="true">●</span><span>Voice message</span>';
  liveStatus.textContent = isRecording ? 'Recording. Select stop and send when finished.' : '';
}

recordButton.addEventListener('click', async () => {
  if (recorder?.state === 'recording') {
    recorder.stop();
    setRecording(false);
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    add('Voice recording is not supported by this browser. You can upload an audio file instead.', 'bot');
    return;
  }
  stopSpeechPlayback();
  try {
    recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']
      .find(type => MediaRecorder.isTypeSupported(type));
    recorder = mimeType ? new MediaRecorder(recordingStream, { mimeType }) : new MediaRecorder(recordingStream);
    chunks = [];
    recorder.ondataavailable = event => {
      if (event.data.size) chunks.push(event.data);
    };
    recorder.onerror = () => {
      clearTimeout(recordingTimer);
      recordingStream?.getTracks().forEach(track => track.stop());
      setRecording(false);
      add('Recording failed. Please try again or upload an audio file.', 'bot');
    };
    recorder.onstop = async () => {
      clearTimeout(recordingTimer);
      recordingStream?.getTracks().forEach(track => track.stop());
      recordingStream = null;
      const type = recorder.mimeType || 'audio/webm';
      const extension = type.includes('mp4') ? 'm4a' : 'webm';
      await sendAudio(new Blob(chunks, { type }), `voice-message.${extension}`);
    };
    recorder.start();
    setRecording(true);
    recordingTimer = setTimeout(() => {
      if (recorder?.state === 'recording') {
        add('Recording stopped at the 60-second limit.', 'bot');
        recorder.stop();
        setRecording(false);
      }
    }, 60000);
  } catch {
    recordingStream?.getTracks().forEach(track => track.stop());
    recordingStream = null;
    setRecording(false);
    add('Microphone access was not available. You can upload an audio file instead.', 'bot');
  }
});

window.addEventListener('beforeunload', () => {
  clearTimeout(recordingTimer);
  recordingStream?.getTracks().forEach(track => track.stop());
});
