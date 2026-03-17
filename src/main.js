// Configuration
const CONFIG = {
    SAMPLE_RATE: 16000,
    BUFFER_SIZE: 4096,
    WS_PATH: '/ws/audio',
    // ESP WebSocket config (same host, different path)
    ESP_WS_PATH: '/wsesp',
    ESP_WS_PORT: 80,
};

// State Management
const state = {
    websocket: null,
    mediaStream: null,
    audioContext: null,
    scriptProcessor: null,
    isRecording: false,
    isProcessing: false,
    reconnectAttempts: 0,
    maxReconnectAttempts: 3,
    pingInterval: null,
    lastPingMs: 0,
    // Translation state
    translateEnabled: false,
    targetLang: 'vi',
    contextMax: 6,
    pendingTranslations: 0,
    // Source language (for Whisper language pinning)
    sourceLang: 'auto',
    // Track transcript order for fallback pairing when ID is missing
    pendingTranslationIds: [],
    // Mic source: false = web mic, true = ESP mic
    useEspMic: false,
    // ESP display mode: "original" = thô, "translated" = đã dịch
    espDisplayMode: 'original',
    // ESP WebSocket connection
    espWebSocket: null,
    isEspConnected: false,
    // For ESP: control audio streaming
    isEspStreaming: false,
};

// DOM Elements
const elements = {
    startBtn: document.getElementById('startBtn'),
    stopBtn: document.getElementById('stopBtn'),
    clearBtn: document.getElementById('clearBtn'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    pingDisplay: document.getElementById('pingDisplay'),
    partialText: document.getElementById('partialText'),
    finalTranscript: document.getElementById('finalTranscript'),
    notification: document.getElementById('notification'),
    visualizer: document.getElementById('visualizer'),
    transcriptScroll: document.getElementById('transcriptScroll'),
    // Source language element
    sourceLangSelect: document.getElementById('sourceLangSelect'),
    // Translation elements
    translateToggle: document.getElementById('translateToggle'),
    langSelect: document.getElementById('langSelect'),
    contextSlider: document.getElementById('contextSlider'),
    contextValue: document.getElementById('contextValue'),
    toggleLabel: document.getElementById('toggleLabel'),
    translationColumn: document.getElementById('translationColumn'),
    translatedTranscript: document.getElementById('translatedTranscript'),
    translationScroll: document.getElementById('translationScroll'),
    translationStatus: document.getElementById('translationStatus'),
    transcriptsContainer: document.getElementById('transcriptsContainer'),
    originalColumn: document.getElementById('originalColumn'),
    // Mic source toggle
    micSourceToggle: document.getElementById('micSourceToggle'),
    micSourceLabel: document.getElementById('micSourceLabel'),
    // ESP display mode
    espDisplayModeSelect: document.getElementById('espDisplayModeSelect'),
};

// ── Utilities ─────────────────────────────────────────────

function escapeHtml(str) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
    return str.replace(/[&<>"']/g, c => map[c]);
}

function isNearBottom(el, threshold = 80) {
    return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
}

function scrollToBottom(el) {
    el.scrollTop = el.scrollHeight;
}

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    initVisualizer();
    setupEventListeners();
    console.log('App initialized');
});

function setupEventListeners() {
    elements.startBtn.addEventListener('click', startRecording);
    elements.stopBtn.addEventListener('click', stopRecording);
    elements.clearBtn.addEventListener('click', clearTranscripts);

    // Source language control
    elements.sourceLangSelect.addEventListener('change', onSourceLangChange);
    // Translation controls
    elements.translateToggle.addEventListener('change', onTranslateToggle);
    elements.langSelect.addEventListener('change', onLangChange);
    elements.contextSlider.addEventListener('input', onContextChange);
    // Mic source toggle
    elements.micSourceToggle.addEventListener('change', onMicSourceChange);
    // ESP display mode
    elements.espDisplayModeSelect.addEventListener('change', onEspDisplayModeChange);

    // Warn before closing if recording
    window.addEventListener('beforeunload', () => {
        if (state.isRecording) {
            stopRecording();
        }
        if (state.websocket) {
            state.websocket.close();
        }
    });
}

function initVisualizer() {
    elements.visualizer.innerHTML = '';
    for (let i = 0; i < 20; i++) {
        const bar = document.createElement('div');
        bar.className = 'bar';
        elements.visualizer.appendChild(bar);
    }
}

// ── Mic Source Control ───────────────────────────────────

async function onMicSourceChange() {
    const previousSource = state.useEspMic;
    state.useEspMic = elements.micSourceToggle.checked;
    elements.micSourceLabel.textContent = state.useEspMic ? 'Using ESP Mic' : 'Use ESP Mic';
    
    sendRuntimeConfig();

    // Apply source switch immediately when recording to avoid mixed audio sessions.
    if (state.isRecording && previousSource !== state.useEspMic) {
        stopRecording(previousSource);
        await startRecording();
    }
}

function onEspDisplayModeChange() {
    state.espDisplayMode = elements.espDisplayModeSelect.value;
    sendRuntimeConfig();
}

function sendControlMessage(type) {
    if (state.websocket && state.websocket.readyState === WebSocket.OPEN) {
        state.websocket.send(JSON.stringify({ type }));
    }
}

// ── ESP WebSocket Connection ──────────────────────────────

function connectEspWebSocket() {
    if (state.espWebSocket) {
        state.espWebSocket.close();
    }
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}${CONFIG.ESP_WS_PATH}`;
    
    try {
        state.espWebSocket = new WebSocket(wsUrl);
        
        state.espWebSocket.onopen = () => {
            console.log('ESP WebSocket Connected');
            state.isEspConnected = true;
            // Send initial config to ESP
            sendRuntimeConfig();
        };
        
        state.espWebSocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleEspMessage(data);
            } catch (e) {
                console.error('ESP JSON Parse Error:', e);
            }
        };
        
        state.espWebSocket.onclose = (event) => {
            console.log('ESP WebSocket Closed', event.code, event.reason);
            state.isEspConnected = false;
            // Reconnect after 3 seconds
            setTimeout(() => {
                if (!state.isEspConnected) {
                    connectEspWebSocket();
                }
            }, 3000);
        };
        
        state.espWebSocket.onerror = (error) => {
            console.error('ESP WebSocket Error:', error);
        };
        
    } catch (error) {
        console.error('ESP Connection Error:', error);
    }
}

function handleEspMessage(data) {
    // Handle ESP-specific messages
    if (data.message_type === 'SessionBegins') {
        console.log('ESP Session started:', data);
    }
    if (data.message_type === 'ConfigAck') {
        console.log('ESP Config acknowledged:', data);
    }
    // Handle transcript messages from ESP (when using ESP mic)
    if (data.message_type === 'FinalTranscript') {
        addFinalTranscript(data.text, data.utterance_id);
    }
    if (data.message_type === 'TranslatedTranscript') {
        addTranslatedTranscript(data.text, data.original, data.utterance_id);
    }
    if (data.message_type === 'SpeechDetected') {
        elements.partialText.textContent = 'Listening...';
        elements.partialText.classList.add('listening');
    }
    if (data.message_type === 'SilenceDetected') {
        elements.partialText.textContent = 'Ready';
        elements.partialText.classList.remove('listening');
    }
}

// Send config to ESP via ESP WebSocket (this goes to backend via /wsesp)
function sendEspConfig() {
    if (state.espWebSocket && state.espWebSocket.readyState === WebSocket.OPEN) {
        state.espWebSocket.send(JSON.stringify({
            type: 'config',
            esp_display_mode: state.espDisplayMode,
            translate: state.translateEnabled,
            target_lang: state.targetLang,
            context_max: state.contextMax,
            source_lang: state.sourceLang,
            // Browser remains source-of-truth controller for runtime config
            client_type: 'browser',
        }));
    }
}

function sendBrowserConfig() {
    if (state.websocket && state.websocket.readyState === WebSocket.OPEN) {
        state.websocket.send(JSON.stringify({
            type: 'config',
            use_esp_mic: state.useEspMic,
            translate: state.translateEnabled,
            target_lang: state.targetLang,
            context_max: state.contextMax,
            source_lang: state.sourceLang,
            esp_display_mode: state.espDisplayMode,
            display_fallback: 'error',
            client_type: 'browser',
        }));
    }
}

function sendRuntimeConfig() {
    sendBrowserConfig();
}

// ── Source Language Control ────────────────────────────────

function onSourceLangChange() {
    state.sourceLang = elements.sourceLangSelect.value;
    // Send to ESP (for audio processing)
    sendRuntimeConfig();
}

// ── Translation Controls ──────────────────────────────────

function onTranslateToggle() {
    state.translateEnabled = elements.translateToggle.checked;
    elements.langSelect.disabled = !state.translateEnabled;
    elements.contextSlider.disabled = !state.translateEnabled;
    elements.toggleLabel.textContent = state.translateEnabled ? 'Translation On' : 'Translation Off';
    elements.toggleLabel.classList.toggle('active', state.translateEnabled);

    if (state.translateEnabled) {
        elements.translationColumn.classList.remove('hidden');
        elements.transcriptsContainer.classList.add('two-columns');
        elements.translationStatus.textContent = 'Ready';
    } else {
        elements.translationColumn.classList.add('hidden');
        elements.transcriptsContainer.classList.remove('two-columns');
        elements.translationStatus.textContent = 'Translation off';
    }

    // Send to ESP (for audio processing and display)
    sendRuntimeConfig();
}

function onLangChange() {
    state.targetLang = elements.langSelect.value;
    // Send to ESP
    sendRuntimeConfig();
}

function onContextChange() {
    state.contextMax = parseInt(elements.contextSlider.value, 10);
    elements.contextValue.textContent = state.contextMax;
    // Send to ESP
    sendRuntimeConfig();
}

// ── WebSocket Logic ───────────────────────────────────────

function connectWebSocket() {
    return new Promise((resolve, reject) => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}${CONFIG.WS_PATH}`;

        try {
            state.websocket = new WebSocket(wsUrl);
            updateStatus('connecting', 'Connecting...');

            state.websocket.onopen = () => {
                console.log('WebSocket Connected');
                updateStatus('connected', 'Connected');
                state.reconnectAttempts = 0;
                startPing();
                resolve();
            };

            state.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    handleServerMessage(data);
                } catch (e) {
                    console.error('JSON Parse Error:', e);
                }
            };

            state.websocket.onclose = (event) => {
                console.log('WebSocket Closed', event.code, event.reason);
                updateStatus('disconnected', 'Disconnected');
                stopPing();

                if (state.isRecording) {
                    stopRecording();
                }

                if (event.code === 1008) {
                    showNotification('Server Busy', 'Another client is currently connected. Please try again later.', 'error');
                } else if (!event.wasClean && state.reconnectAttempts < state.maxReconnectAttempts) {
                    showNotification('Connection Lost', 'WebSocket connection closed unexpectedly.', 'error');
                }
            };

            state.websocket.onerror = (error) => {
                console.error('WebSocket Error:', error);
                reject(error);
            };

        } catch (error) {
            reject(error);
        }
    });
}

function handleServerMessage(data) {
    if (!data.message_type) return;

    switch (data.message_type) {
        case 'ServerReady':
            updateStatus('ready', 'Ready');
            break;
        case 'SpeechDetected':
            elements.partialText.textContent = 'Listening...';
            elements.partialText.classList.add('listening');
            break;
        case 'SilenceDetected':
            elements.partialText.textContent = 'Ready';
            elements.partialText.classList.remove('listening');
            break;
        case 'FinalTranscript':
            addFinalTranscript(data.text, data.utterance_id);
            break;
        case 'TranslatedTranscript':
            addTranslatedTranscript(data.text, data.original, data.utterance_id);
            break;
        case 'Pong':
            handlePong(data);
            break;
        case 'ConfigAck':
            console.log('Config acknowledged:', data);
            break;
        case 'Error':
            showNotification('Server Error', data.text, 'error');
            break;
    }
}

// ── Audio Logic ───────────────────────────────────────────

async function startRecording() {
    try {
        elements.startBtn.disabled = true;

        // Ensure control/config channel is connected.
        if (!state.websocket || state.websocket.readyState !== WebSocket.OPEN) {
            await connectWebSocket();
        }
        sendRuntimeConfig();

        if (state.useEspMic) {
            // Use ESP mic: ask backend to start ESP stream and reset old utterance state.
            sendControlMessage('reset_session');
            sendControlMessage('esp_start');
            state.isEspStreaming = true;
            state.isRecording = true;
            updateUIState(true);
            updateStatus('recording', 'ESP Recording');
            elements.partialText.textContent = 'Waiting for ESP...';
        } else {
            // Use web mic
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: CONFIG.SAMPLE_RATE,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            state.mediaStream = stream;
            state.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: CONFIG.SAMPLE_RATE
            });

            const source = state.audioContext.createMediaStreamSource(stream);
            state.scriptProcessor = state.audioContext.createScriptProcessor(CONFIG.BUFFER_SIZE, 1, 1);

            state.scriptProcessor.onaudioprocess = processAudio;

            source.connect(state.scriptProcessor);
            state.scriptProcessor.connect(state.audioContext.destination);

            state.isRecording = true;
            updateUIState(true);
            updateStatus('recording', 'Recording');
        }

    } catch (error) {
        console.error('Start Recording Error:', error);
        showNotification('Microphone Error', 'Could not access microphone: ' + error.message, 'error');
        elements.startBtn.disabled = false;
        cleanupAudio();
    }
}

function processAudio(event) {
    if (!state.isRecording || state.useEspMic) return;

    const inputData = event.inputBuffer.getChannelData(0);

    // Float32 to Int16
    const int16Data = new Int16Array(inputData.length);
    let sum = 0;

    for (let i = 0; i < inputData.length; i++) {
        const s = Math.max(-1, Math.min(1, inputData[i]));
        int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        sum += s * s;
    }

    if (state.websocket && state.websocket.readyState === WebSocket.OPEN) {
        state.websocket.send(int16Data.buffer);
    }

    // Visualization
    const rms = Math.sqrt(sum / inputData.length);
    updateVisualization(rms);
}

function stopRecording(sourceOverride = null) {
    const activeSourceIsEsp = sourceOverride === null ? state.useEspMic : sourceOverride;
    state.isRecording = false;

    if (activeSourceIsEsp && state.isEspStreaming) {
        sendControlMessage('esp_stop');
        state.isEspStreaming = false;
    }
    sendControlMessage('reset_session');
    
    cleanupAudio();
    updateUIState(false);
    updateStatus('connected', 'Stopped');
}

function cleanupAudio() {
    if (state.scriptProcessor) {
        state.scriptProcessor.disconnect();
        state.scriptProcessor = null;
    }
    if (state.audioContext) {
        state.audioContext.close();
        state.audioContext = null;
    }
    if (state.mediaStream) {
        state.mediaStream.getTracks().forEach(track => track.stop());
        state.mediaStream = null;
    }

    // Reset visualizer
    const bars = document.querySelectorAll('.bar');
    bars.forEach(bar => bar.style.height = '4px');
}

// ── UI Updates ────────────────────────────────────────────

function updateStatus(status, text) {
    elements.statusText.textContent = text;
    elements.statusDot.className = 'status-dot ' + status;
}

function updateUIState(isRecording) {
    elements.startBtn.disabled = isRecording;
    elements.stopBtn.disabled = !isRecording;

    if (isRecording) {
        elements.startBtn.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="23"></line>
                <line x1="8" y1="23" x2="16" y2="23"></line>
            </svg>
            Recording...
        `;
    } else {
        elements.startBtn.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
            Start Recording
        `;
    }
}

// Track transcript IDs for matching translations
let transcriptCounter = 0;

function addFinalTranscript(text, utteranceId = null) {
    if (!text) return;

    let resolvedId = utteranceId;
    if (!Number.isInteger(resolvedId)) {
        transcriptCounter++;
        resolvedId = transcriptCounter;
    } else {
        transcriptCounter = Math.max(transcriptCounter, resolvedId);
    }
    const itemId = `transcript-${resolvedId}`;

    const div = document.createElement('div');
    div.className = 'final-transcript-item';
    div.dataset.id = itemId;
    div.dataset.text = text;

    const time = new Date().toLocaleTimeString();

    div.innerHTML = `
        <span class="timestamp">${escapeHtml(time)}</span>
        <span class="text">${escapeHtml(text)}</span>
    `;

    const shouldScroll = isNearBottom(elements.transcriptScroll);
    elements.finalTranscript.appendChild(div);
    if (shouldScroll) scrollToBottom(elements.transcriptScroll);

    // If translation is enabled, add a placeholder in translation column
    if (state.translateEnabled) {
        state.pendingTranslations++;
        state.pendingTranslationIds.push(itemId);
        addTranslationPlaceholder(itemId, time);
        updateTranslationStatus();
    }

    // Reset listening indicator
    elements.partialText.textContent = 'Ready';
    elements.partialText.classList.remove('listening');
}

function addTranslationPlaceholder(itemId, time) {
    const div = document.createElement('div');
    div.className = 'translated-transcript-item translating';
    div.dataset.id = itemId;

    div.innerHTML = `
        <span class="timestamp">${escapeHtml(time)}</span>
        <span class="text translating-text">Translating...</span>
    `;

    const shouldScroll = isNearBottom(elements.translationScroll);
    elements.translatedTranscript.appendChild(div);
    if (shouldScroll) scrollToBottom(elements.translationScroll);
}

function addTranslatedTranscript(translatedText, originalText, utteranceId = null) {
    if (!translatedText) return;

    let targetId = null;
    if (Number.isInteger(utteranceId)) {
        targetId = `transcript-${utteranceId}`;
    } else if (state.pendingTranslationIds.length > 0) {
        targetId = state.pendingTranslationIds.shift();
    }

    let matched = false;
    if (targetId) {
        const placeholder = elements.translatedTranscript.querySelector(`[data-id="${targetId}"]`);
        if (placeholder) {
            placeholder.classList.remove('translating');
            const textEl = placeholder.querySelector('.text');
            textEl.classList.remove('translating-text');
            textEl.textContent = translatedText;
            matched = true;
            state.pendingTranslations = Math.max(0, state.pendingTranslations - 1);
        }
    }

    // If no placeholder matched (edge case), append directly
    if (!matched) {
        const div = document.createElement('div');
        div.className = 'translated-transcript-item';

        const time = new Date().toLocaleTimeString();
        div.innerHTML = `
            <span class="timestamp">${escapeHtml(time)}</span>
            <span class="text">${escapeHtml(translatedText)}</span>
        `;

        elements.translatedTranscript.appendChild(div);
        state.pendingTranslations = Math.max(0, state.pendingTranslations - 1);
    }

    if (isNearBottom(elements.translationScroll)) scrollToBottom(elements.translationScroll);
    updateTranslationStatus();
}

function updateTranslationStatus() {
    if (state.pendingTranslations > 0) {
        elements.translationStatus.textContent = `Translating ${state.pendingTranslations} item(s)...`;
    } else {
        elements.translationStatus.textContent = 'Ready';
    }
}

function clearTranscripts() {
    elements.finalTranscript.innerHTML = '';
    elements.partialText.textContent = '...';
    elements.translatedTranscript.innerHTML = '';
    elements.translationStatus.textContent = state.translateEnabled ? 'Ready' : 'Translation off';
    state.pendingTranslations = 0;
    state.pendingTranslationIds = [];
    transcriptCounter = 0;
}

function updateVisualization(rms) {
    const bars = document.querySelectorAll('.bar');
    const intensity = Math.min(1, rms * 5);
    const t = performance.now() * 0.004;

    bars.forEach((bar, i) => {
        // Deterministic sine-wave pattern — no Math.random()
        const wave = Math.sin(i * 0.8 + t) * 0.5 + 0.5;
        const height = 4 + wave * 36 * intensity;
        bar.style.height = `${height}px`;
        bar.style.opacity = 0.5 + (intensity * 0.5);
    });
}

// ── Ping / Latency Measurement ────────────────────────────

function startPing() {
    stopPing();
    sendPing();
    state.pingInterval = setInterval(sendPing, 3000);
}

function stopPing() {
    if (state.pingInterval) {
        clearInterval(state.pingInterval);
        state.pingInterval = null;
    }
    if (elements.pingDisplay) {
        elements.pingDisplay.textContent = 'Ping: --';
    }
}

function sendPing() {
    if (state.websocket && state.websocket.readyState === WebSocket.OPEN) {
        const now = Date.now();
        state.websocket.send(JSON.stringify({ type: 'ping', t: now }));
    }
}

function handlePong(data) {
    if (data.t) {
        const rtt = Date.now() - data.t;
        if (elements.pingDisplay) {
            elements.pingDisplay.textContent = `Ping: ${rtt} ms`;
            elements.pingDisplay.className = 'ping-display ' +
                (rtt < 100 ? 'ping-good' : rtt < 300 ? 'ping-ok' : 'ping-bad');
        }
    }
}

function showNotification(title, message, type = 'info') {
    const notif = elements.notification;
    const titleEl = notif.querySelector('.notification-title');
    const msgEl = notif.querySelector('.notification-message');

    notif.className = `notification ${type} show`;
    titleEl.textContent = title;
    msgEl.textContent = message;

    setTimeout(() => {
        notif.classList.remove('show');
    }, 5000);
}
