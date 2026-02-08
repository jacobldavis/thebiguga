// ═══════════════════════════════════════════════════════════
//  offkey – Spectral Fingerprint (Mic → FFT → Hash)
// ═══════════════════════════════════════════════════════════

// ─── Configuration ───────────────────────────────────────
// Charset ordered so adjacent characters = similar sounds.
// a-z (26) + A-Z (26) + 0-9 (10) + symbols (10) = 72 chars
const CHARSET =
    'abcdefghijklmnopqrstuvwxyz' +
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ' +
    '0123456789' +
    '!@#$%^&*()';

const MAX_DURATION_S  = 5;        // max recording length
const NUM_BUCKETS     = 64;       // hash is always exactly this many characters
const FFT_SIZE        = 2048;     // power of 2 for FFT
const SILENCE_CHAR    = '-';      // emitted for silent buckets
const SILENCE_RMS     = 0.001;    // RMS below this → silence

// Spectral centroid is mapped from [FREQ_MIN .. FREQ_MAX] Hz
// onto the CHARSET using a logarithmic scale (perception is log).
const FREQ_MIN        = 50;       // Hz – lower bound
const FREQ_MAX        = 8000;     // Hz – upper bound

// ─── State ───────────────────────────────────────────────
let audioCtx      = null;
let mediaStream   = null;
let sourceNode    = null;
let processorNode = null;
let analyserNode  = null;
let recording     = false;
let capturedChunks = [];          // array of Float32Array chunks
let capturedSampleCount = 0;      // total samples captured so far
let animFrameId   = null;
let startTime     = 0;
let recSampleRate = 44100;

let currentHash    = '';
let currentWavBlob = null;
let currentMode    = null;        // 'record' or 'authenticate'
let awaitingUsername = false;      // waiting for username entry
let voiceDetected  = false;       // has voice crossed the RMS threshold?
let voiceSampleCount = 0;         // samples captured since voice detected
let aboutLoaded    = false;

// ─── DOM refs ────────────────────────────────────────────
const canvas        = document.getElementById('waveform');
const ctx           = canvas.getContext('2d');
const btnRecord     = document.getElementById('btn-record');
const btnAuth       = document.getElementById('btn-authenticate');
const btnWav        = document.getElementById('btn-save-wav');
const btnPlayback   = document.getElementById('btn-playback');
const recIndicator  = document.getElementById('rec-indicator');
const timerEl       = document.getElementById('timer');
const resultSection = document.getElementById('result-section');
const errorDisplay  = document.getElementById('error-display');
const usernameInput = document.getElementById('username-input');
const btnUsernameEnter = document.getElementById('btn-username-enter');
const authResultSection = document.getElementById('auth-result-section');
const authScoreDisplay  = document.getElementById('auth-score-display');
const aboutContentEl    = document.getElementById('about-content');
const spectrogramCanvas  = document.getElementById('spectrogram-canvas');
const spectrogramCtx     = spectrogramCanvas.getContext('2d');
const btnSaveSpectrogram = document.getElementById('btn-save-spectrogram');

// ─── Button event listeners ──────────────────────────────
btnRecord.addEventListener('click', () => {
    if (awaitingUsername) {
        awaitingUsername = false;
        usernameInput.classList.remove('prompting');
    }
    currentMode = 'record';
    toggleRecording().catch(err => {
        console.error('toggleRecording error:', err);
        showError('error: ' + err.message);
    });
});

btnAuth.addEventListener('click', () => {
    if (awaitingUsername) {
        awaitingUsername = false;
        usernameInput.classList.remove('prompting');
    }
    currentMode = 'authenticate';
    toggleRecording().catch(err => {
        console.error('toggleRecording error:', err);
        showError('error: ' + err.message);
    });
});

btnWav.addEventListener('click',  () => saveWAV());
btnPlayback.addEventListener('click', () => playRecording());
btnSaveSpectrogram.addEventListener('click', () => downloadSpectrogram());

// Username submit handler
btnUsernameEnter.addEventListener('click', () => submitUsername());
usernameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') submitUsername();
});

// Draw an idle flat-line on the canvas at load
function resizeCanvas() {
    canvas.width  = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
}
resizeCanvas();
window.addEventListener('resize', () => { resizeCanvas(); drawIdleLine(); });
drawIdleLine();

// ─── Inline error display (works even when alert() is blocked) ──
function showError(msg) {
    errorDisplay.textContent   = msg;
    errorDisplay.classList.remove('hidden');
    resultSection.classList.remove('hidden');
}

// ═════════════════════════════════════════════════════════
//  About Tab Markdown Loader
// ═════════════════════════════════════════════════════════

function escapeHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function formatInlineMarkdown(text) {
    let formatted = escapeHtml(text);
    formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/`(.+?)`/g, '<code>$1</code>');
    formatted = formatted.replace(/\[(.+?)\]\((https?:[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    return formatted;
}

function markdownToHtml(markdown) {
    const lines = markdown.split(/\r?\n/);
    let html = '';
    let inList = false;

    const closeList = () => {
        if (inList) {
            html += '</ul>';
            inList = false;
        }
    };

    for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) {
            closeList();
            continue;
        }

        if (line.startsWith('### ')) {
            closeList();
            html += `<h3>${formatInlineMarkdown(line.slice(4))}</h3>`;
            continue;
        }

        if (line.startsWith('## ')) {
            closeList();
            html += `<h2>${formatInlineMarkdown(line.slice(3))}</h2>`;
            continue;
        }

        if (line.startsWith('# ')) {
            closeList();
            html += `<h1>${formatInlineMarkdown(line.slice(2))}</h1>`;
            continue;
        }

        if (line.startsWith('- ')) {
            if (!inList) {
                html += '<ul>';
                inList = true;
            }
            html += `<li>${formatInlineMarkdown(line.slice(2))}</li>`;
            continue;
        }

        closeList();
        html += `<p>${formatInlineMarkdown(line)}</p>`;
    }

    closeList();
    return html;
}

async function loadAboutContent() {
    if (aboutLoaded || !aboutContentEl) return;
    try {
        const response = await fetch('http://localhost:8000/api/about/', { cache: 'no-cache' });
        if (!response.ok) throw new Error(`status ${response.status}`);
        const payload = await response.json();
        if (!payload || typeof payload.content !== 'string') {
            throw new Error('invalid payload');
        }
        aboutContentEl.innerHTML = markdownToHtml(payload.content);
        aboutLoaded = true;
        if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
            window.MathJax.typesetPromise([aboutContentEl]).catch(err => {
                console.error('MathJax render error:', err);
            });
        }
    } catch (error) {
        console.error('Error loading about content:', error);
        aboutContentEl.innerHTML = '<p class="about-error">could not load about page.</p>';
    }
}

// ═════════════════════════════════════════════════════════
//  Recording (Microphone → raw PCM via ScriptProcessorNode)
// ═════════════════════════════════════════════════════════

async function toggleRecording() {
    if (!recording) await startRecording();
    else            stopRecording();
}

async function startRecording() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation:  false,
                noiseSuppression:  false,
                autoGainControl:   true,   // boost quiet mic signals
            }
        });
    } catch (e) {
        console.error('getUserMedia failed:', e);
        showError('microphone access denied or unavailable: ' + e.message
            + '\n\ntip: open http://localhost:8000 in chrome/edge instead of the vs code simple browser.');
        return;
    }

    audioCtx      = new (window.AudioContext || window.webkitAudioContext)();
    recSampleRate  = audioCtx.sampleRate;
    sourceNode    = audioCtx.createMediaStreamSource(mediaStream);

    // Analyser → live waveform visualisation
    analyserNode          = audioCtx.createAnalyser();
    analyserNode.fftSize  = 2048;
    sourceNode.connect(analyserNode);

    // ScriptProcessor → capture raw PCM Float32 chunks
    processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
    capturedChunks = [];
    capturedSampleCount = 0;
    voiceDetected = false;
    voiceSampleCount = 0;
    const targetSamples = MAX_DURATION_S * recSampleRate;
    const VOICE_RMS_THRESHOLD = 0.015;  // RMS level that counts as "voice started"

    processorNode.onaudioprocess = (e) => {
        if (!recording) return;
        const buf = new Float32Array(e.inputBuffer.getChannelData(0));
        capturedChunks.push(buf);
        capturedSampleCount += buf.length;

        // Check if voice has been detected yet (RMS exceeds threshold)
        if (!voiceDetected) {
            let sum = 0;
            for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
            const rms = Math.sqrt(sum / buf.length);
            if (rms >= VOICE_RMS_THRESHOLD) {
                voiceDetected = true;
                timerEl.textContent = '0.0s';
            } else {
                timerEl.textContent = 'waiting for voice...';
                return;  // don't count pre-voice silence toward the 5s
            }
        }

        voiceSampleCount += buf.length;

        // Update timer based on samples since voice started
        const capturedSecs = voiceSampleCount / recSampleRate;
        timerEl.textContent = `${capturedSecs.toFixed(1)}s`;

        // Auto-stop once we've captured enough samples after voice start
        if (voiceSampleCount >= targetSamples) stopRecording();
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioCtx.destination); // must connect for events to fire

    recording  = true;
    startTime  = performance.now();
    currentHash    = '';
    currentWavBlob = null;

    // ── UI ──
    const activeBtn = (currentMode === 'authenticate') ? btnAuth : btnRecord;
    activeBtn.textContent = 'stop';
    activeBtn.classList.add('recording');
    // Disable the other button while recording
    const otherBtn = (currentMode === 'authenticate') ? btnRecord : btnAuth;
    otherBtn.disabled = true;
    recIndicator.classList.remove('hidden');
    resultSection.classList.add('hidden');
    authResultSection.classList.add('hidden');
    btnWav.disabled  = true;
    btnSaveSpectrogram.disabled = true;
    btnPlayback.disabled = true;

    // Live waveform
    drawLiveWaveform();
}

function stopRecording() {
    recording = false;
    cancelAnimationFrame(animFrameId);

    // Tear down audio graph
    if (processorNode) { processorNode.disconnect(); processorNode = null; }
    if (sourceNode)    { sourceNode.disconnect();    sourceNode    = null; }
    if (analyserNode)  { analyserNode.disconnect();  analyserNode  = null; }
    if (mediaStream)   { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    if (audioCtx)      { audioCtx.close(); audioCtx = null; }

    // ── UI ──
    const activeBtn = (currentMode === 'authenticate') ? btnAuth : btnRecord;
    activeBtn.textContent = (currentMode === 'authenticate') ? 'authenticate' : 'record';
    activeBtn.classList.remove('recording');
    const otherBtn = (currentMode === 'authenticate') ? btnRecord : btnAuth;
    otherBtn.disabled = false;
    recIndicator.classList.add('hidden');

    // ── Build WAV blob from captured audio ──
    const totalLen = capturedChunks.reduce((n, c) => n + c.length, 0);
    const samples  = new Float32Array(totalLen);
    let off = 0;
    for (const chunk of capturedChunks) { samples.set(chunk, off); off += chunk.length; }
    capturedChunks = [];

    if (samples.length < FFT_SIZE) {
        showError('recording too short – please record at least a brief sound.');
        drawIdleLine();
        return;
    }

    // ── Trim leading/trailing silence ──
    const trimmed = trimSilence(samples, recSampleRate);
    if (trimmed.length < FFT_SIZE) {
        showError('recording too quiet – please speak louder.');
        drawIdleLine();
        return;
    }

    currentWavBlob = samplesToWav(trimmed, recSampleRate);
    drawStaticWaveform(trimmed);

    // ── Generate spectrogram visualization ──
    generateSpectrogram(trimmed, recSampleRate);

    // ── Show save buttons immediately ──
    btnWav.disabled = false;
    btnPlayback.disabled = false;
    errorDisplay.classList.add('hidden');
    resultSection.classList.remove('hidden');
    processAudio();

    // ── Prompt for username ──
    awaitingUsername = true;
    usernameInput.value = '';
    usernameInput.classList.add('prompting');
    usernameInput.focus();
    timerEl.textContent = currentMode === 'record'
        ? 'enter username to save...'
        : 'enter username to authenticate...';
}

// ═══════════════════════════════════════════════════════════
//  Username Submission  →  Save or Authenticate
// ═══════════════════════════════════════════════════════════

async function submitUsername() {
    if (!awaitingUsername || !currentWavBlob) return;

    const username = usernameInput.value.trim();
    if (!username) {
        usernameInput.focus();
        return;
    }

    awaitingUsername = false;
    usernameInput.classList.remove('prompting');
    timerEl.textContent = '';

    if (currentMode === 'record') {
        await saveVoicePrint(username);
    } else if (currentMode === 'authenticate') {
        await authenticateVoice(username);
    }
}

async function saveVoicePrint(username) {
    timerEl.textContent = 'saving...';
    try {
        const formData = new FormData();
        formData.append('audio', currentWavBlob, 'recording.wav');
        formData.append('username', username);

        const response = await fetch('http://localhost:8000/api/register-voice/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'server error');
        }

        const result = await response.json();
        timerEl.textContent = `saved voice print for "${username}"`;

        // Also process spectral hash for display
        await processAudio();

    } catch (error) {
        console.error('Error saving voice print:', error);
        showError('error saving: ' + error.message);
    }
}

async function authenticateVoice(username) {
    timerEl.textContent = 'authenticating...';
    try {
        const formData = new FormData();
        formData.append('audio', currentWavBlob, 'recording.wav');
        formData.append('username', username);

        const response = await fetch('http://localhost:8000/api/authenticate-voice/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'server error');
        }

        const result = await response.json();

        // Display auth score
        if (result.error) {
            authScoreDisplay.innerHTML = `<span class="score-error">${result.error}</span>`;
        } else {
            const score = parseFloat(result.similarity);
            const pct = (score * 100).toFixed(1);
            const scoreClass = score >= 0.8 ? 'score-good' : 'score-bad';
            authScoreDisplay.innerHTML =
                `<span class="${scoreClass}">${pct}%</span><span class="score-label">similarity to ${username}'s voice print</span>`;
        }
        authResultSection.classList.remove('hidden');
        timerEl.textContent = '';

    } catch (error) {
        console.error('Error authenticating:', error);
        authScoreDisplay.innerHTML = `<span class="score-error">error: ${error.message}</span>`;
        authResultSection.classList.remove('hidden');
        timerEl.textContent = '';
    }
}

// ═══════════════════════════════════════════════════════════
//  Audio Processing  →  Send to Backend for Spectral Hash
// ═══════════════════════════════════════════════════════════

async function processAudio() {
    if (!currentWavBlob) return;

    // Send WAV to backend for spectral hash processing
    try {
        const formData = new FormData();
        formData.append('audio', currentWavBlob, 'recording.wav');

        const response = await fetch('http://localhost:8000/api/process-audio/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'server error');
        }

        const result = await response.json();
        currentHash = result.hash;

        // ── Store hash result (displayed via spectrogram now) ──
        resultSection.classList.remove('hidden');
        btnWav.disabled  = false;
        btnPlayback.disabled = false;

    } catch (error) {
        console.error('Error processing audio:', error);
        showError('error processing audio: ' + error.message);
        drawIdleLine();
    }
}

// ═════════════════════════════════════════════════════════
//  Spectrogram Generation (client-side FFT → magical color map)
// ═════════════════════════════════════════════════════════

function generateSpectrogram(samples, sampleRate) {
    const fftSize = 1024;
    const hop     = 256;
    const nFrames = Math.floor((samples.length - fftSize) / hop);
    const nBins   = fftSize / 2;

    if (nFrames < 2) return;

    // Compute STFT magnitude (simple Hann-windowed FFT per frame)
    const magnitudes = [];
    const hann = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) {
        hann[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (fftSize - 1)));
    }

    for (let f = 0; f < nFrames; f++) {
        const start = f * hop;
        const windowed = new Float32Array(fftSize);
        for (let i = 0; i < fftSize; i++) {
            windowed[i] = (samples[start + i] || 0) * hann[i];
        }
        const mag = computeFFTMagnitude(windowed, fftSize);
        magnitudes.push(mag);
    }

    // Convert to log scale (dB)
    let maxDb = -Infinity, minDb = Infinity;
    const dbFrames = magnitudes.map(mag => {
        const db = new Float32Array(nBins);
        for (let i = 0; i < nBins; i++) {
            db[i] = 20 * Math.log10(mag[i] + 1e-10);
            if (db[i] > maxDb) maxDb = db[i];
            if (db[i] < minDb) minDb = db[i];
        }
        return db;
    });

    // Clamp dynamic range so quiet noise doesn't wash out
    const floorDb = maxDb - 80;
    minDb = Math.max(minDb, floorDb);
    const range = maxDb - minDb || 1;

    // ── Canvas layout with axis margins ──
    const w = 1280;  // hi-dpi: 640 CSS px × 2
    const h = 480;   // hi-dpi: 240 CSS px × 2
    spectrogramCanvas.width  = w;
    spectrogramCanvas.height = h;

    const marginLeft   = 80;   // space for frequency labels
    const marginBottom = 36;   // space for time labels
    const marginTop    = 12;
    const marginRight  = 12;
    const plotW = w - marginLeft - marginRight;
    const plotH = h - marginTop - marginBottom;

    // ── Log-frequency mapping ──
    const nyquist = sampleRate / 2;
    const freqMin = 500;          // Hz — lower display bound
    const freqMax = Math.min(nyquist, 8000); // Hz — upper display bound
    const logMin  = Math.log(freqMin);
    const logMax  = Math.log(freqMax);

    // Helper: log-freq position → y pixel (0 = top of plot area)
    function freqToY(freq) {
        const t = (Math.log(freq) - logMin) / (logMax - logMin);
        return plotH * (1 - t); // flip so low freq at bottom
    }

    // Helper: bin index → frequency
    const binToFreq = (bin) => bin * nyquist / nBins;

    // ── Fill background ──
    spectrogramCtx.fillStyle = '#e8d6a8';
    spectrogramCtx.fillRect(0, 0, w, h);

    // ── Render spectrogram pixels ──
    const imgData = spectrogramCtx.createImageData(plotW, plotH);
    const data = imgData.data;

    for (let px = 0; px < plotW; px++) {
        const frameIdx = Math.min(Math.floor(px / plotW * nFrames), nFrames - 1);
        const frame = dbFrames[frameIdx];

        for (let py = 0; py < plotH; py++) {
            // Map pixel row → frequency via log scale
            const tNorm = 1 - py / plotH;  // 0 = bottom (low freq), 1 = top (high freq)
            const freq  = Math.exp(logMin + tNorm * (logMax - logMin));
            // Map frequency → FFT bin (linear interpolation between bins)
            const exactBin = freq / nyquist * nBins;
            const binLo  = Math.floor(exactBin);
            const binHi  = Math.min(binLo + 1, nBins - 1);
            const frac   = exactBin - binLo;
            const dbVal  = frame[binLo] * (1 - frac) + frame[binHi] * frac;

            const val = Math.max(0, Math.min(1, (dbVal - minDb) / range));
            const [r, g, b] = spectrogramColorMap(val);

            const idx = (py * plotW + px) * 4;
            data[idx]     = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
            data[idx + 3] = 255;
        }
    }

    spectrogramCtx.putImageData(imgData, marginLeft, marginTop);

    // ── Draw frequency axis labels (log-spaced) ──
    spectrogramCtx.fillStyle = 'rgba(92, 79, 61, 0.7)';
    spectrogramCtx.font = `${Math.round(h * 0.038)}px Georgia`;
    spectrogramCtx.textAlign = 'right';
    const freqLabels = [100, 200, 500, 1000, 2000, 4000, 8000].filter(f => f >= freqMin && f <= freqMax);
    for (const freq of freqLabels) {
        const yPos = marginTop + freqToY(freq);
        const label = freq >= 1000 ? (freq / 1000) + 'k' : String(freq);
        spectrogramCtx.fillText(label + ' hz', marginLeft - 8, yPos + 5);
        // Faint grid line
        spectrogramCtx.strokeStyle = 'rgba(92, 79, 61, 0.1)';
        spectrogramCtx.beginPath();
        spectrogramCtx.moveTo(marginLeft, yPos);
        spectrogramCtx.lineTo(w - marginRight, yPos);
        spectrogramCtx.stroke();
    }

    // ── Draw time axis labels ──
    spectrogramCtx.textAlign = 'center';
    const duration = samples.length / sampleRate;
    const timeStep = duration > 3 ? 1.0 : duration > 1 ? 0.5 : 0.25;
    for (let t = 0; t <= duration; t += timeStep) {
        const xPos = marginLeft + (t / duration) * plotW;
        spectrogramCtx.fillText(t.toFixed(1) + 's', xPos, h - 6);
    }

    // Enable download button
    btnSaveSpectrogram.disabled = false;
}

function spectrogramColorMap(t) {
    // Pink → purple gradient (purple = highest intensity)
    const stops = [
        [0.00, 232, 214, 168],  // parchment (background)
        [0.10, 230, 200, 180],  // warm blush
        [0.25, 220, 160, 170],  // dusty rose
        [0.40, 210, 120, 160],  // pink
        [0.55, 190, 80,  150],  // hot pink
        [0.70, 160, 50,  140],  // magenta
        [0.82, 120, 30,  140],  // deep magenta
        [0.92, 85,  20,  130],  // dark purple
        [1.00, 55,  10,  100],  // deepest purple
    ];

    // Find surrounding stops
    let lo = stops[0], hi = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) {
        if (t >= stops[i][0] && t <= stops[i + 1][0]) {
            lo = stops[i];
            hi = stops[i + 1];
            break;
        }
    }

    const f = (t - lo[0]) / (hi[0] - lo[0] + 1e-8);
    return [
        Math.round(lo[1] + f * (hi[1] - lo[1])),
        Math.round(lo[2] + f * (hi[2] - lo[2])),
        Math.round(lo[3] + f * (hi[3] - lo[3])),
    ];
}

function computeFFTMagnitude(real, n) {
    // Radix-2 in-place FFT (Cooley-Tukey)
    const imag = new Float32Array(n);
    // Bit-reverse permutation
    let j = 0;
    for (let i = 0; i < n; i++) {
        if (i < j) {
            [real[i], real[j]] = [real[j], real[i]];
        }
        let m = n >> 1;
        while (m >= 1 && j >= m) { j -= m; m >>= 1; }
        j += m;
    }
    // FFT butterfly
    for (let size = 2; size <= n; size *= 2) {
        const halfSize = size / 2;
        const angle = -2 * Math.PI / size;
        for (let i = 0; i < n; i += size) {
            for (let k = 0; k < halfSize; k++) {
                const twR = Math.cos(angle * k);
                const twI = Math.sin(angle * k);
                const idx1 = i + k;
                const idx2 = i + k + halfSize;
                const tR = twR * real[idx2] - twI * imag[idx2];
                const tI = twR * imag[idx2] + twI * real[idx2];
                real[idx2] = real[idx1] - tR;
                imag[idx2] = imag[idx1] - tI;
                real[idx1] += tR;
                imag[idx1] += tI;
            }
        }
    }
    // Return magnitudes for first half
    const mag = new Float32Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
        mag[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
    }
    return mag;
}

function downloadSpectrogram() {
    spectrogramCanvas.toBlob((blob) => {
        if (!blob) return;
        download(blob, 'spectrogram.png');
    }, 'image/png');
}

// ═════════════════════════════════════════════════════════
//  Canvas Visualisation
// ═════════════════════════════════════════════════════════

function drawIdleLine() {
    ctx.fillStyle = '#e8d6a8';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#c8b080';
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
}

function drawLiveWaveform() {
    if (!analyserNode) return;
    const bufLen = analyserNode.frequencyBinCount;
    const data   = new Uint8Array(bufLen);

    function frame() {
        if (!recording) return;
        animFrameId = requestAnimationFrame(frame);

        analyserNode.getByteTimeDomainData(data);
        ctx.fillStyle = '#e8d6a8';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth   = 2;
        ctx.strokeStyle = '#7a5c3a';
        ctx.beginPath();

        const sliceW = canvas.width / bufLen;
        let x = 0;
        for (let i = 0; i < bufLen; i++) {
            const y = (data[i] / 255) * canvas.height;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            x += sliceW;
        }
        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    }
    frame();
}

function drawStaticWaveform(samples) {
    ctx.fillStyle = '#e8d6a8';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const step = Math.ceil(samples.length / canvas.width);
    const amp  = canvas.height / 2;

    ctx.lineWidth   = 1;
    ctx.strokeStyle = '#b8a078';
    ctx.beginPath();

    for (let i = 0; i < canvas.width; i++) {
        let min = 1, max = -1;
        for (let j = 0; j < step; j++) {
            const s = samples[i * step + j] || 0;
            if (s < min) min = s;
            if (s > max) max = s;
        }
        ctx.moveTo(i, amp + min * amp);
        ctx.lineTo(i, amp + max * amp);
    }
    ctx.stroke();
}

// ═════════════════════════════════════════════════════════
//  Silence Trimmer  (matches librosa.effects.trim top_db=30)
// ═════════════════════════════════════════════════════════

function trimSilence(samples, sampleRate, topDb = 35) {
    // Compute RMS in short frames and find where it exceeds the threshold
    const frameLen = Math.floor(sampleRate * 0.02);  // 20ms frames
    const hop      = Math.floor(frameLen / 2);        // 50% overlap
    const nFrames  = Math.floor((samples.length - frameLen) / hop) + 1;

    if (nFrames < 1) return samples;

    // Compute per-frame RMS
    const rms = new Float32Array(nFrames);
    let maxRms = 0;
    for (let i = 0; i < nFrames; i++) {
        let sum = 0;
        const base = i * hop;
        for (let j = 0; j < frameLen; j++) {
            const s = samples[base + j];
            sum += s * s;
        }
        rms[i] = Math.sqrt(sum / frameLen);
        if (rms[i] > maxRms) maxRms = rms[i];
    }

    if (maxRms < 1e-8) return samples;  // all silence

    // Threshold: topDb below the peak RMS
    const threshold = maxRms * Math.pow(10, -topDb / 20);

    // Find first and last frame above threshold
    let startFrame = 0;
    let endFrame   = nFrames - 1;
    while (startFrame < nFrames && rms[startFrame] < threshold) startFrame++;
    while (endFrame > startFrame && rms[endFrame]   < threshold) endFrame--;

    // Convert frames back to sample indices
    const startSample = startFrame * hop;
    const endSample   = Math.min(endFrame * hop + frameLen, samples.length);

    // Add a tiny margin (50ms) to avoid cutting speech onsets/releases
    const margin = Math.floor(sampleRate * 0.05);
    const trimStart = Math.max(0, startSample - margin);
    const trimEnd   = Math.min(samples.length, endSample + margin);

    return samples.slice(trimStart, trimEnd);
}

// ═════════════════════════════════════════════════════════
//  WAV Encoder  (PCM 16-bit mono)
// ═════════════════════════════════════════════════════════

function samplesToWav(samples, sampleRate) {
    const numCh   = 1;
    const bps     = 16;
    const byteRate   = sampleRate * numCh * (bps / 8);
    const blockAlign = numCh * (bps / 8);
    const dataSize   = samples.length * (bps / 8);
    const bufSize    = 44 + dataSize;
    const buf        = new ArrayBuffer(bufSize);
    const v          = new DataView(buf);

    let o = 0;
    const ws = (s) => { for (let i = 0; i < s.length; i++) v.setUint8(o++, s.charCodeAt(i)); };
    const w32 = (n) => { v.setUint32(o, n, true); o += 4; };
    const w16 = (n) => { v.setUint16(o, n, true); o += 2; };

    ws('RIFF'); w32(bufSize - 8); ws('WAVE');
    ws('fmt '); w32(16); w16(1); w16(numCh);
    w32(sampleRate); w32(byteRate); w16(blockAlign); w16(bps);
    ws('data'); w32(dataSize);

    for (let i = 0; i < samples.length; i++) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        s = s < 0 ? s * 0x8000 : s * 0x7FFF;
        v.setInt16(o, s | 0, true);
        o += 2;
    }

    return new Blob([buf], { type: 'audio/wav' });
}

// ═════════════════════════════════════════════════════════
//  Save / Download helpers
// ═════════════════════════════════════════════════════════

function saveWAV() {
    if (!currentWavBlob) return;
    download(currentWavBlob, 'recording.wav');
}

function playRecording() {
    if (!currentWavBlob) return;
    const audio = new Audio(URL.createObjectURL(currentWavBlob));
    audio.play();
}

function download(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a   = document.createElement('a');
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

// ═══════════════════════════════════════════════════════════
//  Cursor Trail Sparks
// ═══════════════════════════════════════════════════════════

const sparks = [];
const maxSparks = 30;
const lilacColors = ['#e8b8d8', '#d8a8e8', '#f8c8e8', '#c8a8d8'];

document.addEventListener('mousemove', (e) => {
    // Check if mouse is over content cards only (not header)
    const target = e.target;
    const isOverButton = target.closest('button');
    const isOverWaveform = target.closest('#waveform');
    
    // Create spark
    const spark = document.createElement('div');
    spark.className = 'cursor-spark';
    spark.style.left = e.pageX + 'px';
    spark.style.top = e.pageY + 'px';
    spark.style.opacity = (isOverButton || isOverWaveform) ? '0' : '1';
    
    // Random color from lilac palette
    spark.style.background = lilacColors[Math.floor(Math.random() * lilacColors.length)];
    
    // Random offset
    const offsetX = (Math.random() - 0.5) * 30;
    const offsetY = (Math.random() - 0.5) * 30;
    spark.style.setProperty('--offset-x', offsetX + 'px');
    spark.style.setProperty('--offset-y', offsetY + 'px');
    
    document.body.appendChild(spark);
    sparks.push(spark);
    
    // Remove old sparks
    if (sparks.length > maxSparks) {
        const oldSpark = sparks.shift();
        oldSpark.remove();
    }
    
    // Fade out and remove
    setTimeout(() => {
        spark.style.opacity = '0';
        setTimeout(() => {
            spark.remove();
            const idx = sparks.indexOf(spark);
            if (idx > -1) sparks.splice(idx, 1);
        }, 800);
    }, 150);
});

// ═══════════════════════════════════════════════════════════
//  Background Orbs Animation
// ═══════════════════════════════════════════════════════════

function getRandomColor() {
    // Generate random hue (0-360), moderate saturation, and darker values
    const hue = Math.floor(Math.random() * 360);
    const saturation = Math.floor(Math.random() * 40) + 20; // 20-60%
    const lightness = Math.floor(Math.random() * 15) + 10; // 10-25% (dark)
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

function createRandomGradient() {
    const center = getRandomColor();
    const mid1 = getRandomColor();
    const mid2 = getRandomColor();
    
    // Random choice between ripple (3 stops) or pure gradient (5 stops)
    if (Math.random() > 0.5) {
        // Ripple effect - lighter start, darker middle
        return `radial-gradient(circle, ${center} 0%, ${mid1} 35%, ${mid2} 60%, #f4ede4 100%)`;
    } else {
        // Pure gradient - very dark center
        const mid3 = getRandomColor();
        return `radial-gradient(circle, ${center} 0%, ${mid1} 25%, ${mid2} 50%, ${mid3} 75%, #f4ede4 100%)`;
    }
}

function createBackgroundOrb() {
    const orb = document.createElement('div');
    orb.className = 'bg-orb';
    
    // Random size (large and subtle)
    const size = Math.random() * 200 + 150;
    orb.style.width = size + 'px';
    orb.style.height = size + 'px';
    orb.style.setProperty('--orb-size', size + 'px');
    
    // Random horizontal position
    const startX = Math.random() * window.innerWidth;
    orb.style.left = startX + 'px';
    orb.style.top = -size + 'px';
    
    // Random gradient with random colors
    orb.style.background = createRandomGradient();
    
    // Random drift
    const driftX = (Math.random() - 0.5) * 100;
    orb.style.setProperty('--drift-x', driftX + 'px');
    
    // Faster duration (15-25 seconds)
    const duration = Math.random() * 10 + 15;
    orb.style.animationDuration = duration + 's';
    
    document.body.appendChild(orb);
    
    // Remove after animation completes (orb will be fully past bottom)
    setTimeout(() => {
        orb.remove();
    }, duration * 1000);
}

// Create orbs periodically
setInterval(createBackgroundOrb, 8000);
// Create initial orbs
for (let i = 0; i < 3; i++) {
    setTimeout(createBackgroundOrb, i * 2000);
}

// ═══════════════════════════════════════════════════════════
//  Tab Navigation
// ═══════════════════════════════════════════════════════════

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');

        // Redraw idle lines on the captcha canvases when switching to that tab
        if (btn.dataset.tab === 'captcha') {
            captchaResizeCanvases();
            captchaDrawIdle(captchaChallengeCtx, captchaChallengeCanvas);
            captchaDrawIdle(captchaResponseCtx, captchaResponseCanvas);
            if (captchaChallengeSamples) captchaDrawStatic(captchaChallengeCtx, captchaChallengeCanvas, captchaChallengeSamples);
            if (captchaResponseSamples)  captchaDrawStatic(captchaResponseCtx, captchaResponseCanvas, captchaResponseSamples);
        } else if (btn.dataset.tab === 'about') {
            loadAboutContent();
        } else {
            resizeCanvas();
            drawIdleLine();
        }
    });
});

// ═══════════════════════════════════════════════════════════
//  Audio Captcha System
// ═══════════════════════════════════════════════════════════

// ─── Captcha DOM refs ────────────────────────────────────
const captchaChallengeCanvas  = document.getElementById('captcha-waveform');
const captchaChallengeCtx     = captchaChallengeCanvas.getContext('2d');
const captchaResponseCanvas   = document.getElementById('captcha-response-waveform');
const captchaResponseCtx      = captchaResponseCanvas.getContext('2d');
const btnCaptchaGenerate      = document.getElementById('btn-captcha-generate');
const btnCaptchaPlay          = document.getElementById('btn-captcha-play');
const btnCaptchaRecord        = document.getElementById('btn-captcha-record');
const btnCaptchaSaveChallenge  = document.getElementById('btn-captcha-save-challenge');
const btnCaptchaSaveResponse   = document.getElementById('btn-captcha-save-response');
const captchaRecIndicator     = document.getElementById('captcha-rec-indicator');
const captchaTimerEl          = document.getElementById('captcha-timer');
const captchaResultSection    = document.getElementById('captcha-result-section');
const captchaScoreDisplay     = document.getElementById('captcha-score-display');

// ─── Captcha state ───────────────────────────────────────
const CAPTCHA_SAMPLE_RATE = 44100;
const CAPTCHA_DURATION_S  = 3;
const CAPTCHA_REC_DURATION_S = 4; // give user a little extra time
let captchaTapTimes       = [];   // onset times in seconds for the challenge
let captchaChallengeSamples = null;
let captchaChallengeBlob  = null;
let captchaResponseSamples = null;
let captchaResponseBlob   = null;
let captchaRecording      = false;
let captchaAudioCtx       = null;
let captchaMediaStream    = null;
let captchaSourceNode     = null;
let captchaProcessorNode  = null;
let captchaAnalyserNode   = null;
let captchaChunks         = [];
let captchaSampleCount    = 0;
let captchaAnimFrame      = null;

// ─── Canvas helpers ──────────────────────────────────────
function captchaResizeCanvases() {
    captchaChallengeCanvas.width  = captchaChallengeCanvas.clientWidth;
    captchaChallengeCanvas.height = captchaChallengeCanvas.clientHeight;
    captchaResponseCanvas.width   = captchaResponseCanvas.clientWidth;
    captchaResponseCanvas.height  = captchaResponseCanvas.clientHeight;
}

function captchaDrawIdle(drawCtx, cvs) {
    drawCtx.fillStyle = '#e8d6a8';
    drawCtx.fillRect(0, 0, cvs.width, cvs.height);
    drawCtx.strokeStyle = '#c8b080';
    drawCtx.beginPath();
    drawCtx.moveTo(0, cvs.height / 2);
    drawCtx.lineTo(cvs.width, cvs.height / 2);
    drawCtx.stroke();
}

function captchaDrawStatic(drawCtx, cvs, samples) {
    drawCtx.fillStyle = '#e8d6a8';
    drawCtx.fillRect(0, 0, cvs.width, cvs.height);

    const step = Math.ceil(samples.length / cvs.width);
    const amp  = cvs.height / 2;

    drawCtx.lineWidth   = 1;
    drawCtx.strokeStyle = '#b8a078';
    drawCtx.beginPath();

    for (let i = 0; i < cvs.width; i++) {
        let min = 1, max = -1;
        for (let j = 0; j < step; j++) {
            const s = samples[i * step + j] || 0;
            if (s < min) min = s;
            if (s > max) max = s;
        }
        drawCtx.moveTo(i, amp + min * amp);
        drawCtx.lineTo(i, amp + max * amp);
    }
    drawCtx.stroke();
}

// ─── Generate rhythmic pattern ───────────────────────────
// Creates a sequence of short "tap" impulses at random-ish intervals

function generateTapPattern() {
    const sr       = CAPTCHA_SAMPLE_RATE;
    const duration = CAPTCHA_DURATION_S;
    const totalSamples = sr * duration;
    const samples  = new Float32Array(totalSamples);

    // Generate 4–7 taps spread across the duration
    const numTaps = Math.floor(Math.random() * 4) + 4; // 4-7
    const tapTimes = [];

    // First tap between 0.15s and 0.4s
    tapTimes.push(0.15 + Math.random() * 0.25);

    // Subsequent taps with varying intervals (0.25s–0.65s)
    for (let i = 1; i < numTaps; i++) {
        const gap = 0.25 + Math.random() * 0.4;
        const next = tapTimes[i - 1] + gap;
        if (next > duration - 0.1) break;
        tapTimes.push(next);
    }

    captchaTapTimes = tapTimes;

    // Synthesize each tap as a short burst (damped sine click)
    for (const t of tapTimes) {
        const startSample = Math.floor(t * sr);
        const tapLen = Math.floor(0.02 * sr); // 20ms tap
        const freq = 800 + Math.random() * 400; // 800-1200 Hz
        for (let i = 0; i < tapLen && (startSample + i) < totalSamples; i++) {
            const env = Math.exp(-i / (0.004 * sr)); // fast decay
            samples[startSample + i] += 0.8 * env * Math.sin(2 * Math.PI * freq * i / sr);
        }
    }

    captchaChallengeSamples = samples;

    // Build WAV blob for playback
    captchaChallengeBlob = captchaSamplesToWav(samples, sr);

    // Draw on challenge canvas
    captchaResizeCanvases();
    captchaDrawStatic(captchaChallengeCtx, captchaChallengeCanvas, samples);

    // Reset response
    captchaResponseSamples = null;
    captchaDrawIdle(captchaResponseCtx, captchaResponseCanvas);
    captchaResultSection.classList.add('hidden');
    captchaTimerEl.textContent = `${tapTimes.length} taps — listen, then tap it back`;

    btnCaptchaPlay.disabled   = false;
    btnCaptchaRecord.disabled = false;
    btnCaptchaSaveChallenge.disabled = false;
    btnCaptchaSaveResponse.disabled  = true;
}

// ─── Play challenge audio ────────────────────────────────
function playCaptchaChallenge() {
    if (!captchaChallengeBlob) return;
    const audio = new Audio(URL.createObjectURL(captchaChallengeBlob));
    audio.play();
}

// ─── Record user response ────────────────────────────────
async function toggleCaptchaRecording() {
    if (!captchaRecording) await startCaptchaRecording();
    else stopCaptchaRecording();
}

async function startCaptchaRecording() {
    try {
        captchaMediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl:  true,
            }
        });
    } catch (e) {
        captchaTimerEl.textContent = 'mic access denied';
        return;
    }

    captchaAudioCtx   = new (window.AudioContext || window.webkitAudioContext)();
    const sr          = captchaAudioCtx.sampleRate;
    captchaSourceNode = captchaAudioCtx.createMediaStreamSource(captchaMediaStream);

    captchaAnalyserNode         = captchaAudioCtx.createAnalyser();
    captchaAnalyserNode.fftSize = 2048;
    captchaSourceNode.connect(captchaAnalyserNode);

    captchaProcessorNode = captchaAudioCtx.createScriptProcessor(4096, 1, 1);
    captchaChunks      = [];
    captchaSampleCount = 0;
    const targetSamples = CAPTCHA_REC_DURATION_S * sr;

    captchaProcessorNode.onaudioprocess = (e) => {
        if (!captchaRecording) return;
        const buf = new Float32Array(e.inputBuffer.getChannelData(0));
        captchaChunks.push(buf);
        captchaSampleCount += buf.length;

        const capturedSecs = captchaSampleCount / sr;
        captchaTimerEl.textContent = `${capturedSecs.toFixed(1)}s`;

        if (captchaSampleCount >= targetSamples) stopCaptchaRecording();
    };

    captchaSourceNode.connect(captchaProcessorNode);
    captchaProcessorNode.connect(captchaAudioCtx.destination);

    captchaRecording = true;

    btnCaptchaRecord.textContent = 'stop';
    btnCaptchaRecord.classList.add('recording');
    btnCaptchaGenerate.disabled = true;
    btnCaptchaPlay.disabled     = true;
    captchaRecIndicator.classList.remove('hidden');
    captchaResultSection.classList.add('hidden');

    // Live waveform
    captchaDrawLive();
}

function stopCaptchaRecording() {
    captchaRecording = false;
    cancelAnimationFrame(captchaAnimFrame);

    if (captchaProcessorNode) { captchaProcessorNode.disconnect(); captchaProcessorNode = null; }
    if (captchaSourceNode)    { captchaSourceNode.disconnect();    captchaSourceNode    = null; }
    if (captchaAnalyserNode)  { captchaAnalyserNode.disconnect();  captchaAnalyserNode  = null; }
    if (captchaMediaStream)   { captchaMediaStream.getTracks().forEach(t => t.stop()); captchaMediaStream = null; }

    const sr = captchaAudioCtx ? captchaAudioCtx.sampleRate : CAPTCHA_SAMPLE_RATE;
    if (captchaAudioCtx) { captchaAudioCtx.close(); captchaAudioCtx = null; }

    btnCaptchaRecord.textContent = 'record';
    btnCaptchaRecord.classList.remove('recording');
    btnCaptchaGenerate.disabled = false;
    btnCaptchaPlay.disabled     = false;
    captchaRecIndicator.classList.add('hidden');

    // Merge chunks
    const totalLen = captchaChunks.reduce((n, c) => n + c.length, 0);
    const samples  = new Float32Array(totalLen);
    let off = 0;
    for (const chunk of captchaChunks) { samples.set(chunk, off); off += chunk.length; }
    captchaChunks = [];

    if (samples.length < 2048) {
        captchaTimerEl.textContent = 'too short — try again';
        captchaDrawIdle(captchaResponseCtx, captchaResponseCanvas);
        return;
    }

    captchaResponseSamples = samples;
    captchaResponseBlob = captchaSamplesToWav(samples, sr);
    btnCaptchaSaveResponse.disabled = false;
    captchaDrawStatic(captchaResponseCtx, captchaResponseCanvas, samples);

    // Compare patterns
    const score = compareTapPatterns(captchaChallengeSamples, CAPTCHA_SAMPLE_RATE,
                                     captchaResponseSamples, sr);
    displayCaptchaResult(score);
}

function captchaDrawLive() {
    if (!captchaAnalyserNode) return;
    const bufLen = captchaAnalyserNode.frequencyBinCount;
    const data   = new Uint8Array(bufLen);

    function frame() {
        if (!captchaRecording) return;
        captchaAnimFrame = requestAnimationFrame(frame);

        captchaAnalyserNode.getByteTimeDomainData(data);
        captchaResponseCtx.fillStyle = '#e8d6a8';
        captchaResponseCtx.fillRect(0, 0, captchaResponseCanvas.width, captchaResponseCanvas.height);

        captchaResponseCtx.lineWidth   = 2;
        captchaResponseCtx.strokeStyle = '#7a5c3a';
        captchaResponseCtx.beginPath();

        const sliceW = captchaResponseCanvas.width / bufLen;
        let x = 0;
        for (let i = 0; i < bufLen; i++) {
            const y = (data[i] / 255) * captchaResponseCanvas.height;
            i === 0 ? captchaResponseCtx.moveTo(x, y) : captchaResponseCtx.lineTo(x, y);
            x += sliceW;
        }
        captchaResponseCtx.lineTo(captchaResponseCanvas.width, captchaResponseCanvas.height / 2);
        captchaResponseCtx.stroke();
    }
    frame();
}

// ─── Tap onset detection & comparison ────────────────────

function detectOnsets(samples, sr) {
    // Compute energy envelope in short frames, find peaks
    const frameLen  = Math.floor(0.01 * sr);  // 10ms frames
    const hopLen    = Math.floor(0.005 * sr);  // 5ms hop
    const numFrames = Math.floor((samples.length - frameLen) / hopLen);
    const energy    = new Float32Array(numFrames);

    for (let f = 0; f < numFrames; f++) {
        let sum = 0;
        const start = f * hopLen;
        for (let i = 0; i < frameLen; i++) {
            const s = samples[start + i] || 0;
            sum += s * s;
        }
        energy[f] = Math.sqrt(sum / frameLen); // RMS
    }

    // Adaptive threshold: mean + 2*std of energy
    let mean = 0;
    for (let i = 0; i < numFrames; i++) mean += energy[i];
    mean /= numFrames;

    let variance = 0;
    for (let i = 0; i < numFrames; i++) variance += (energy[i] - mean) ** 2;
    const std = Math.sqrt(variance / numFrames);

    const threshold = mean + 1.5 * std;
    const minGapFrames = Math.floor(0.1 * sr / hopLen); // at least 100ms between onsets

    const onsets = [];
    let lastOnsetFrame = -minGapFrames;

    for (let f = 1; f < numFrames; f++) {
        if (energy[f] > threshold && energy[f] > energy[f - 1] && (f - lastOnsetFrame) >= minGapFrames) {
            onsets.push(f * hopLen / sr); // time in seconds
            lastOnsetFrame = f;
        }
    }

    return onsets;
}

function compareTapPatterns(challengeSamples, challengeSr, responseSamples, responseSr) {
    // Use the known tap times from the generated challenge
    const challengeOnsets = captchaTapTimes;
    const responseOnsets  = detectOnsets(responseSamples, responseSr);

    if (responseOnsets.length === 0) return 0;

    // Compare inter-onset intervals (IOI)
    const challengeIOI = [];
    for (let i = 1; i < challengeOnsets.length; i++) {
        challengeIOI.push(challengeOnsets[i] - challengeOnsets[i - 1]);
    }

    const responseIOI = [];
    for (let i = 1; i < responseOnsets.length; i++) {
        responseIOI.push(responseOnsets[i] - responseOnsets[i - 1]);
    }

    if (challengeIOI.length === 0 || responseIOI.length === 0) {
        // If only 1 tap in either, score based on tap count match
        return challengeOnsets.length === responseOnsets.length ? 85 : 30;
    }

    // Score 1: Number of taps similarity (0-100)
    const tapCountDiff = Math.abs(challengeOnsets.length - responseOnsets.length);
    const tapCountScore = Math.max(0, 100 - tapCountDiff * 25);

    // Score 2: IOI similarity using DTW-like alignment
    // Use simpler approach: align the shorter IOI sequence to the longer one
    const minLen = Math.min(challengeIOI.length, responseIOI.length);
    const maxLen = Math.max(challengeIOI.length, responseIOI.length);

    let ioiScore = 0;
    if (minLen > 0) {
        // Compare corresponding intervals
        const ref = challengeIOI;
        const test = responseIOI;

        let totalError = 0;
        const pairLen = Math.min(ref.length, test.length);
        for (let i = 0; i < pairLen; i++) {
            const relError = Math.abs(ref[i] - test[i]) / ref[i];
            totalError += relError;
        }
        const avgRelError = totalError / pairLen;

        // Convert to score: 0% error → 100, 50%+ error → 0
        ioiScore = Math.max(0, Math.min(100, (1 - avgRelError / 0.5) * 100));

        // Penalise for unmatched intervals
        if (maxLen > minLen) {
            ioiScore *= (minLen / maxLen);
        }
    }

    // Weighted combination: 40% tap count, 60% timing
    const finalScore = 0.4 * tapCountScore + 0.6 * ioiScore;
    return Math.round(Math.max(0, Math.min(100, finalScore)));
}

function displayCaptchaResult(score) {
    const passed = score >= 60;

    captchaScoreDisplay.className = passed ? 'captcha-pass' : 'captcha-fail';
    captchaScoreDisplay.innerHTML =
        `${score}%` +
        `<span class="captcha-label">${passed ? 'captcha passed — human verified' : 'captcha failed — try again'}</span>`;
    captchaResultSection.classList.remove('hidden');
    captchaTimerEl.textContent = '';
}

// ─── WAV encoder for captcha ─────────────────────────────
function captchaSamplesToWav(samples, sampleRate) {
    const numCh    = 1;
    const bps      = 16;
    const byteRate = sampleRate * numCh * (bps / 8);
    const blockAlign = numCh * (bps / 8);
    const dataSize = samples.length * (bps / 8);
    const bufSize  = 44 + dataSize;
    const buf      = new ArrayBuffer(bufSize);
    const v        = new DataView(buf);

    let o = 0;
    const ws  = (s) => { for (let i = 0; i < s.length; i++) v.setUint8(o++, s.charCodeAt(i)); };
    const w32 = (n) => { v.setUint32(o, n, true); o += 4; };
    const w16 = (n) => { v.setUint16(o, n, true); o += 2; };

    ws('RIFF'); w32(bufSize - 8); ws('WAVE');
    ws('fmt '); w32(16); w16(1); w16(numCh);
    w32(sampleRate); w32(byteRate); w16(blockAlign); w16(bps);
    ws('data'); w32(dataSize);

    for (let i = 0; i < samples.length; i++) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        s = s < 0 ? s * 0x8000 : s * 0x7FFF;
        v.setInt16(o, s | 0, true);
        o += 2;
    }

    return new Blob([buf], { type: 'audio/wav' });
}

// ─── Captcha event listeners ─────────────────────────────
btnCaptchaGenerate.addEventListener('click', generateTapPattern);
btnCaptchaPlay.addEventListener('click', playCaptchaChallenge);
btnCaptchaRecord.addEventListener('click', () => {
    toggleCaptchaRecording().catch(err => {
        console.error('captcha recording error:', err);
        captchaTimerEl.textContent = 'error: ' + err.message;
    });
});
btnCaptchaSaveChallenge.addEventListener('click', () => {
    if (!captchaChallengeBlob) return;
    download(captchaChallengeBlob, 'captcha_challenge.wav');
});
btnCaptchaSaveResponse.addEventListener('click', () => {
    if (!captchaResponseBlob) return;
    download(captchaResponseBlob, 'captcha_response.wav');
});
