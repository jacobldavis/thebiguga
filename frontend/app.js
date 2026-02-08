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

// ─── DOM refs ────────────────────────────────────────────
const canvas        = document.getElementById('waveform');
const ctx           = canvas.getContext('2d');
const btnRecord     = document.getElementById('btn-record');
const btnWav        = document.getElementById('btn-save-wav');
const btnHash       = document.getElementById('btn-save-hash');
const btnPlayback   = document.getElementById('btn-playback');
const recIndicator  = document.getElementById('rec-indicator');
const timerEl       = document.getElementById('timer');
const resultSection = document.getElementById('result-section');
const hashDisplay   = document.getElementById('hash-display');
const hashLengthEl  = document.getElementById('hash-length');
const bucketCountEl = document.getElementById('bucket-count');
const bucketSizeEl  = document.getElementById('bucket-size');
const sampleRateEl  = document.getElementById('sample-rate');

// ─── Button event listeners ──────────────────────────────
btnRecord.addEventListener('click', () => {
    toggleRecording().catch(err => {
        console.error('toggleRecording error:', err);
        showError('error: ' + err.message);
    });
});
btnWav.addEventListener('click',  () => saveWAV());
btnHash.addEventListener('click', () => saveHash());
btnPlayback.addEventListener('click', () => playRecording());

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
    hashDisplay.textContent   = msg;
    resultSection.classList.remove('hidden');
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
    const targetSamples = MAX_DURATION_S * recSampleRate;

    processorNode.onaudioprocess = (e) => {
        if (!recording) return;
        const buf = new Float32Array(e.inputBuffer.getChannelData(0));
        capturedChunks.push(buf);
        capturedSampleCount += buf.length;

        // Update timer display based on actual captured audio
        const capturedSecs = capturedSampleCount / recSampleRate;
        timerEl.textContent = `${capturedSecs.toFixed(1)}s`;

        // Auto-stop once we've captured enough samples
        if (capturedSampleCount >= targetSamples) stopRecording();
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioCtx.destination); // must connect for events to fire

    recording  = true;
    startTime  = performance.now();
    currentHash    = '';
    currentWavBlob = null;

    // ── UI ──
    btnRecord.textContent = 'stop';
    btnRecord.classList.add('recording');
    recIndicator.classList.remove('hidden');
    resultSection.classList.add('hidden');
    btnWav.disabled  = true;
    btnHash.disabled = true;
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
    btnRecord.textContent = 'record';
    btnRecord.classList.remove('recording');
    recIndicator.classList.add('hidden');

    // ── Process captured audio ──
    processAudio();
}

// ═══════════════════════════════════════════════════════════
//  Audio Processing  →  Send to Backend for Spectral Hash
// ═══════════════════════════════════════════════════════════

async function processAudio() {
    // Merge chunks into one contiguous Float32Array
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

    // Build WAV blob
    currentWavBlob = samplesToWav(samples, recSampleRate);

    // Send WAV to backend for processing
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

        // ── Display results ──
        const bucketLen = Math.floor(samples.length / NUM_BUCKETS);
        hashDisplay.textContent   = currentHash;
        hashLengthEl.textContent  = result.hashLength;
        bucketCountEl.textContent = NUM_BUCKETS;
        bucketSizeEl.textContent  = bucketLen + ' (~' + Math.round(bucketLen / recSampleRate * 1000) + ' ms)';
        sampleRateEl.textContent  = result.sampleRate;
        resultSection.classList.remove('hidden');
        btnWav.disabled  = false;
        btnHash.disabled = false;
        btnPlayback.disabled = false;

        // Draw static waveform of the recording
        drawStaticWaveform(samples);

    } catch (error) {
        console.error('Error processing audio:', error);
        showError('error processing audio: ' + error.message);
        drawIdleLine();
    }
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

function saveHash() {
    if (!currentHash) return;
    const payload = {
        hash:       currentHash,
        hashLength: NUM_BUCKETS,
        sampleRate: recSampleRate,
        fftSize:    FFT_SIZE,
        silentBuckets:  (currentHash.match(/-/g) || []).length,
        activeBuckets:  NUM_BUCKETS - (currentHash.match(/-/g) || []).length,
    };
    const blob = new Blob(
        [JSON.stringify(payload, null, 2)],
        { type: 'application/json' }
    );
    download(blob, 'spectral_hash.json');
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
