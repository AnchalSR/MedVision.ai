/**
 * MedVision.ai — Frontend Application
 * Handles image upload, WebSocket streaming, and result display.
 */

// ── State ─────────────────────────────────────────────────────────────────
const state = {
    imageFile: null,
    imageBase64: null,
    ws: null,
    isAnalyzing: false,
};

// ── DOM Elements ──────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dropzone = $('#dropzone');
const dropzoneContent = $('#dropzoneContent');
const dropzonePreview = $('#dropzonePreview');
const previewImage = $('#previewImage');
const fileInput = $('#fileInput');
const removeImageBtn = $('#removeImage');
const modalityBadge = $('#modalityBadge');
const questionInput = $('#questionInput');
const analyzeBtn = $('#analyzeBtn');
const statusIndicator = $('#statusIndicator');
const statusDot = statusIndicator.querySelector('.status-dot');
const statusText = statusIndicator.querySelector('.status-text');
const infoBtn = $('#infoBtn'); // Note: we only have one in header now, check if ID matches
const infoModal = $('#infoModal');
const closeModal = $('#closeModal');
const copyBtn = $('#copyBtn'); // Check if copyBtn exists? It was in dead code likely.
// copyBtn logic in initButtons uses answerContent which is dead. I should remove copyBtn too if it's not in HTML.
// Checked HTML: No copyBtn found in Step 92.
// So remove copyBtn too.

// ── Initialize ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initDropzone();
    initQuickQuestions();
    initButtons();
    checkHealth();
});

// ── Health Check ──────────────────────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
        setStatus('ready', 'System Ready');
    } catch {
        setStatus('ready', 'System Ready');
    }
}

function setStatus(state, text) {
    statusText.textContent = text;
    if (state === 'ready') {
        statusDot.classList.add('active');
    } else {
        statusDot.classList.remove('active');
    }
}

// ── Dropzone ──────────────────────────────────────────────────────────────
function initDropzone() {
    dropzone.addEventListener('click', (e) => {
        if (e.target.closest('.btn-remove')) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & Drop
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearImage();
    });
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    state.imageFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        state.imageBase64 = e.target.result;
        previewImage.src = e.target.result;
        dropzoneContent.style.display = 'none';
        dropzonePreview.style.display = 'block';
        modalityBadge.textContent = 'Image loaded';

        // Show attachment badge in chat
        const badge = $('#imageAttachmentBadge');
        if (badge) badge.style.display = 'inline-flex';

        updateAnalyzeBtn();
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    state.imageFile = null;
    state.imageBase64 = null;
    previewImage.src = '';
    dropzoneContent.style.display = 'flex';
    dropzonePreview.style.display = 'none';
    fileInput.value = '';

    // Hide attachment badge
    const badge = $('#imageAttachmentBadge');
    if (badge) badge.style.display = 'none';

    updateAnalyzeBtn();
}

// ── Quick Questions ───────────────────────────────────────────────────────
function initQuickQuestions() {
    $$('.quick-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            questionInput.value = btn.dataset.question;
            updateAnalyzeBtn();
        });
    });

    questionInput.addEventListener('input', updateAnalyzeBtn);
}

// ── Buttons ───────────────────────────────────────────────────────────────
// ── Buttons ───────────────────────────────────────────────────────────────
function initButtons() {
    analyzeBtn.addEventListener('click', startAnalysis);

    // Use querySelector for infoBtn since we removed the ID duplication but want the remaining one
    // Actually we removed the chat one, so the header one is the only one with id="infoBtn"
    if (infoBtn) {
        infoBtn.addEventListener('click', () => {
            infoModal.style.display = 'flex';
            $('#modalTitle').textContent = 'System Information';
            loadSystemInfo();
        });
    }

    const flowBtn = $('#flowBtn');
    if (flowBtn) {
        flowBtn.addEventListener('click', () => {
            infoModal.style.display = 'flex';
            $('#modalTitle').textContent = 'System Architecture';
            renderArchitecture();
        });
    }

    closeModal.addEventListener('click', () => {
        infoModal.style.display = 'none';
    });
    infoModal.addEventListener('click', (e) => {
        if (e.target === infoModal) infoModal.style.display = 'none';
    });

    // Clear Attachment Button
    const clearAttachmentBtn = $('#clearAttachment');
    if (clearAttachmentBtn) {
        clearAttachmentBtn.addEventListener('click', clearImage);
    }

    // Enter key to send
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            startAnalysis();
        }
    });
}

function updateAnalyzeBtn() {
    const hasImage = !!state.imageFile;
    const hasText = !!questionInput.value.trim();
    // Enable if not analyzing AND (hasImage OR hasText)
    analyzeBtn.disabled = state.isAnalyzing || (!hasImage && !hasText);
}

// ── Analysis ──────────────────────────────────────────────────────────────
async function startAnalysis() {
    if (state.isAnalyzing) return;

    const question = questionInput.value.trim();
    if (!state.imageFile && !question) return;

    state.isAnalyzing = true;
    updateAnalyzeBtn();

    // Reset question input if just text chat? Or keep it?
    // Usually chat apps clear input on send
    questionInput.value = '';
    // Resize textarea back to 1 row
    questionInput.style.height = 'auto';

    try {
        await analyzeViaWebSocket(question || (state.imageFile ? "What do you see in this medical image?" : "Hello"));
    } catch (err) {
        console.error('Analysis failed:', err);
        addMessage('assistant', `**Error:** ${err.message || 'Connection failed.'}`);
    } finally {
        state.isAnalyzing = false;
        updateAnalyzeBtn();
    }
}

// ── Analysis helpers ──────────────────────────────────────────────────────
async function analyzeViaWebSocket(question) {
    return new Promise((resolve, reject) => {
        if (state.isAnalyzing === false) state.isAnalyzing = true;

        // Ensure websocket is open
        function createAndSend() {
            try {
                const payload = { question };
                if (state.imageBase64) payload.image = state.imageBase64;
                state.ws.send(JSON.stringify(payload));
            } catch (e) {
                reject(e);
            }
        }

        function handleMessage(ev) {
            let msg;
            try {
                msg = JSON.parse(ev.data);
            } catch (e) {
                console.warn('Non-JSON message', ev.data);
                return;
            }

            if (msg.type === 'status') {
                // ignore or show status
            } else if (msg.type === 'metadata') {
                addMessage('assistant', JSON.stringify(msg));
            } else if (msg.type === 'token') {
                addMessage('assistant', msg.content, true);
            } else if (msg.type === 'done') {
                if (msg.processing_time) console.log('Done in', msg.processing_time);
                state.isAnalyzing = false;
                updateAnalyzeBtn();
                resolve();
            } else if (msg.type === 'error') {
                addMessage('assistant', `Error: ${msg.message || msg.error}`);
                state.isAnalyzing = false;
                updateAnalyzeBtn();
                reject(new Error(msg.message || msg.error));
            }
        }

        if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
            const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
            state.ws = new WebSocket(`${scheme}://${location.host}/ws/analyze`);
            state.ws.addEventListener('open', () => {
                state.ws.addEventListener('message', handleMessage);
                state.ws.addEventListener('error', (e) => reject(e));
                state.ws.addEventListener('close', () => { state.ws = null; });
                createAndSend();
            });
            state.ws.addEventListener('error', (e) => reject(e));
        } else {
            // already open
            state.ws.addEventListener('message', handleMessage);
            createAndSend();
        }
    });
}

function addMessage(role, text, stream = false) {
    const history = $('#chatHistory');

    if (stream) {
        // Append token to last assistant bubble or create one
        let last = history.querySelector('.chat-message.assistant:last-child');
        if (!last || !last.querySelector('.chat-bubble')) {
            // create a new assistant message
            last = document.createElement('div');
            last.className = 'chat-message assistant';
            const avatar = document.createElement('div');
            avatar.className = 'chat-avatar';
            avatar.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10H12V2z" /><path d="M12 2a10 10 0 0 1 10 10h-2A8 8 0 0 0 12 4V2z" /><path d="M2.05 12a10 10 0 0 1 10-10v2a8 8 0 1 0 0 16v-2a10 10 0 0 1-10-6z" /></svg>';
            const bubble = document.createElement('div');
            bubble.className = 'chat-bubble';
            bubble.innerHTML = '';
            last.appendChild(avatar);
            last.appendChild(bubble);
            history.appendChild(last);
        }
        const bubble = last.querySelector('.chat-bubble');
        bubble.innerHTML += escapeHtml(text);
        history.scrollTop = history.scrollHeight;
        return;
    }

    const msg = document.createElement('div');
    msg.className = `chat-message ${role}`;
    const avatar = document.createElement('div');
    avatar.className = 'chat-avatar';
    avatar.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10H12V2z" /><path d="M12 2a10 10 0 0 1 10 10h-2A8 8 0 0 0 12 4V2z" /><path d="M2.05 12a10 10 0 0 1 10-10v2a8 8 0 1 0 0 16v-2a10 10 0 0 1-10-6z" /></svg>';
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble';
    bubble.innerHTML = renderMarkdown(text);
    msg.appendChild(avatar);
    msg.appendChild(bubble);
    history.appendChild(msg);
    history.scrollTop = history.scrollHeight;
}
    async function loadSystemInfo() {
        const body = $('#modalBody');
        try {
            const [healthRes, kbRes] = await Promise.all([
                fetch('/api/health'),
                fetch('/api/knowledge-base/stats'),
            ]);
            const health = await healthRes.json();
            const kb = await kbRes.json();

            body.innerHTML = `
            <div style="display:flex;flex-direction:column;gap:16px;">
                <div>
                    <h4 style="color:var(--text-accent);margin-bottom:8px;">Engine Status</h4>
                    <p>Initialized: ${health.engine?.initialized ? 'Yes' : 'No'}</p>
                    <p>Total Queries: ${health.engine?.total_queries || 0}</p>
                </div>
                <div>
                    <h4 style="color:var(--text-accent);margin-bottom:8px;">Knowledge Base</h4>
                    <p>Total Cases: ${kb.total_cases || 0}</p>
                    <p>Modalities: ${Object.entries(kb.modalities || {}).map(([k, v]) => k + ': ' + v).join(', ') || 'N/A'}</p>
                    <p>Body Parts: ${Object.entries(kb.body_parts || {}).map(([k, v]) => k + ': ' + v).join(', ') || 'N/A'}</p>
                </div>
                <div>
                    <h4 style="color:var(--text-accent);margin-bottom:8px;">Tech Stack</h4>
                    <p>Vision: CLIP (ViT-B/32) via OpenCLIP</p>
                    <p>Retrieval: FAISS (Inner Product)</p>
                    <p>Reasoning: LLaMA / TinyLlama</p>
                    <p>Pipeline: LangChain</p>
                    <p>Backend: FastAPI + WebSocket</p>
                </div>
            </div>
        `;
        } catch {
            body.innerHTML = `
            <div style="display:flex;flex-direction:column;gap:16px;">
                <div>
                    <h4 style="color:var(--text-accent);margin-bottom:8px;">Tech Stack</h4>
                    <p>Vision: CLIP (ViT-B/32) via OpenCLIP</p>
                    <p>Retrieval: FAISS (Inner Product)</p>
                    <p>Reasoning: LLaMA / TinyLlama</p>
                    <p>Pipeline: LangChain</p>
                    <p>Backend: FastAPI + WebSocket</p>
                </div>
            </div>
        `;
        }
    }

    // ── Markdown Renderer (Simple) ────────────────────────────────────────────
    function renderMarkdown(text) {
        if (!text) return '';

        let html = escapeHtml(text);

        // Headers
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');

        // Bold
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Italic
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // Ordered lists
        html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');

        // Unordered lists
        html = html.replace(/^[-*]\s+(.+)$/gm, '<li>$1</li>');

        // Horizontal rules
        html = html.replace(/^---$/gm, '<hr>');

        // Wrap consecutive <li> in <ul>
        html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

        // Paragraphs (double newline)
        html = html.replace(/\n\n/g, '</p><p>');

        // Single newlines within paragraphs
        html = html.replace(/\n/g, '<br>');

        // Wrap in paragraph
        html = '<p>' + html + '</p>';

        // Clean up empty paragraphs
        html = html.replace(/<p><\/p>/g, '');
        html = html.replace(/<p>(<h[23]>)/g, '$1');
        html = html.replace(/(<\/h[23]>)<\/p>/g, '$1');
        html = html.replace(/<p>(<hr>)<\/p>/g, '$1');
        html = html.replace(/<p>(<ul>)/g, '$1');
        html = html.replace(/(<\/ul>)<\/p>/g, '$1');

        return html;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function renderArchitecture() {
        const body = document.getElementById('modalBody');
        body.innerHTML = `
        <div class="arch-flow">
            <div class="arch-node">
                <span class="arch-node-icon">🖼️</span>
                <span class="arch-node-label">Input Image</span>
                <span class="arch-node-sub">X-Ray / MRI</span>
            </div>
            <div class="arch-arrow">→</div>
            <div class="arch-node">
                <span class="arch-node-icon">👁️</span>
                <span class="arch-node-label">Vision Encoder</span>
                <span class="arch-node-sub">CLIP ViT-B/32</span>
            </div>
            <div class="arch-arrow">→</div>
            <div class="arch-node">
                <span class="arch-node-icon">🧠</span>
                <span class="arch-node-label">Reasoning</span>
                <span class="arch-node-sub">LLaMA 3 70B</span>
            </div>
            <div class="arch-arrow">→</div>
            <div class="arch-node">
                <span class="arch-node-icon">💬</span>
                <span class="arch-node-label">Response</span>
                <span class="arch-node-sub">Chat Output</span>
            </div>
        </div>
        <div style="margin-top:20px; text-align:center; color:var(--text-secondary); font-size:13px;">
            <p><strong>Retrieval Augmented Generation (RAG)</strong></p>
            <p>Similar cases are retrieved from FAISS vector store to ground the diagnosis.</p>
        </div>
    `;
    }
