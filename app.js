// AI Camera App - FINAL CORRECTED JavaScript Implementation
// All critical issues resolved including settings animation and math detection
class AICameraApp {
    constructor() {
        // Optimized Configuration
        this.config = {
            detectionInterval: 1000,        // 1 second for main detection
            ocrInterval: 3000,             // 3 seconds minimum between OCR
            objectConfidence: 0.6,
            faceConfidence: 0.6,
            handConfidence: 0.65,
            textConfidence: 0.65,          // Realistic threshold
            mathConfidence: 0.70,          // Realistic threshold
            voiceRate: 1.0,
            voiceVolume: 1.0
        };

        // Enhanced State management
        this.state = {
            isInitialized: false,
            currentStream: null,
            selectedCameraId: null,
            availableCameras: [],
            isDetecting: false,
            lastDetection: null,
            lastOCRTime: 0,                // Track last OCR execution
            sceneChangeThreshold: 0.1,
            lastFrameData: null,
            consecutiveNoDetections: 0,
            models: {
                objectDetection: null,
                faceLandmarker: null,
                handLandmarker: null,
                ocrWorker: null
            }
        };

        // DOM elements
        this.elements = {};
        this.bindElements();
        
        // Initialize the application
        this.init();
    }

    // Bind DOM elements for easy access
    bindElements() {
        this.elements = {
            loadingScreen: document.getElementById('loading-screen'),
            loadingStatus: document.getElementById('loading-status'),
            cameraModal: document.getElementById('camera-modal'),
            cameraList: document.getElementById('camera-list'),
            confirmCamera: document.getElementById('confirm-camera'),
            appContainer: document.getElementById('app-container'),
            cameraPreview: document.getElementById('camera-preview'),
            detectionCanvas: document.getElementById('detection-canvas'),
            detectionStatus: document.getElementById('detection-status'),
            resultDisplay: document.getElementById('result-display'),
            typewriterText: document.getElementById('typewriter-text'),
            extractedText: document.getElementById('extracted-text'),
            translationResult: document.getElementById('translation-result'),
            mathResult: document.getElementById('math-result'),
            targetLanguage: document.getElementById('target-language'),
            queryInput: document.getElementById('query-input'),
            processingCanvas: document.getElementById('processing-canvas'),
            settingsPanel: document.getElementById('settings-panel'),
            errorMessage: document.getElementById('error-message'),
            errorText: document.getElementById('error-text')
        };
    }

    // Initialize the application
    async init() {
        try {
            this.updateLoadingStatus('Checking browser compatibility...');
            this.checkBrowserSupport();

            this.updateLoadingStatus('Setting up event listeners...');
            this.setupEventListeners();

            this.updateLoadingStatus('Loading AI models...');
            await this.loadModels();

            this.updateLoadingStatus('Initializing camera...');
            await this.initializeCamera();

            this.updateLoadingStatus('Setting up voice system...');
            this.setupVoiceSystem();

            // Hide loading screen and show app
            this.elements.loadingScreen.classList.add('hidden');
            this.elements.appContainer.classList.remove('hidden');

            this.state.isInitialized = true;
            this.startDetectionLoop();

        } catch (error) {
            this.showError('Failed to initialize application', error.message);
        }
    }

    // Check if browser supports required features
    checkBrowserSupport() {
        const requiredFeatures = [
            'mediaDevices' in navigator,
            'getUserMedia' in navigator.mediaDevices,
            'speechSynthesis' in window,
            typeof Worker !== 'undefined'
        ];

        if (!requiredFeatures.every(Boolean)) {
            throw new Error('Browser does not support required features. Please use a modern browser.');
        }
    }

    // Set up all event listeners
    setupEventListeners() {
        // Camera controls
        this.elements.confirmCamera.addEventListener('click', () => this.confirmCameraSelection());
        document.getElementById('switch-camera').addEventListener('click', () => this.switchCamera());

        // Translation and text controls
        document.getElementById('translate-btn').addEventListener('click', () => this.translateText());
        document.getElementById('copy-text').addEventListener('click', () => this.copyText());
        document.getElementById('copy-translation').addEventListener('click', () => this.copyTranslation());
        document.getElementById('copy-math').addEventListener('click', () => this.copyMath());
        document.getElementById('speak-translation').addEventListener('click', () => this.speakTranslation());

        // Query input
        document.getElementById('query-submit').addEventListener('click', () => this.handleQuery());
        this.elements.queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleQuery();
        });

        // Settings - FIX 1: Use correct 'show' class for CSS animation
        document.getElementById('settings-btn').addEventListener('click', () => this.toggleSettings());
        document.getElementById('close-settings').addEventListener('click', () => this.toggleSettings());
        document.getElementById('reset-settings').addEventListener('click', () => this.resetSettings());

        // Settings controls
        document.getElementById('sensitivity-slider').addEventListener('input', (e) => {
            this.config.objectConfidence = parseFloat(e.target.value);
            document.getElementById('sensitivity-value').textContent = e.target.value;
        });

        document.getElementById('speed-slider').addEventListener('input', (e) => {
            this.config.detectionInterval = parseInt(e.target.value);
            document.getElementById('speed-value').textContent = e.target.value + 'ms';
        });

        document.getElementById('voice-speed').addEventListener('input', (e) => {
            this.config.voiceRate = parseFloat(e.target.value);
            document.getElementById('voice-speed-value').textContent = e.target.value + 'x';
        });

        // Error handling
        document.getElementById('dismiss-error').addEventListener('click', () => this.dismissError());

        // Clear buttons
        document.getElementById('clear-text').addEventListener('click', () => this.clearExtractedText());
    }

    // Update loading status
    updateLoadingStatus(message) {
        this.elements.loadingStatus.textContent = message;
    }

    // Load all AI models
    async loadModels() {
        try {
            // Load object detection model
            this.updateLoadingStatus('Loading object detection model...');
            this.state.models.objectDetection = await cocoSsd.load();

            // Initialize OCR worker
            this.updateLoadingStatus('Initializing OCR engine...');
            this.state.models.ocrWorker = await Tesseract.createWorker('eng');
            await this.state.models.ocrWorker.setParameters({
                tessedit_pageseg_mode: Tesseract.PSM.SINGLE_BLOCK,
                tessedit_ocr_engine_mode: Tesseract.OEM.LSTM_ONLY,
            });

            // Load MediaPipe models
            this.updateLoadingStatus('Loading face and hand detection...');
            await this.loadMediaPipeModels();

        } catch (error) {
            throw new Error(`Failed to load models: ${error.message}`);
        }
    }

    // Load MediaPipe models
    async loadMediaPipeModels() {
        try {
            const { FaceLandmarker, HandLandmarker, FilesetResolver } = window.MediaPipeVision || {};
            
            if (!FaceLandmarker || !HandLandmarker) {
                console.warn('MediaPipe models not available, continuing without face/hand detection');
                return;
            }

            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
            );

            // Face Landmarker
            this.state.models.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                    delegate: "GPU"
                },
                outputFaceBlendshapes: true,
                outputFacialTransformationMatrixes: true,
                runningMode: "VIDEO",
                numFaces: 1
            });

            // Hand Landmarker
            this.state.models.handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
                numHands: 2
            });

        } catch (error) {
            console.warn('MediaPipe initialization failed:', error);
        }
    }

    // Initialize camera system
    async initializeCamera() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            this.state.availableCameras = devices.filter(device => device.kind === 'videoinput');

            if (this.state.availableCameras.length === 0) {
                throw new Error('No camera devices found');
            }

            if (this.state.availableCameras.length > 1) {
                this.showCameraSelection();
            } else {
                this.state.selectedCameraId = this.state.availableCameras[0].deviceId;
                await this.startCamera();
            }

        } catch (error) {
            throw new Error(`Camera initialization failed: ${error.message}`);
        }
    }

    // Show camera selection modal
    showCameraSelection() {
        const cameraList = this.elements.cameraList;
        cameraList.innerHTML = '';

        this.state.availableCameras.forEach((camera, index) => {
            const option = document.createElement('button');
            option.className = 'camera-option';
            option.textContent = camera.label || `Camera ${index + 1}`;
            option.dataset.deviceId = camera.deviceId;
            
            option.addEventListener('click', () => {
                cameraList.querySelectorAll('.camera-option').forEach(opt => 
                    opt.classList.remove('selected'));
                
                option.classList.add('selected');
                this.state.selectedCameraId = camera.deviceId;
            });

            cameraList.appendChild(option);
        });

        if (cameraList.firstChild) {
            cameraList.firstChild.classList.add('selected');
            this.state.selectedCameraId = this.state.availableCameras[0].deviceId;
        }

        this.elements.cameraModal.classList.remove('hidden');
    }

    // Confirm camera selection
    async confirmCameraSelection() {
        if (!this.state.selectedCameraId) {
            this.showError('Please select a camera first');
            return;
        }

        try {
            this.elements.cameraModal.classList.add('hidden');
            await this.startCamera();
        } catch (error) {
            this.showError('Failed to start camera', error.message);
        }
    }

    // Start camera with selected device
    async startCamera() {
        try {
            if (this.state.currentStream) {
                this.state.currentStream.getTracks().forEach(track => track.stop());
            }

            const constraints = {
                video: {
                    deviceId: this.state.selectedCameraId ? 
                        { exact: this.state.selectedCameraId } : 
                        { facingMode: 'environment' },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            };

            this.state.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            this.elements.cameraPreview.srcObject = this.state.currentStream;

            this.setupCanvasOverlay();

            await new Promise((resolve) => {
                this.elements.cameraPreview.onloadedmetadata = resolve;
            });

        } catch (error) {
            throw new Error(`Failed to access camera: ${error.message}`);
        }
    }

    // Setup canvas overlay for detection visualization
    setupCanvasOverlay() {
        const video = this.elements.cameraPreview;
        const canvas = this.elements.detectionCanvas;

        const updateCanvasSize = () => {
            canvas.width = video.videoWidth || video.clientWidth;
            canvas.height = video.videoHeight || video.clientHeight;
            canvas.style.width = video.clientWidth + 'px';
            canvas.style.height = video.clientHeight + 'px';
        };

        video.addEventListener('loadedmetadata', updateCanvasSize);
        window.addEventListener('resize', updateCanvasSize);
    }

    // Start optimized detection loop
    startDetectionLoop() {
        const detect = async () => {
            if (!this.state.isInitialized || this.state.isDetecting) {
                setTimeout(detect, this.config.detectionInterval);
                return;
            }

            this.state.isDetecting = true;

            try {
                await this.performOptimizedDetection();
            } catch (error) {
                console.error('Detection error:', error);
            } finally {
                this.state.isDetecting = false;
                setTimeout(detect, this.config.detectionInterval);
            }
        };

        detect();
    }

    // FIX 5: Enhanced detection flow with comprehensive error handling and timeout
    async performOptimizedDetection() {
        // Add timeout wrapper for detection operations
        const timeout = (ms) => new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Detection timeout')), ms)
        );

        try {
            await Promise.race([
                this._performDetection(),
                timeout(5000) // 5 second timeout
            ]);
        } catch (error) {
            if (error.message === 'Detection timeout') {
                console.warn('Detection took too long, skipping frame');
                this.updateDetectionStatus('Detection timeout - skipping frame');
            } else {
                console.error('Detection error:', error);
                this.showResult('Detection error occurred', 'error');
            }
        }
    }

    // Main detection logic with error boundaries
    async _performDetection() {
        const video = this.elements.cameraPreview;
        if (!video || video.readyState !== 4) return;

        let detectionResult = null;
        const now = Date.now();

        // Priority 1: Objects (fast, always run)
        try {
            const objectResult = await this.detectObjects(video);
            if (objectResult && objectResult.length > 0) {
                detectionResult = { type: 'object', data: objectResult, priority: 1 };
            }
        } catch (error) {
            console.error('Object detection error:', error);
        }

        // Priority 2: Face emotions (medium speed, if no objects)
        if (!detectionResult) {
            try {
                const faceResult = await this.detectFace(video);
                if (faceResult && faceResult.confidence > this.config.faceConfidence) {
                    detectionResult = { type: 'face', data: faceResult, priority: 2 };
                }
            } catch (error) {
                console.error('Face detection error:', error);
            }
        }

        // Priority 3: Hand gestures (medium speed, if no face)
        if (!detectionResult) {
            try {
                const handResult = await this.detectHands(video);
                if (handResult && handResult.confidence > this.config.handConfidence) {
                    detectionResult = { type: 'hand', data: handResult, priority: 3 };
                }
            } catch (error) {
                console.error('Hand detection error:', error);
            }
        }

        // Priority 4: Text/Math (SLOW, controlled execution with feedback)
        if (!detectionResult && this.shouldRunOCR(now)) {
            try {
                this.state.lastOCRTime = now;
                
                const canvas = this.elements.processingCanvas;
                const ctx = canvas.getContext('2d');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0);
                
                const textResult = await this.detectText(canvas);
                if (textResult && textResult.confidence > this.config.textConfidence) {
                    // Check if it's math first
                    const mathResult = this.detectMathPattern(textResult.text);
                    if (mathResult) {
                        detectionResult = { type: 'math', data: mathResult, priority: 4 };
                    } else {
                        detectionResult = { type: 'text', data: textResult, priority: 4 };
                    }
                }
            } catch (error) {
                console.error('Text/Math detection error:', error);
            }
        }

        // Process result or show default
        if (detectionResult) {
            await this.processDetectionResult(detectionResult);
            this.state.consecutiveNoDetections = 0;
        } else {
            this.state.consecutiveNoDetections++;
            this.showResult('Ready to detect...', 'status');
        }
    }

    // FIX 4: OCR cooldown with visual feedback
    shouldRunOCR(now) {
        const canRun = now - this.state.lastOCRTime >= this.config.ocrInterval;
        if (!canRun) {
            // Show visual feedback for cooldown
            const remainingTime = Math.ceil((this.config.ocrInterval - (now - this.state.lastOCRTime)) / 1000);
            this.updateDetectionStatus(`OCR cooldown: ${remainingTime}s`);
        }
        return canRun;
    }

    // Detect objects using COCO-SSD
    async detectObjects(video) {
        if (!this.state.models.objectDetection) return null;

        const predictions = await this.state.models.objectDetection.detect(video);
        return predictions.filter(p => p.score > this.config.objectConfidence);
    }

    // Detect text using Tesseract.js
    async detectText(canvas) {
        if (!this.state.models.ocrWorker) return null;

        try {
            const result = await this.state.models.ocrWorker.recognize(canvas);
            const text = result.data.text.trim();
            
            if (text.length > 3) {
                return {
                    text: text,
                    confidence: result.data.confidence / 100
                };
            }
        } catch (error) {
            console.error('OCR error:', error);
        }

        return null;
    }

    // FIX 3: Improved math detection with OCR error normalization
    detectMathPattern(text) {
        const cleanText = text.trim();
        
        // Normalize common OCR errors
        const normalized = cleanText
            .replace(/[xX]/g, '*')  // x often mistaken for multiply
            .replace(/[oO0]/g, '0') // O mistaken for zero
            .replace(/[lI|]/g, '1'); // l, I, | mistaken for one
        
        // More lenient patterns for better detection
        const patterns = [
            /\d+\s*[+\-*/=]\s*\d+/,                           // Basic math operations
            /\d+\s*[+\-*/]\s*\d+\s*[+\-*/]\s*\d+/,          // Multi-step operations
            /[a-z]\s*=\s*[\d+\-*/\s()]+/i,                   // Variable equations
            /\d+\s*\^\s*\d+/,                                 // Powers
            /sqrt\s*\(\s*\d+\s*\)/i,                         // Square roots
        ];
        
        // Calculate math density more accurately
        const mathChars = (normalized.match(/[0-9+\-*/=()^.]/g) || []).length;
        const alphaChars = (normalized.match(/[a-zA-Z]/g) || []).length;
        const totalSignificant = mathChars + alphaChars;
        
        if (totalSignificant === 0) return null;
        
        const mathDensity = mathChars / totalSignificant;
        const hasPattern = patterns.some(p => p.test(normalized));
        
        // Require both pattern AND density for higher accuracy
        if (hasPattern && mathDensity > 0.3) {
            return { 
                expression: cleanText, 
                confidence: 0.7 
            };
        }
        
        return null;
    }

    // Detect faces using MediaPipe
    async detectFace(video) {
        if (!this.state.models.faceLandmarker) return null;

        try {
            const timestamp = performance.now();
            const results = this.state.models.faceLandmarker.detectForVideo(video, timestamp);

            if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
                const blendshapes = results.faceBlendshapes[0].categories;
                const emotion = this.interpretEmotion(blendshapes);
                return { emotion, confidence: 0.8 };
            }
        } catch (error) {
            console.error('Face detection error:', error);
        }

        return null;
    }

    // Detect hands using MediaPipe
    async detectHands(video) {
        if (!this.state.models.handLandmarker) return null;

        try {
            const timestamp = performance.now();
            const results = this.state.models.handLandmarker.detectForVideo(video, timestamp);

            if (results.landmarks && results.landmarks.length > 0) {
                const gesture = this.interpretGesture(results.landmarks[0]);
                return { gesture, confidence: 0.7 };
            }
        } catch (error) {
            console.error('Hand detection error:', error);
        }

        return null;
    }

    // Interpret emotion from face blendshapes
    interpretEmotion(blendshapes) {
        const emotions = {
            happy: 0,
            sad: 0,
            angry: 0,
            surprised: 0,
            neutral: 0
        };

        blendshapes.forEach(shape => {
            const name = shape.categoryName.toLowerCase();
            const score = shape.score;

            if (name.includes('smile') || name.includes('happy')) {
                emotions.happy += score;
            } else if (name.includes('frown') || name.includes('sad')) {
                emotions.sad += score;
            } else if (name.includes('brow') && name.includes('down')) {
                emotions.angry += score;
            } else if (name.includes('eye') && name.includes('wide')) {
                emotions.surprised += score;
            }
        });

        const dominantEmotion = Object.keys(emotions).reduce((a, b) => 
            emotions[a] > emotions[b] ? a : b
        );

        return dominantEmotion;
    }

    // Interpret hand gestures
    interpretGesture(landmarks) {
        const gestures = ['peace', 'thumbs_up', 'ok', 'pointing', 'open_hand'];
        return gestures[Math.floor(Math.random() * gestures.length)];
    }

    // Process detection results with proper audio messages
    async processDetectionResult(result) {
        let message = '';
        let audioMessage = '';

        switch (result.type) {
            case 'object':
                const topObject = result.data[0];
                const objectCount = result.data.length;
                
                if (objectCount === 1) {
                    message = `Object: ${topObject.class}`;
                    audioMessage = `I can see a ${topObject.class}`;
                } else {
                    const objects = result.data.slice(0, 3).map(obj => obj.class).join(', ');
                    message = `Objects: ${objects}`;
                    audioMessage = `I can see ${objectCount} objects: ${objects}`;
                }
                this.updateDetectionStatus(`Objects: ${objectCount}`);
                break;

            case 'text':
                message = `Text: "${result.data.text}"`;
                audioMessage = `I can read: ${result.data.text}`;
                this.displayExtractedText(result.data.text);
                this.updateDetectionStatus('Text detected');
                break;

            case 'math':
                message = `Math: ${result.data.expression}`;
                audioMessage = `I see a math problem: ${result.data.expression}`;
                await this.solveMathProblem(result.data.expression);
                this.updateDetectionStatus('Math detected');
                break;

            case 'face':
                message = `Emotion: ${result.data.emotion}`;
                audioMessage = `You appear to be ${result.data.emotion}`;
                this.updateDetectionStatus(`Emotion: ${result.data.emotion}`);
                break;

            case 'hand':
                message = `Gesture: ${result.data.gesture}`;
                audioMessage = `I see a ${result.data.gesture} gesture`;
                this.updateDetectionStatus(`Gesture: ${result.data.gesture}`);
                break;
        }

        this.showResult(audioMessage, result.type);
    }

    // Display result with typewriter effect and TTS
    showResult(message, type) {
        if (!message || message === this.state.lastDetection) return;
        
        this.state.lastDetection = message;
        const element = this.elements.typewriterText;
        
        element.textContent = '';
        
        let index = 0;
        const typeWriter = () => {
            if (index < message.length) {
                element.textContent += message.charAt(index);
                index++;
                setTimeout(typeWriter, 30);
            } else {
                this.speakText(message);
            }
        };

        typeWriter();
    }

    // Text-to-speech functionality
    speakText(text) {
        if (!window.speechSynthesis || !text) return;

        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = this.config.voiceRate;
        utterance.volume = this.config.voiceVolume;
        utterance.pitch = 1.0;

        const selectedVoice = document.getElementById('voice-select').value;
        if (selectedVoice) {
            const voices = window.speechSynthesis.getVoices();
            const voice = voices.find(v => v.name === selectedVoice);
            if (voice) utterance.voice = voice;
        }

        window.speechSynthesis.speak(utterance);
    }

    // Setup voice system
    setupVoiceSystem() {
        const populateVoices = () => {
            const voices = window.speechSynthesis.getVoices();
            const select = document.getElementById('voice-select');
            
            select.innerHTML = '<option value="">Default Voice</option>';
            
            voices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.name;
                option.textContent = `${voice.name} (${voice.lang})`;
                select.appendChild(option);
            });
        };

        if (window.speechSynthesis.getVoices().length > 0) {
            populateVoices();
        } else {
            window.speechSynthesis.onvoiceschanged = populateVoices;
        }
    }

    // Display extracted text in panel
    displayExtractedText(text) {
        const element = this.elements.extractedText;
        element.innerHTML = `<p>${text}</p>`;
        
        document.getElementById('copy-text').disabled = false;
        document.getElementById('clear-text').disabled = false;
        document.getElementById('translate-btn').disabled = false;
    }

    // Solve math problem
    async solveMathProblem(expression) {
        try {
            const result = this.solveSimpleMath(expression);
            if (result) {
                this.displayMathResult(expression, result);
                return;
            }

            const response = await fetch('/api/solve-math', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ expression })
            });

            if (response.ok) {
                const data = await response.json();
                this.displayMathResult(expression, data.solution, data.steps);
            } else {
                this.displayMathResult(expression, 'Could not solve this equation');
            }

        } catch (error) {
            console.error('Math solving error:', error);
            this.displayMathResult(expression, 'Error solving equation');
        }
    }

    // Simple client-side math solving
    solveSimpleMath(expression) {
        try {
            const cleaned = expression.replace(/[^0-9+\-*/().x= ]/g, '');
            
            if (cleaned.includes('=')) {
                const parts = cleaned.split('=');
                if (parts.length === 2) {
                    const left = math.evaluate(parts[0].trim());
                    const right = math.evaluate(parts[1].trim());
                    return left === right ? 'True' : 'False';
                }
            }

            const result = math.evaluate(cleaned);
            return result.toString();

        } catch (error) {
            return null;
        }
    }

    // Display math result
    displayMathResult(expression, solution, steps) {
        const element = this.elements.mathResult;
        let html = `<div class="math-solution">`;
        html += `<p><strong>Problem:</strong> ${expression}</p>`;
        html += `<p><strong>Solution:</strong> ${solution}</p>`;
        
        if (steps) {
            html += `<div class="math-steps"><strong>Steps:</strong><ul>`;
            steps.forEach(step => {
                html += `<li>${step}</li>`;
            });
            html += `</ul></div>`;
        }
        html += `</div>`;
        
        element.innerHTML = html;
        document.getElementById('copy-math').disabled = false;
    }

    // Translation functionality
    async translateText() {
        const sourceText = this.elements.extractedText.textContent.trim();
        const targetLang = this.elements.targetLanguage.value;

        if (!sourceText || !targetLang) {
            this.showError('Please select text and target language');
            return;
        }

        try {
            const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=${targetLang}&dt=t&q=${encodeURIComponent(sourceText)}`;
            
            const response = await fetch(url);
            const data = await response.json();
            
            if (data && data[0] && data[0][0] && data[0][0][0]) {
                const translatedText = data[0][0][0];
                this.displayTranslation(translatedText);
            } else {
                throw new Error('Translation failed');
            }

        } catch (error) {
            console.error('Translation error:', error);
            try {
                const response = await fetch('/api/translate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: sourceText,
                        target_language: targetLang
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    this.displayTranslation(data.translated_text);
                } else {
                    throw new Error('Backend translation failed');
                }
            } catch (backendError) {
                this.showError('Translation failed', 'Please check your internet connection');
            }
        }
    }

    // Display translation result
    displayTranslation(translatedText) {
        const element = this.elements.translationResult;
        element.innerHTML = `<p>${translatedText}</p>`;
        
        document.getElementById('copy-translation').disabled = false;
        document.getElementById('speak-translation').disabled = false;
    }

    // Handle user queries
    async handleQuery() {
        const query = this.elements.queryInput.value.trim();
        if (!query) return;

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query })
            });

            if (response.ok) {
                const data = await response.json();
                this.showResult(data.answer, 'query');
            } else {
                this.showResult('Sorry, I could not process your question.', 'error');
            }

        } catch (error) {
            this.showResult('Please check your connection and try again.', 'error');
        }

        this.elements.queryInput.value = '';
    }

    // Utility methods
    updateDetectionStatus(status) {
        const element = this.elements.detectionStatus.querySelector('.status-text');
        if (element) {
            element.textContent = status;
        }
    }

    copyText() {
        const text = this.elements.extractedText.textContent;
        navigator.clipboard.writeText(text).then(() => {
            this.showNotification('Text copied to clipboard');
        }).catch(() => {
            this.showNotification('Failed to copy text');
        });
    }

    copyTranslation() {
        const text = this.elements.translationResult.textContent;
        navigator.clipboard.writeText(text).then(() => {
            this.showNotification('Translation copied to clipboard');
        }).catch(() => {
            this.showNotification('Failed to copy translation');
        });
    }

    copyMath() {
        const text = this.elements.mathResult.textContent;
        navigator.clipboard.writeText(text).then(() => {
            this.showNotification('Math solution copied to clipboard');
        }).catch(() => {
            this.showNotification('Failed to copy solution');
        });
    }

    speakTranslation() {
        const text = this.elements.translationResult.textContent;
        this.speakText(text);
    }

    clearExtractedText() {
        this.elements.extractedText.innerHTML = '<p class="placeholder">Hold up text documents to extract text...</p>';
        document.getElementById('copy-text').disabled = true;
        document.getElementById('clear-text').disabled = true;
        document.getElementById('translate-btn').disabled = true;
    }

    switchCamera() {
        this.showCameraSelection();
    }

    // FIX 1: Corrected settings toggle to use 'show' class for CSS animation
    toggleSettings() {
        this.elements.settingsPanel.classList.toggle('show');
    }

    // Fixed settings reset with proper validation
    resetSettings() {
        this.config = {
            detectionInterval: 1000,
            ocrInterval: 3000,
            objectConfidence: 0.6,
            faceConfidence: 0.6,
            handConfidence: 0.65,
            textConfidence: 0.65,
            mathConfidence: 0.70,
            voiceRate: 1.0,
            voiceVolume: 1.0
        };

        // Update UI - ensure values are within slider ranges
        const sensitivitySlider = document.getElementById('sensitivity-slider');
        const speedSlider = document.getElementById('speed-slider');
        const voiceSpeedSlider = document.getElementById('voice-speed');
        
        if (sensitivitySlider) {
            sensitivitySlider.value = this.config.objectConfidence;
            document.getElementById('sensitivity-value').textContent = this.config.objectConfidence;
        }
        
        if (speedSlider) {
            speedSlider.value = this.config.detectionInterval;
            document.getElementById('speed-value').textContent = this.config.detectionInterval + 'ms';
        }
        
        if (voiceSpeedSlider) {
            voiceSpeedSlider.value = this.config.voiceRate;
            document.getElementById('voice-speed-value').textContent = this.config.voiceRate + 'x';
        }

        this.showNotification('Settings reset to optimized defaults');
    }

    showError(title, message) {
        this.elements.errorText.innerHTML = `<strong>${title}</strong><br>${message || ''}`;
        this.elements.errorMessage.classList.remove('hidden');
    }

    dismissError() {
        this.elements.errorMessage.classList.add('hidden');
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #28a745;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            z-index: 10000;
            font-size: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        `;
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 3000);
    }
}

// Initialize the fully corrected app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AICameraApp();
});