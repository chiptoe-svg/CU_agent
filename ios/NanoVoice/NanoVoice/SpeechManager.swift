import AVFoundation
import Speech

/// Manages speech recognition (STT) and text-to-speech (TTS) using Apple's native frameworks.
class SpeechManager: NSObject, ObservableObject {
    @Published var isListening = false
    @Published var isSpeaking = false
    @Published var transcribedText = ""
    @Published var authorizationStatus: SFSpeechRecognizerAuthorizationStatus = .notDetermined

    /// When true, auto-sends after a natural pause in speech.
    @Published var autoSend = false {
        didSet { UserDefaults.standard.set(autoSend, forKey: "autoSend") }
    }

    /// Called when silence is detected and autoSend is on.
    var onAutoSend: (() -> Void)?

    /// How long to wait after the last speech before auto-sending (seconds).
    var silenceThreshold: TimeInterval = 1.8

    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private let audioEngine = AVAudioEngine()
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let synthesizer = AVSpeechSynthesizer()

    private var silenceTimer: Timer?
    private var lastTranscriptionTime: Date?
    private var hasReceivedSpeech = false

    override init() {
        super.init()
        synthesizer.delegate = self
        autoSend = UserDefaults.standard.bool(forKey: "autoSend")
        requestAuthorization()
    }

    func requestAuthorization() {
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            DispatchQueue.main.async {
                self?.authorizationStatus = status
            }
        }
    }

    // MARK: - Speech Recognition (STT)

    func startListening() {
        guard authorizationStatus == .authorized else { return }
        guard !isListening else { return }

        // Stop any ongoing speech
        if isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }

        transcribedText = ""
        hasReceivedSpeech = false
        lastTranscriptionTime = nil

        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("Audio session setup failed: \(error)")
            return
        }

        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest else { return }
        recognitionRequest.shouldReportPartialResults = true

        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self else { return }

            if let result {
                let text = result.bestTranscription.formattedString
                DispatchQueue.main.async {
                    self.transcribedText = text
                    self.hasReceivedSpeech = !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    self.lastTranscriptionTime = Date()
                    self.resetSilenceTimer()
                }

                // If the recognizer signals final (it detected a natural end), auto-send
                if result.isFinal && self.autoSend && self.hasReceivedSpeech {
                    DispatchQueue.main.async {
                        self.triggerAutoSend()
                    }
                    return
                }
            }

            if error != nil || (result?.isFinal ?? false) {
                self.stopListeningInternal()
            }
        }

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }

        do {
            audioEngine.prepare()
            try audioEngine.start()
            DispatchQueue.main.async {
                self.isListening = true
                if self.autoSend {
                    self.startSilenceTimer()
                }
            }
        } catch {
            print("Audio engine start failed: \(error)")
        }
    }

    func stopListening() {
        guard isListening else { return }
        cancelSilenceTimer()
        stopListeningInternal()
    }

    private func stopListeningInternal() {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionRequest = nil
        recognitionTask = nil
        cancelSilenceTimer()

        DispatchQueue.main.async {
            self.isListening = false
        }
    }

    // MARK: - Silence Detection

    private func startSilenceTimer() {
        cancelSilenceTimer()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: 0.3, repeats: true) { [weak self] _ in
            self?.checkSilence()
        }
    }

    private func cancelSilenceTimer() {
        silenceTimer?.invalidate()
        silenceTimer = nil
    }

    private func resetSilenceTimer() {
        // Timer keeps running, checkSilence uses lastTranscriptionTime
    }

    private func checkSilence() {
        guard autoSend, isListening, hasReceivedSpeech else { return }
        guard let lastTime = lastTranscriptionTime else { return }

        let elapsed = Date().timeIntervalSince(lastTime)
        if elapsed >= silenceThreshold {
            triggerAutoSend()
        }
    }

    private func triggerAutoSend() {
        guard isListening, hasReceivedSpeech else { return }
        cancelSilenceTimer()
        stopListeningInternal()
        onAutoSend?()
    }

    // MARK: - Text-to-Speech (TTS)

    func speak(_ text: String) {
        // Stop listening if active
        if isListening {
            stopListening()
        }

        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.pitchMultiplier = 1.0

        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playback, mode: .default)
            try audioSession.setActive(true)
        } catch {
            print("Audio session setup for playback failed: \(error)")
        }

        DispatchQueue.main.async {
            self.isSpeaking = true
        }
        synthesizer.speak(utterance)
    }

    func stopSpeaking() {
        synthesizer.stopSpeaking(at: .immediate)
        DispatchQueue.main.async {
            self.isSpeaking = false
        }
    }
}

// MARK: - AVSpeechSynthesizerDelegate

extension SpeechManager: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        DispatchQueue.main.async {
            self.isSpeaking = false
        }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        DispatchQueue.main.async {
            self.isSpeaking = false
        }
    }
}
