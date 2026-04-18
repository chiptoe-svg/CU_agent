import SwiftUI

struct Message: Identifiable {
    let id = UUID()
    let text: String
    let isUser: Bool
    let timestamp = Date()
}

struct ContentView: View {
    @StateObject private var client = NanoClawClient()
    @StateObject private var speech = SpeechManager()
    @State private var messages: [Message] = []
    @State private var errorMessage: String?
    @State private var showSettings = false
    @State private var autoSpeak = true

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Conversation history
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(messages) { msg in
                                MessageBubble(message: msg)
                                    .id(msg.id)
                            }

                            if client.isLoading {
                                HStack {
                                    ProgressView()
                                        .scaleEffect(0.8)
                                    Text("Thinking...")
                                        .font(.subheadline)
                                        .foregroundStyle(.secondary)
                                }
                                .padding(.horizontal)
                                .id("loading")
                            }
                        }
                        .padding()
                    }
                    .onChange(of: messages.count) {
                        withAnimation {
                            if let lastMessageID = messages.last?.id {
                                proxy.scrollTo(lastMessageID, anchor: .bottom)
                            } else {
                                proxy.scrollTo("loading", anchor: .bottom)
                            }
                        }
                    }
                }

                // Error banner
                if let error = errorMessage {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.white)
                        .padding(8)
                        .frame(maxWidth: .infinity)
                        .background(.red.opacity(0.8))
                        .onTapGesture { errorMessage = nil }
                }

                // Live transcription
                if speech.isListening && !speech.transcribedText.isEmpty {
                    Text(speech.transcribedText)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal)
                        .padding(.vertical, 4)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.ultraThinMaterial)
                }

                Divider()

                // Controls
                HStack(spacing: 20) {
                    // Left column: toggles
                    VStack(spacing: 8) {
                        // Auto-speak toggle
                        Button {
                            autoSpeak.toggle()
                        } label: {
                            Image(systemName: autoSpeak ? "speaker.wave.2.fill" : "speaker.slash.fill")
                                .font(.callout)
                                .foregroundStyle(autoSpeak ? .blue : .secondary)
                        }

                        // Auto-send toggle (sends on natural pause)
                        Button {
                            speech.autoSend.toggle()
                        } label: {
                            Image(systemName: speech.autoSend ? "hand.raised.slash.fill" : "hand.raised.fill")
                                .font(.callout)
                                .foregroundStyle(speech.autoSend ? .green : .secondary)
                        }
                    }
                    .frame(width: 36)

                    Spacer()

                    // Mic button
                    Button {
                        if speech.isListening {
                            speech.stopListening()
                            sendTranscribedText()
                        } else if speech.isSpeaking {
                            speech.stopSpeaking()
                        } else {
                            speech.startListening()
                        }
                    } label: {
                        ZStack {
                            Circle()
                                .fill(micButtonColor)
                                .frame(width: 72, height: 72)
                                .shadow(color: micButtonColor.opacity(0.4), radius: speech.isListening ? 12 : 4)
                                .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: speech.isListening)

                            Image(systemName: micButtonIcon)
                                .font(.title)
                                .foregroundStyle(.white)
                        }
                    }
                    .disabled(!client.isConfigured || client.isLoading || speech.authorizationStatus != .authorized)

                    Spacer()

                    // Stop speaking
                    Button {
                        speech.stopSpeaking()
                    } label: {
                        Image(systemName: "stop.circle")
                            .font(.title3)
                            .foregroundStyle(speech.isSpeaking ? .red : .secondary)
                    }
                    .disabled(!speech.isSpeaking)
                    .frame(width: 36)
                }
                .padding()
                .background(.ultraThinMaterial)

                // Mode indicator
                if speech.autoSend {
                    Text("Hands-free: sends automatically after you pause")
                        .font(.caption2)
                        .foregroundStyle(.green)
                        .padding(.bottom, 4)
                }
            }
            .navigationTitle(client.activeServer?.name ?? "NanoVoice")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    // Server picker — tap to switch between instances
                    if client.servers.count > 1 {
                        Menu {
                            ForEach(client.servers) { server in
                                Button {
                                    client.setActive(server)
                                    messages.removeAll()
                                } label: {
                                    HStack {
                                        Text(server.name)
                                        if server.id == client.activeServerID {
                                            Image(systemName: "checkmark")
                                        }
                                    }
                                }
                            }
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: "server.rack")
                                Image(systemName: "chevron.down")
                                    .font(.caption2)
                            }
                        }
                    } else {
                        Button {
                            messages.removeAll()
                        } label: {
                            Image(systemName: "trash")
                        }
                        .disabled(messages.isEmpty)
                    }
                }

                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: 12) {
                        if client.servers.count > 1 {
                            Button {
                                messages.removeAll()
                            } label: {
                                Image(systemName: "trash")
                            }
                            .disabled(messages.isEmpty)
                        }

                        Button {
                            showSettings = true
                        } label: {
                            Image(systemName: "gear")
                        }
                    }
                }
            }
            .sheet(isPresented: $showSettings) {
                SettingsView(client: client)
            }
            .onAppear {
                if !client.isConfigured {
                    showSettings = true
                }
                speech.onAutoSend = { [self] in
                    sendTranscribedText()
                }
            }
        }
    }

    private var micButtonColor: Color {
        if speech.isListening { return .red }
        if speech.isSpeaking { return .orange }
        if client.isLoading { return .gray }
        return .blue
    }

    private var micButtonIcon: String {
        if speech.isListening { return "mic.fill" }
        if speech.isSpeaking { return "waveform" }
        return "mic"
    }

    private func sendTranscribedText() {
        let text = speech.transcribedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        messages.append(Message(text: text, isUser: true))
        errorMessage = nil

        Task {
            do {
                let response = try await client.sendMessage(text)
                messages.append(Message(text: response, isUser: false))
                if autoSpeak {
                    speech.speak(response)
                }
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: Message

    var body: some View {
        HStack {
            if message.isUser { Spacer(minLength: 60) }

            Text(message.text)
                .padding(12)
                .background(message.isUser ? Color.blue : Color(.systemGray5))
                .foregroundStyle(message.isUser ? .white : .primary)
                .clipShape(RoundedRectangle(cornerRadius: 16))

            if !message.isUser { Spacer(minLength: 60) }
        }
    }
}

// MARK: - Settings

struct SettingsView: View {
    @ObservedObject var client: NanoClawClient
    @Environment(\.dismiss) private var dismiss
    @State private var editingServer: ServerInstance?
    @State private var showAddSheet = false

    var body: some View {
        NavigationStack {
            List {
                Section("Servers") {
                    ForEach(client.servers) { server in
                        Button {
                            client.setActive(server)
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(server.name)
                                        .font(.headline)
                                        .foregroundStyle(.primary)
                                    Text(server.url)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }

                                Spacer()

                                if server.id == client.activeServerID {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(.blue)
                                }
                            }
                        }
                        .swipeActions(edge: .trailing) {
                            Button(role: .destructive) {
                                client.deleteServer(server)
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }

                            Button {
                                editingServer = server
                            } label: {
                                Label("Edit", systemImage: "pencil")
                            }
                            .tint(.orange)
                        }
                    }

                    Button {
                        showAddSheet = true
                    } label: {
                        Label("Add Server", systemImage: "plus.circle")
                    }
                }

                Section("About") {
                    Text("NanoVoice connects to NanoClaw agents via the HTTP API channel. Each server needs a URL and API key from the server's .env file (HTTP_API_KEY).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .sheet(isPresented: $showAddSheet) {
                ServerEditView(client: client, server: ServerInstance())
            }
            .sheet(item: $editingServer) { server in
                ServerEditView(client: client, server: server, isEditing: true)
            }
        }
    }
}

// MARK: - Server Edit

struct ServerEditView: View {
    @ObservedObject var client: NanoClawClient
    @Environment(\.dismiss) private var dismiss
    @State var server: ServerInstance
    var isEditing = false
    @State private var testResult: String?
    @State private var isTesting = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Server Details") {
                    TextField("Name (e.g., Home, Office, Lab)", text: $server.name)
                        .autocapitalization(.words)

                    TextField("URL (e.g., http://192.168.1.50:3100)", text: $server.url)
                        .textContentType(.URL)
                        .autocapitalization(.none)
                        .keyboardType(.URL)

                    SecureField("API Key", text: $server.apiKey)
                        .autocapitalization(.none)
                }

                Section {
                    Button {
                        testConnection()
                    } label: {
                        HStack {
                            Text("Test Connection")
                            Spacer()
                            if isTesting {
                                ProgressView()
                                    .scaleEffect(0.8)
                            } else if let result = testResult {
                                Image(systemName: result == "ok" ? "checkmark.circle.fill" : "xmark.circle.fill")
                                    .foregroundStyle(result == "ok" ? .green : .red)
                            }
                        }
                    }
                    .disabled(server.url.isEmpty || server.apiKey.isEmpty || isTesting)

                    if let result = testResult, result != "ok" {
                        Text(result)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }
            }
            .navigationTitle(isEditing ? "Edit Server" : "Add Server")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Save") {
                        if server.name.isEmpty {
                            server.name = "Server"
                        }
                        if isEditing {
                            client.updateServer(server)
                        } else {
                            client.addServer(server)
                        }
                        dismiss()
                    }
                    .disabled(server.url.isEmpty || server.apiKey.isEmpty)
                }
            }
        }
    }

    private func testConnection() {
        isTesting = true
        testResult = nil

        Task {
            do {
                let url = URL(string: "\(server.url)/api/message")!
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.setValue("Bearer \(server.apiKey)", forHTTPHeaderField: "Authorization")
                request.timeoutInterval = 30

                let body: [String: String] = ["text": "ping"]
                request.httpBody = try JSONSerialization.data(withJSONObject: body)

                let (_, response) = try await URLSession.shared.data(for: request)
                let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0

                await MainActor.run {
                    isTesting = false
                    testResult = statusCode == 200 ? "ok" : "HTTP \(statusCode)"
                }
            } catch {
                await MainActor.run {
                    isTesting = false
                    testResult = error.localizedDescription
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
