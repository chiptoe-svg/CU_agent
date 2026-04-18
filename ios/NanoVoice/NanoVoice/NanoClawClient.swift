import Foundation

/// A saved NanoClaw server instance.
struct ServerInstance: Codable, Identifiable, Equatable {
    var id: UUID
    var name: String
    var url: String
    var apiKey: String

    init(id: UUID = UUID(), name: String = "", url: String = "http://", apiKey: String = "") {
        self.id = id
        self.name = name
        self.url = url
        self.apiKey = apiKey
    }
}

/// HTTP client for the NanoClaw API channel. Supports multiple server instances.
class NanoClawClient: ObservableObject {
    @Published var isLoading = false
    @Published var servers: [ServerInstance] = []
    @Published var activeServerID: UUID?

    private let serversKey = "savedServers"
    private let activeKey = "activeServerID"

    var activeServer: ServerInstance? {
        servers.first { $0.id == activeServerID }
    }

    var isConfigured: Bool {
        guard let server = activeServer else { return false }
        return !server.url.isEmpty && !server.apiKey.isEmpty
    }

    init() {
        loadServers()
    }

    // MARK: - Persistence

    func loadServers() {
        if let data = UserDefaults.standard.data(forKey: serversKey),
           let decoded = try? JSONDecoder().decode([ServerInstance].self, from: data) {
            servers = decoded
        }
        if let idString = UserDefaults.standard.string(forKey: activeKey),
           let id = UUID(uuidString: idString) {
            activeServerID = id
        }
        // Migrate from old single-server format
        if servers.isEmpty, let oldURL = UserDefaults.standard.string(forKey: "serverURL"),
           let oldKey = UserDefaults.standard.string(forKey: "apiKey"), !oldKey.isEmpty {
            let migrated = ServerInstance(name: "Default", url: oldURL, apiKey: oldKey)
            servers = [migrated]
            activeServerID = migrated.id
            saveServers()
            UserDefaults.standard.removeObject(forKey: "serverURL")
            UserDefaults.standard.removeObject(forKey: "apiKey")
        }
    }

    func saveServers() {
        if let data = try? JSONEncoder().encode(servers) {
            UserDefaults.standard.set(data, forKey: serversKey)
        }
        if let id = activeServerID {
            UserDefaults.standard.set(id.uuidString, forKey: activeKey)
        }
    }

    func addServer(_ server: ServerInstance) {
        servers.append(server)
        if servers.count == 1 {
            activeServerID = server.id
        }
        saveServers()
    }

    func updateServer(_ server: ServerInstance) {
        if let idx = servers.firstIndex(where: { $0.id == server.id }) {
            servers[idx] = server
            saveServers()
        }
    }

    func deleteServer(_ server: ServerInstance) {
        servers.removeAll { $0.id == server.id }
        if activeServerID == server.id {
            activeServerID = servers.first?.id
        }
        saveServers()
    }

    func setActive(_ server: ServerInstance) {
        activeServerID = server.id
        saveServers()
    }

    // MARK: - API

    func sendMessage(_ text: String) async throws -> String {
        guard let server = activeServer else {
            throw ClientError.notConfigured
        }

        let url = URL(string: "\(server.url)/api/message")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(server.apiKey)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 300

        let body: [String: String] = ["text": text]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        await MainActor.run { isLoading = true }
        defer { Task { @MainActor in isLoading = false } }

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw ClientError.invalidResponse
        }

        guard httpResponse.statusCode == 200 else {
            let errorBody = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw ClientError.serverError(statusCode: httpResponse.statusCode, message: errorBody)
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let responseText = json["response"] as? String else {
            throw ClientError.invalidResponse
        }

        return responseText
    }

    enum ClientError: LocalizedError {
        case notConfigured
        case invalidResponse
        case serverError(statusCode: Int, message: String)

        var errorDescription: String? {
            switch self {
            case .notConfigured:
                return "No server selected. Open Settings to add one."
            case .invalidResponse:
                return "Invalid response from server."
            case .serverError(let code, let message):
                return "Server error (\(code)): \(message)"
            }
        }
    }
}
