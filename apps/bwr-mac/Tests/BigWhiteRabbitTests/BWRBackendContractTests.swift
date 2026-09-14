// Does the bwr backend answer the shapes this app decodes?
//
// `DTOFixtureTests` guards the same DTOs against fixtures captured from an
// oMLX server. Those pass whether or not bwr agrees, because bwr is a
// different backend with its own handlers — so every screen could be broken
// against the server this app actually ships with and the suite would stay
// green. That is not hypothetical: it is how `/admin/api/models` came to
// omit `is_loading` and `estimated_size`, and `/admin/api/stats` to return
// `uptime_s` where `StatsDTO` requires `uptime_seconds` — each one enough to
// make the decode throw and blank the whole screen, since a Swift `Decodable`
// fails the entire value on one missing non-optional key.
//
// The fixtures here are captured from a live `bwr serve --model-dir …`:
//
//   .venv/bin/python -m bwr.cli serve --model-dir ./models --port 1919
//   python3 tools/capture_bwr_fixtures.py
//
// Re-capture after any intentional change to `bwr/server/webui/routes.py`.

import XCTest
@testable import BigWhiteRabbit

final class BWRBackendContractTests: XCTestCase {

    /// Matches BWRClient's decoder exactly; a test that decoded differently
    /// from the client would prove nothing about the client.
    private static func makeDecoder() -> JSONDecoder {
        let dec = JSONDecoder()
        dec.keyDecodingStrategy = .convertFromSnakeCase
        return dec
    }

    private func fixture(_ name: String) throws -> Data {
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
        return try Data(contentsOf: dir.appendingPathComponent("\(name).json"))
    }

    private func decode<T: Decodable>(_ type: T.Type, _ fixtureName: String) throws -> T {
        let data = try fixture(fixtureName)
        do {
            return try Self.makeDecoder().decode(type, from: data)
        } catch let DecodingError.keyNotFound(key, ctx) {
            XCTFail("""
                \(fixtureName).json is missing "\(key.stringValue)", which \
                \(type) requires. The whole response is discarded, so the \
                screen reading it renders nothing. Path: \(ctx.codingPath.map(\.stringValue))
                """)
            throw DecodingError.keyNotFound(key, ctx)
        } catch let DecodingError.typeMismatch(expected, ctx) {
            XCTFail("""
                \(fixtureName).json has the wrong type for \
                \(ctx.codingPath.map(\.stringValue)): \(type) expects \(expected).
                """)
            throw DecodingError.typeMismatch(expected, ctx)
        }
    }

    // MARK: - Status screen + menubar

    func testServerInfoDecodes() throws {
        let info = try decode(ServerInfoDTO.self, "bwr-server-info")
        XCTAssertFalse(info.host.isEmpty)
        XCTAssertGreaterThan(info.port, 0)
    }

    func testStatsDecodes() throws {
        // StatsDTO backs both StatusScreen and the typed menubar poller.
        _ = try decode(StatsDTO.self, "bwr-stats")
    }

    func testDeviceInfoDecodes() throws {
        _ = try decode(DeviceInfoDTO.self, "bwr-device-info")
    }

    // MARK: - Models screen

    func testModelListDecodes() throws {
        let list = try decode(ListModelsResponse.self, "bwr-models")
        XCTAssertFalse(list.models.isEmpty, "fixture captured with no models visible")
        for m in list.models {
            XCTAssertFalse(m.id.isEmpty)
        }
    }

    func testModelSettingsDecodes() throws {
        _ = try decode(ModelSettingsDTO.self, "bwr-model-settings")
    }

    // MARK: - Server screen

    func testGlobalSettingsDecodes() throws {
        _ = try decode(GlobalSettingsDTO.self, "bwr-global-settings")
    }

    // MARK: - Logs screen

    func testLogsDecodes() throws {
        let logs = try decode(LogsDTO.self, "bwr-logs")
        XCTAssertFalse(logs.logFile.isEmpty)
    }

    // MARK: - Usage history

    func testUsageDecodes() throws {
        _ = try decode(UsageHistoryDTO.self, "bwr-usage")
    }
}
