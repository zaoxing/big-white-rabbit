// ServerScreenVM.storageDiff is what gates the Apply button and the actual
// migration. The risk is false-positive diffs (claiming a change when there
// isn't one) that would bounce the server for an idempotent click — or
// false-negative diffs that silently swallow user edits.
//
// All cases below feed the diff text fields that look textually different
// from `services.config` but are semantically equivalent — and assert no
// diff is reported.

import XCTest
@testable import BigWhiteRabbit

@MainActor
final class ServerScreenVMStorageDiffTests: XCTestCase {

    private func makeServices(basePath: String, modelDirs: [String]) -> AppServices {
        let cfg = AppConfig(
            bindAddress: "127.0.0.1",
            port: 8080,
            apiKey: nil,
            basePath: basePath,
            modelDir: modelDirs[0],
            modelDirs: modelDirs,
            hfEndpoint: ""
        )
        return AppServices(config: cfg, server: nil)
    }

    private func makeServices(basePath: String, modelDir: String) -> AppServices {
        makeServices(basePath: basePath, modelDirs: [modelDir])
    }

    func testNoChanges() {
        let services = makeServices(basePath: "/Users/Fido/.bwr",
                                    modelDir: "/Users/Fido/.bwr/models")
        let vm = ServerScreenVM()
        vm.basePathText = "/Users/Fido/.bwr"
        vm.modelDirTexts = ["/Users/Fido/.bwr/models"]

        let diff = vm.storageDiff(services: services)
        XCTAssertFalse(diff.baseChanged)
        XCTAssertFalse(diff.modelDirsChanged)
        XCTAssertFalse(diff.hasChanges)
    }

    func testBaseChangedOnly() {
        let services = makeServices(basePath: "/Users/Fido/.bwr",
                                    modelDir: "/Users/Fido/.bwr/models")
        let vm = ServerScreenVM()
        vm.basePathText = "/Users/Fido/.bwr-other"
        vm.modelDirTexts = ["/Users/Fido/.bwr/models"]

        let diff = vm.storageDiff(services: services)
        XCTAssertTrue(diff.baseChanged)
        XCTAssertFalse(diff.modelDirsChanged)
        XCTAssertEqual(diff.normalizedBase, "/Users/Fido/.bwr-other")
    }

    func testModelDirChangedOnly() {
        let services = makeServices(basePath: "/Users/Fido/.bwr",
                                    modelDir: "/Users/Fido/.bwr/models")
        let vm = ServerScreenVM()
        vm.basePathText = "/Users/Fido/.bwr"
        vm.modelDirTexts = ["/Volumes/SSD/models"]

        let diff = vm.storageDiff(services: services)
        XCTAssertFalse(diff.baseChanged)
        XCTAssertTrue(diff.modelDirsChanged)
        XCTAssertEqual(diff.normalizedModelDir, "/Volumes/SSD/models")
        XCTAssertEqual(diff.normalizedModelDirs, ["/Volumes/SSD/models"])
    }

    func testBothChanged() {
        let services = makeServices(basePath: "/Users/Fido/.bwr",
                                    modelDir: "/Users/Fido/.bwr/models")
        let vm = ServerScreenVM()
        vm.basePathText = "/Users/Fido/.bwr-other"
        vm.modelDirTexts = ["/Volumes/SSD/models"]

        XCTAssertTrue(vm.storageDiff(services: services).hasChanges)
    }

    func testTrailingSlashNormalizesToNoDiff() {
        // standardizingPath strips the trailing slash. Typing it in the
        // field must not flip the Apply button into "pending".
        let services = makeServices(basePath: "/Users/Fido/.bwr",
                                    modelDir: "/Users/Fido/.bwr/models")
        let vm = ServerScreenVM()
        vm.basePathText = "/Users/Fido/.bwr/"
        vm.modelDirTexts = ["/Users/Fido/.bwr/models/"]

        let diff = vm.storageDiff(services: services)
        XCTAssertFalse(diff.baseChanged)
        XCTAssertFalse(diff.modelDirsChanged)
    }

    func testWhitespaceNormalizesToNoDiff() {
        let services = makeServices(basePath: "/Users/Fido/.bwr",
                                    modelDir: "/Users/Fido/.bwr/models")
        let vm = ServerScreenVM()
        vm.basePathText = "  /Users/Fido/.bwr  "
        vm.modelDirTexts = ["\n/Users/Fido/.bwr/models\t"]

        let diff = vm.storageDiff(services: services)
        XCTAssertFalse(diff.baseChanged)
        XCTAssertFalse(diff.modelDirsChanged)
    }

    func testTildeExpansion() {
        let home = NSHomeDirectory()
        let services = makeServices(basePath: "\(home)/.bwr",
                                    modelDir: "\(home)/.bwr/models")
        let vm = ServerScreenVM()
        vm.basePathText = "~/.bwr"
        vm.modelDirTexts = ["~/.bwr/models"]

        let diff = vm.storageDiff(services: services)
        XCTAssertFalse(diff.baseChanged,
                       "tilde must expand before comparing to the home-absolute config value")
        XCTAssertFalse(diff.modelDirsChanged)
    }

    func testEmptyModelDirsTriggersInvalidChange() {
        // Clearing every row is a real edit, but applyServerSettings rejects
        // it before sending a patch because the server requires at least one
        // model root.
        let services = makeServices(basePath: "/Users/Fido/.bwr",
                                    modelDir: "/Users/Fido/.bwr/models")
        let vm = ServerScreenVM()
        vm.basePathText = ""
        vm.modelDirTexts = [""]

        let diff = vm.storageDiff(services: services)
        XCTAssertFalse(diff.baseChanged)
        XCTAssertTrue(diff.modelDirsChanged)
        XCTAssertEqual(diff.normalizedModelDirs, [])
    }

    func testMultipleModelDirsNormalizeToNoDiff() {
        let services = makeServices(
            basePath: "/Users/Fido/.bwr",
            modelDirs: ["/Users/Fido/.bwr/models", "/Volumes/SSD/models"]
        )
        let vm = ServerScreenVM()
        vm.basePathText = "/Users/Fido/.bwr"
        vm.modelDirTexts = [
            " /Users/Fido/.bwr/models/ ",
            "/Volumes/SSD/models",
            "/Volumes/SSD/models/"
        ]

        let diff = vm.storageDiff(services: services)
        XCTAssertFalse(diff.modelDirsChanged)
        XCTAssertEqual(diff.normalizedModelDirs, [
            "/Users/Fido/.bwr/models",
            "/Volumes/SSD/models"
        ])
    }

    func testModelDirReorderTriggersChange() {
        let services = makeServices(
            basePath: "/Users/Fido/.bwr",
            modelDirs: ["/Users/Fido/.bwr/models", "/Volumes/SSD/models"]
        )
        let vm = ServerScreenVM()
        vm.basePathText = "/Users/Fido/.bwr"
        vm.modelDirTexts = ["/Volumes/SSD/models", "/Users/Fido/.bwr/models"]

        let diff = vm.storageDiff(services: services)
        XCTAssertTrue(diff.modelDirsChanged)
        XCTAssertEqual(diff.normalizedModelDir, "/Volumes/SSD/models")
    }

    func testApplyConfigKeepsIPv4WildcardButUsesLoopbackEndpoint() {
        let cfg = AppConfig(
            bindAddress: "0.0.0.0",
            port: 8080,
            apiKey: nil,
            basePath: "/Users/Fido/.bwr",
            modelDir: "/Users/Fido/.bwr/models",
            modelDirs: ["/Users/Fido/.bwr/models"],
            hfEndpoint: ""
        )
        let vm = ServerScreenVM()

        vm.applyConfig(cfg)

        XCTAssertEqual(vm.host, "0.0.0.0")
        XCTAssertEqual(vm.appliedBindAddress, "0.0.0.0")
        XCTAssertEqual(vm.effectiveHost, "127.0.0.1")
    }

    func testApplyConfigDualStackWildcardUsesLoopbackEndpoint() {
        let cfg = AppConfig(
            bindAddress: "::",
            port: 8080,
            apiKey: nil,
            basePath: "/Users/Fido/.bwr",
            modelDir: "/Users/Fido/.bwr/models",
            modelDirs: ["/Users/Fido/.bwr/models"],
            hfEndpoint: ""
        )
        let vm = ServerScreenVM()

        vm.applyConfig(cfg)

        XCTAssertEqual(vm.host, "::")
        XCTAssertEqual(vm.appliedBindAddress, "::")
        // :: is a dual-stack wildcard: it accepts IPv4 and IPv6, but the app's
        // own client connects over loopback.
        XCTAssertEqual(vm.effectiveHost, "127.0.0.1")
    }

    func testApplyConfigSeedsAutoStartWhenServerIsOffline() {
        let cfg = AppConfig(
            bindAddress: "127.0.0.1",
            port: 8080,
            autoStartOnLaunch: false,
            apiKey: nil,
            basePath: "/Users/Fido/.bwr",
            modelDir: "/Users/Fido/.bwr/models",
            modelDirs: ["/Users/Fido/.bwr/models"],
            hfEndpoint: ""
        )
        let vm = ServerScreenVM()

        vm.applyConfig(cfg)

        XCTAssertFalse(vm.autoStartOnLaunch)
    }

    func testAudioUploadSizeValidation() {
        for valid in ["1B", "100MB", "1.5GB", "1024"] {
            XCTAssertTrue(ServerScreenVM.isValidAudioUploadSize(valid), valid)
        }
        for invalid in ["", "0MB", "-1MB", "100 MB", "infMB", "bogus"] {
            XCTAssertFalse(ServerScreenVM.isValidAudioUploadSize(invalid), invalid)
        }
    }

    func testAudioUploadChangeEnablesApply() {
        let services = makeServices(basePath: "/Users/Fido/.bwr",
                                    modelDir: "/Users/Fido/.bwr/models")
        let vm = ServerScreenVM()
        vm.basePathText = "/Users/Fido/.bwr"
        vm.modelDirTexts = ["/Users/Fido/.bwr/models"]
        vm.maxAudioUploadSizeText = "250MB"

        XCTAssertTrue(vm.hasPendingServerChanges(services: services))
    }
}
