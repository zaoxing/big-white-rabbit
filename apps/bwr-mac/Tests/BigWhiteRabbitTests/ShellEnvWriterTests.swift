import XCTest
@testable import BigWhiteRabbit

final class ShellEnvWriterTests: XCTestCase {
    private var tempHome: URL!
    private var oldPath: String?

    override func setUpWithError() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ShellEnvWriterTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempHome = dir
        oldPath = getenv("PATH").map { String(cString: $0) }
        ShellEnvWriter.homeOverrideForTests = dir
        ShellEnvWriter.shellOverrideForTests = "/bin/zsh"
        ShellEnvWriter.publicBinDirsOverrideForTests = []
        ShellEnvWriter.cliPathPrefsURLOverrideForTests = dir
            .appendingPathComponent("prefs", isDirectory: true)
            .appendingPathComponent("cli-path-prefs.json")
        setenv("PATH", "/usr/bin", 1)
    }

    override func tearDownWithError() throws {
        ShellEnvWriter.homeOverrideForTests = nil
        ShellEnvWriter.shellOverrideForTests = nil
        ShellEnvWriter.publicBinDirsOverrideForTests = nil
        ShellEnvWriter.cliPathPrefsURLOverrideForTests = nil
        if let oldPath {
            setenv("PATH", oldPath, 1)
        }
        if let tempHome {
            try? FileManager.default.removeItem(at: tempHome)
        }
    }

    func testEnsureCLIShimWritesExecutableWrapperWithoutEditingShellFiles() throws {
        let appURL = try makeFakeAppURL()

        let result = try ShellEnvWriter.ensureCLIShim(appBundleURL: appURL)

        let shim = tempHome
            .appendingPathComponent(".bwr", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("bwr")
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: shim.path))
        let shimText = try String(contentsOf: shim, encoding: .utf8)
        XCTAssertTrue(shimText.contains("Contents/MacOS/bwr-cli"))
        XCTAssertTrue(shimText.contains("exec "))

        let zshrc = tempHome.appendingPathComponent(".zshrc")
        XCTAssertFalse(FileManager.default.fileExists(atPath: zshrc.path))
        if case .needsShellPathPrompt = result {
            // Expected: rc edits require an explicit prompt now.
        } else {
            XCTFail("Expected shell PATH prompt when no public bin dir is available")
        }
    }

    /// #2680 — the coordinator probes ~/.bwr/bin/bwr-cluster-python before
    /// anything else. Nothing shipped it, so a peer with the app installed
    /// failed every discovery candidate and was reported as "worker runtime is
    /// not installed".
    func testEnsureCLIShimAlsoPublishesTheClusterInterpreter() throws {
        let appURL = try makeFakeAppURL()

        try ShellEnvWriter.ensureCLIShim(appBundleURL: appURL)

        let shim = tempHome
            .appendingPathComponent(".bwr", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("bwr-cluster-python")
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: shim.path))
        let shimText = try String(contentsOf: shim, encoding: .utf8)
        XCTAssertTrue(shimText.contains("Contents/MacOS/bwr-cluster-python"))
        XCTAssertTrue(shimText.contains("\"$@\""))
        // It must not be the CLI launcher: that one forces -m bwr.cli and can
        // never answer `-c 'import bwr'`.
        XCTAssertFalse(shimText.contains("MacOS/bwr-cli"))
    }

    func testClusterInterpreterShimForwardsArgumentsVerbatim() throws {
        let appURL = try makeFakeAppURL()
        let output = tempHome.appendingPathComponent("cluster-python-args.txt")
        let wrapper = appURL
            .appendingPathComponent("Contents", isDirectory: true)
            .appendingPathComponent("MacOS", isDirectory: true)
            .appendingPathComponent("bwr-cluster-python")
        try """
        #!/bin/sh
        printf "%s" "$*" > \(shellQuote(output.path))
        """.write(to: wrapper, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: wrapper.path
        )

        try ShellEnvWriter.ensureCLIShim(appBundleURL: appURL)

        let shim = tempHome
            .appendingPathComponent(".bwr", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("bwr-cluster-python")
        let process = Process()
        process.executableURL = shim
        process.arguments = ["-c", "import bwr"]
        process.environment = ["HOME": tempHome.path, "PATH": "/usr/bin:/bin"]
        try process.run()
        process.waitUntilExit()

        XCTAssertEqual(process.terminationStatus, 0)
        XCTAssertEqual(
            try String(contentsOf: output, encoding: .utf8),
            "-c import bwr"
        )
    }

    /// An older bundle has no cluster wrapper; installing the CLI shim must
    /// still succeed rather than throwing the whole app launch.
    func testMissingClusterInterpreterDoesNotBlockCLIShimInstall() throws {
        let appURL = try makeFakeAppURL(includeClusterPython: false)

        try ShellEnvWriter.ensureCLIShim(appBundleURL: appURL)

        let bin = tempHome
            .appendingPathComponent(".bwr", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
        XCTAssertTrue(
            FileManager.default.isExecutableFile(
                atPath: bin.appendingPathComponent("bwr").path
            )
        )
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: bin.appendingPathComponent("bwr-cluster-python").path
            )
        )
    }

    func testEnsureCLIShimCreatesPublicSymlinkWhenWritable() throws {
        let publicBin = tempHome.appendingPathComponent("public-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: publicBin, withIntermediateDirectories: true)
        ShellEnvWriter.publicBinDirsOverrideForTests = [publicBin]
        setenv("PATH", "\(publicBin.path):/usr/bin", 1)
        let appURL = try makeFakeAppURL()

        let result = try ShellEnvWriter.ensureCLIShim(appBundleURL: appURL)

        let publicCLI = publicBin.appendingPathComponent("bwr")
        XCTAssertTrue(FileManager.default.fileExists(atPath: publicCLI.path))
        let destination = try FileManager.default.destinationOfSymbolicLink(atPath: publicCLI.path)
        XCTAssertTrue(destination.hasSuffix("/.bwr/bin/bwr"))
        XCTAssertEqual(result, .publicCommandReady(path: publicCLI.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: tempHome.appendingPathComponent(".zshrc").path))
    }

    func testEnsureCLIShimDoesNotOverwriteExistingPublicCommand() throws {
        let publicBin = tempHome.appendingPathComponent("public-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: publicBin, withIntermediateDirectories: true)
        ShellEnvWriter.publicBinDirsOverrideForTests = [publicBin]
        setenv("PATH", "\(publicBin.path):/usr/bin", 1)
        let existing = publicBin.appendingPathComponent("bwr")
        try "#!/bin/sh\n".write(to: existing, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: existing.path)

        let result = try ShellEnvWriter.ensureCLIShim(appBundleURL: try makeFakeAppURL())

        let text = try String(contentsOf: existing, encoding: .utf8)
        XCTAssertEqual(text, "#!/bin/sh\n")
        if case .needsShellPathPrompt(let reason) = result {
            XCTAssertTrue(reason.contains("already exists"))
        } else {
            XCTFail("Expected shell PATH prompt when public command conflicts")
        }
    }

    func testExplicitShellPathExportIsIdempotent() throws {
        try ShellEnvWriter.ensureShellPathExport()
        try ShellEnvWriter.ensureShellPathExport()

        let zshrc = tempHome.appendingPathComponent(".zshrc")
        let rcText = try String(contentsOf: zshrc, encoding: .utf8)
        XCTAssertTrue(rcText.contains("# BigWhiteRabbit: CLI shim path begin"))
        XCTAssertTrue(rcText.contains("$HOME/.bwr/bin"))
        let count = rcText.components(separatedBy: "# BigWhiteRabbit: CLI shim path begin").count - 1
        XCTAssertEqual(count, 1)
    }

    func testShimExportsBootstrapBasePath() throws {
        let appURL = try makeFakeAppURL()
        let output = tempHome.appendingPathComponent("base-path-output.txt")
        let cli = appURL
            .appendingPathComponent("Contents", isDirectory: true)
            .appendingPathComponent("MacOS", isDirectory: true)
            .appendingPathComponent("bwr-cli")
        try """
        #!/bin/sh
        printf "%s" "$BWR_BASE_PATH" > \(shellQuote(output.path))
        """.write(to: cli, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: cli.path)

        try ShellEnvWriter.ensureCLIShim(appBundleURL: appURL)

        let support = tempHome
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent("Application Support", isDirectory: true)
            .appendingPathComponent("BigWhiteRabbit", isDirectory: true)
        try FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        try "/tmp/custom-bwr\n".write(
            to: support.appendingPathComponent("base-path"),
            atomically: true,
            encoding: .utf8
        )

        let shim = tempHome
            .appendingPathComponent(".bwr", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("bwr")
        let process = Process()
        process.executableURL = shim
        process.environment = [
            "HOME": tempHome.path,
            "PATH": "/usr/bin:/bin",
        ]
        try process.run()
        process.waitUntilExit()

        XCTAssertEqual(process.terminationStatus, 0)
        XCTAssertEqual(try String(contentsOf: output, encoding: .utf8), "/tmp/custom-bwr")
    }

    func testEnsureCLIShimSkipsPromptWhenShellExportInstalled() throws {
        try ShellEnvWriter.ensureShellPathExport()

        let result = try ShellEnvWriter.ensureCLIShim(appBundleURL: try makeFakeAppURL())

        let shim = tempHome
            .appendingPathComponent(".bwr", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("bwr")
        XCTAssertEqual(result, .publicCommandReady(path: shim.path))
    }

    func testDismissForeverPreferenceRoundTrips() throws {
        XCTAssertFalse(ShellEnvWriter.shouldSuppressCLIPathPrompt())

        ShellEnvWriter.suppressCLIPathPromptForever()

        XCTAssertTrue(ShellEnvWriter.shouldSuppressCLIPathPrompt())
    }

    private func makeFakeAppURL(includeClusterPython: Bool = true) throws -> URL {
        let appURL = tempHome
            .appendingPathComponent("Apps", isDirectory: true)
            .appendingPathComponent("BigWhiteRabbit.app", isDirectory: true)
        let macOS = appURL
            .appendingPathComponent("Contents", isDirectory: true)
            .appendingPathComponent("MacOS", isDirectory: true)
        try FileManager.default.createDirectory(
            at: macOS,
            withIntermediateDirectories: true
        )
        var names = ["bwr-cli"]
        if includeClusterPython {
            names.append("bwr-cluster-python")
        }
        for name in names {
            let executable = macOS.appendingPathComponent(name)
            try "#!/bin/sh\n".write(to: executable, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o755],
                ofItemAtPath: executable.path
            )
        }
        return appURL
    }

    private func shellQuote(_ value: String) -> String {
        if value.isEmpty { return "''" }
        return "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }
}
