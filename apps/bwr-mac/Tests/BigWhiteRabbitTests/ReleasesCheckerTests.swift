import XCTest
@testable import BigWhiteRabbit

final class ReleasesCheckerTests: XCTestCase {

    func testCompareVersionsOrdersPrereleaseSuffixes() {
        XCTAssertEqual(
            ReleasesChecker.compareVersions("0.4.0rc2", "0.4.0rc1"),
            .orderedDescending
        )
        XCTAssertEqual(
            ReleasesChecker.compareVersions("0.4.0", "0.4.0rc2"),
            .orderedDescending
        )
        XCTAssertEqual(
            ReleasesChecker.compareVersions("0.4.0rc1", "0.4.0.dev1"),
            .orderedDescending
        )
    }

    func testStableChannelExcludesPrereleases() {
        let selected = ReleasesChecker.selectLatest(
            [
                release("v0.4.0rc2"),
                release("v0.3.12"),
            ],
            channel: .stable
        )

        XCTAssertEqual(selected?.tagName, "v0.3.12")
    }

    func testReleaseCandidateChannelIncludesRCButExcludesDev() {
        let selected = ReleasesChecker.selectLatest(
            [
                release("v0.4.1.dev1"),
                release("v0.4.0rc2"),
                release("v0.4.0rc1"),
            ],
            channel: .releaseCandidate
        )

        XCTAssertEqual(selected?.tagName, "v0.4.0rc2")
    }

    func testDevChannelIncludesDev() {
        let selected = ReleasesChecker.selectLatest(
            [
                release("v0.4.1.dev1"),
                release("v0.4.0rc2"),
                release("v0.4.0"),
            ],
            channel: .dev
        )

        XCTAssertEqual(selected?.tagName, "v0.4.1.dev1")
    }

    func testFindMatchingDMGSupportsMacOSRangeAssets() {
        let sequoia = "BigWhiteRabbit-0.4.4-macos15-sequoia.dmg"
        let tahoeAndNext = "BigWhiteRabbit-0.4.4-macos26-27.dmg"
        let assets = [
            asset(sequoia),
            asset(tahoeAndNext),
        ]

        XCTAssertEqual(
            ReleasesChecker.findMatchingDMG(
                assets: assets,
                macOSMajor: 15
            )?.name,
            sequoia
        )
        XCTAssertEqual(
            ReleasesChecker.findMatchingDMG(
                assets: assets,
                macOSMajor: 26
            )?.name,
            tahoeAndNext
        )
        XCTAssertEqual(
            ReleasesChecker.findMatchingDMG(
                assets: assets,
                macOSMajor: 27
            )?.name,
            tahoeAndNext
        )
        XCTAssertNil(
            ReleasesChecker.findMatchingDMG(
                assets: assets,
                macOSMajor: 28
            )
        )
    }

    func testFindMatchingDMGPrefersExactAssetOverRangeAsset() {
        let range = "BigWhiteRabbit-0.4.4-macos26-27.dmg"
        let exact = "BigWhiteRabbit-0.4.4-macos27-beta.dmg"
        let assets = [
            asset(range),
            asset(exact),
        ]

        XCTAssertEqual(
            ReleasesChecker.findMatchingDMG(
                assets: assets,
                macOSMajor: 27
            )?.name,
            exact
        )
    }

    private func release(
        _ tag: String,
        prerelease: Bool = false,
        draft: Bool = false
    ) -> GitHubRelease {
        GitHubRelease(
            tagName: tag,
            name: tag,
            body: nil,
            htmlURL: URL(string: "https://github.com/zaoxing/big-white-rabbit/releases/tag/\(tag)")!,
            prerelease: prerelease,
            draft: draft,
            assets: []
        )
    }

    private func asset(_ name: String) -> GitHubRelease.Asset {
        GitHubRelease.Asset(
            name: name,
            browserDownloadURL: URL(string: "https://example.com/\(name)")!,
            size: 123
        )
    }
}

@MainActor
final class UpdateControllerPrefsTests: XCTestCase {

    func testLegacyAutoDownloadPrefMigratesToAutoNotify() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("bwr-update-prefs-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = dir.appendingPathComponent("update-prefs.json")
        try Data(
            #"{"channel":"stable","autoCheck":true,"autoDownload":true}"#.utf8
        ).write(to: url)

        let controller = UpdateController(storeURL: url, currentVersion: "0.0.0")
        XCTAssertTrue(controller.autoNotify)

        controller.autoNotify = false

        let saved = try JSONSerialization.jsonObject(
            with: Data(contentsOf: url)
        ) as? [String: Any]
        XCTAssertEqual(saved?["autoNotify"] as? Bool, false)
        XCTAssertNil(saved?["autoDownload"])
    }
}

// MARK: - Update feed ownership

/// The fork's rename pass rewrote the repository NAME but not its OWNER, so
/// the updater pointed at `zaoxing/big-white-rabbit` — upstream's account with our repo
/// name on the end. That URL is not cosmetic: `check()` reads a release's
/// `browser_download_url` from it and `UpdateInstaller` atomically swaps the
/// running .app with what it downloads. A feed in someone else's namespace is
/// therefore an install path we do not control.
///
/// It 404s today, which is the only reason this was inert rather than
/// exploitable. Pinned here so a resync from upstream cannot quietly restore
/// it.
final class ReleasesFeedOwnershipTests: XCTestCase {

    func testUpdateFeedPointsAtThisProjectsOwnRepository() {
        let url = ReleasesChecker.releasesURL
        XCTAssertEqual(url.host, "api.github.com")
        XCTAssertTrue(
            url.path.hasPrefix("/repos/zaoxing/big-white-rabbit/"),
            "Update feed must live in this project's namespace, got \(url.path)"
        )
    }

    func testUpdateFeedIsNotUpstreamsNamespace() {
        XCTAssertFalse(
            ReleasesChecker.releasesURL.absoluteString.contains("jundot"),
            "Updater would install DMGs from the upstream author's account."
        )
    }
}
