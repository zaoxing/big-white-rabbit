import AppKit
import SwiftUI
import XCTest
@testable import BigWhiteRabbit

final class ThemeTests: XCTestCase {

    func testLightWindowBackgroundUsesStandardWindowColor() {
        let actual = resolvedRGBA(BWRTheme.light.windowBg, appearance: .aqua)
        let expected = resolvedRGBA(Color(nsColor: .windowBackgroundColor),
                                    appearance: .aqua)

        // The theme's light window colour IS the system window colour, and
        // that colour is near-white. No assertion against
        // .underPageBackgroundColor here: on macOS 26 and earlier it was a
        // mid grey in light mode, and on macOS 27 it is near-white too, so
        // its distance from the window colour is a fact about the OS, not
        // about the theme.
        assertClose(actual, expected)
        XCTAssertGreaterThan(actual.red, 0.95)
    }

    func testDarkWindowBackgroundKeepsUnderPageColor() {
        let actual = resolvedRGBA(BWRTheme.dark.windowBg, appearance: .darkAqua)
        let expected = resolvedRGBA(Color(nsColor: .underPageBackgroundColor),
                                    appearance: .darkAqua)

        assertClose(actual, expected)
    }

    func testLightGroupBackgroundIsSubtleGrayWash() {
        let actual = resolvedRGBA(BWRTheme.light.groupBg, appearance: .aqua)

        XCTAssertLessThan(actual.red, 0.01)
        XCTAssertLessThan(actual.green, 0.01)
        XCTAssertLessThan(actual.blue, 0.01)
        XCTAssertGreaterThan(actual.alpha, 0.025)
        XCTAssertLessThan(actual.alpha, 0.05)
    }

    private typealias RGBA = (
        red: CGFloat,
        green: CGFloat,
        blue: CGFloat,
        alpha: CGFloat
    )

    private func resolvedRGBA(_ color: Color, appearance: NSAppearance.Name) -> RGBA {
        let nsColor = NSColor(color)
        var components: RGBA?
        NSAppearance(named: appearance)!.performAsCurrentDrawingAppearance {
            let resolved = nsColor.usingColorSpace(.sRGB)!
            components = (
                red: resolved.redComponent,
                green: resolved.greenComponent,
                blue: resolved.blueComponent,
                alpha: resolved.alphaComponent
            )
        }
        return components!
    }

    private func assertClose(
        _ actual: RGBA,
        _ expected: RGBA,
        accuracy: CGFloat = 0.001,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(
            actual.red,
            expected.red,
            accuracy: accuracy,
            file: file,
            line: line
        )
        XCTAssertEqual(
            actual.green,
            expected.green,
            accuracy: accuracy,
            file: file,
            line: line
        )
        XCTAssertEqual(
            actual.blue,
            expected.blue,
            accuracy: accuracy,
            file: file,
            line: line
        )
        XCTAssertEqual(
            actual.alpha,
            expected.alpha,
            accuracy: accuracy,
            file: file,
            line: line
        )
    }
}
