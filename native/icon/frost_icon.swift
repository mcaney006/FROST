// FROST app icon: a bold F cut from a hexagon.
//
// Run:   swift native/icon/frost_icon.swift <out.png>
//   or:  swiftc native/icon/frost_icon.swift -o frost_icon && ./frost_icon <out.png>
// Output: 1024x1024 opaque sRGB PNG, full bleed (macOS applies its own icon mask).
//
// One idea: a bold F whose every terminal is cut 30 degrees off square, the way
// hexagon edges meet. The two bars end on 60-degree cuts, the stem bottom on a
// 30-degree cut, and the top-left corner carries a matching 30-degree chamfer, so
// the silhouette is built entirely from hexagon angles (30/60/90/120) and reads
// as a faceted piece of ice and as the letter at once. Two flat colors, one closed
// polygon. No outline, no gradient, no shadow. All geometry is parametric; tune
// the constants below and nothing else.

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

// MARK: - Tunables ------------------------------------------------------------

let canvas: CGFloat = 1024

// Colors (sRGB 0...1). Dark cold ground, pale ice mark. The mark is brighter than
// plain #DDEBF4 so the antialiased edge ring at 16 px stays light instead of muddy
// gray; it keeps a cool tint so it does not read as white. Contrast > 15:1.
let ground = (r: 0x0C / 255.0, g: 0x13 / 255.0, b: 0x20 / 255.0)   // #0C1320
let ice    = (r: 0xE6 / 255.0, g: 0xF2 / 255.0, b: 0xFA / 255.0)   // #E6F2FA

// F metrics, in canvas px at 1024.
let stroke: CGFloat      = 156          // stem and bar thickness (15.2% of canvas, 2.44 px at 16)
let fHeight: CGFloat     = 740          // top of top bar to lowest point of stem (72% of canvas)
let fWidth: CGFloat      = 530          // stem left edge to farthest point of top bar
let midWidth: CGFloat    = 400          // stem left edge to farthest point of mid bar
let midCenterY: CGFloat  = 0.53         // mid bar center as a fraction of fHeight (from top)
let chamfer: CGFloat     = 0.5          // top-left chamfer run as a fraction of stroke (0 = none)
let opticalShiftX: CGFloat = 26         // F is left-heavy (stem + top bar); nudge right so the
                                        // visual centroid sits on the canvas axis
let opticalShiftY: CGFloat = 0

// Optional single thin accent: continue the stem's bottom cut line to the right,
// as if the hexagon edge the F was cut from were still faintly there. 0 = off.
// Off by default: it read as a scratch rather than a facet.
let accentLength: CGFloat = 0
let accentWidth: CGFloat  = 6

// Hex angles. Bars are cut 60 deg from horizontal, stem 30 deg from horizontal:
// each terminal is 30 deg off square, which is how hexagon edges meet.
let tan30 = CGFloat(tan(Double.pi / 6))

// MARK: - Geometry (top-down coordinates, y grows downward) --------------------

let barCutRun  = stroke * tan30       // horizontal offset across a bar's 60-degree end cut
let stemDrop   = stroke * tan30       // vertical offset across the stem's 30-degree bottom cut
let chamferRun = stroke * chamfer
let chamferDrop = chamferRun * tan30

let stemH = fHeight - stemDrop        // stem left edge length (bottom-left corner is the high point)
let midTop = fHeight * midCenterY - stroke / 2
let gap = midTop - stroke             // dark gap between the two bars

let x0 = (canvas - fWidth) / 2 + opticalShiftX
let y0 = (canvas - fHeight) / 2 + opticalShiftY

// Single closed polygon, clockwise from the stem's left edge just below the chamfer.
let f: [CGPoint] = [
    CGPoint(x: x0,                         y: y0 + chamferDrop),
    CGPoint(x: x0 + chamferRun,            y: y0),
    CGPoint(x: x0 + fWidth - barCutRun,    y: y0),
    CGPoint(x: x0 + fWidth,                y: y0 + stroke),
    CGPoint(x: x0 + stroke,                y: y0 + stroke),
    CGPoint(x: x0 + stroke,                y: y0 + midTop),
    CGPoint(x: x0 + midWidth - barCutRun,  y: y0 + midTop),
    CGPoint(x: x0 + midWidth,              y: y0 + midTop + stroke),
    CGPoint(x: x0 + stroke,                y: y0 + midTop + stroke),
    CGPoint(x: x0 + stroke,                y: y0 + fHeight),
    CGPoint(x: x0,                         y: y0 + stemH),
]

// Self-check: the mark must fit with margin and stay legible when scaled to 16 px.
let scale16 = 16 / canvas
precondition(gap * scale16 >= 1.9, "bar gap under 2 px at 16 px: \(gap * scale16)")
precondition(stroke * scale16 >= 2.0, "stroke under 2 px at 16 px")
precondition(fWidth - midWidth >= stroke * 0.6, "top and mid bar too similar in length")
for p in f {
    precondition(p.x >= canvas * 0.08 && p.x <= canvas * 0.92, "mark too close to edge: \(p)")
    precondition(p.y >= canvas * 0.08 && p.y <= canvas * 0.92, "mark too close to edge: \(p)")
}

// MARK: - Render --------------------------------------------------------------

let side = Int(canvas)
let cs = CGColorSpace(name: CGColorSpace.sRGB)!
guard let ctx = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8,
                          bytesPerRow: 0, space: cs,
                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
    fatalError("CGContext")
}
ctx.setShouldAntialias(true)
ctx.setAllowsAntialiasing(true)

// Flip to top-down so the geometry above reads like a design spec.
ctx.translateBy(x: 0, y: canvas)
ctx.scaleBy(x: 1, y: -1)

ctx.setFillColor(red: ground.r, green: ground.g, blue: ground.b, alpha: 1)
ctx.fill(CGRect(x: 0, y: 0, width: canvas, height: canvas))

let path = CGMutablePath()
path.addLines(between: f)
path.closeSubpath()
ctx.setFillColor(red: ice.r, green: ice.g, blue: ice.b, alpha: 1)
ctx.addPath(path)
ctx.fillPath()

if accentLength > 0 {
    // Continue the stem's bottom cut from its low corner, same 30-degree slope.
    let start = CGPoint(x: x0 + stroke, y: y0 + fHeight)
    let dir = CGPoint(x: cos(Double.pi / 6), y: sin(Double.pi / 6))
    let end = CGPoint(x: start.x + dir.x * accentLength, y: start.y + dir.y * accentLength)
    ctx.setStrokeColor(red: ice.r, green: ice.g, blue: ice.b, alpha: 1)
    ctx.setLineWidth(accentWidth)
    ctx.setLineCap(.butt)
    ctx.move(to: CGPoint(x: start.x + dir.x * stroke * 0.35, y: start.y + dir.y * stroke * 0.35))
    ctx.addLine(to: end)
    ctx.strokePath()
}

// MARK: - Write ---------------------------------------------------------------

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write("usage: frost_icon <out.png>\n".data(using: .utf8)!)
    exit(2)
}
let image = ctx.makeImage()!
let outURL = URL(fileURLWithPath: CommandLine.arguments[1])
let dest = CGImageDestinationCreateWithURL(outURL as CFURL, UTType.png.identifier as CFString, 1, nil)!
CGImageDestinationAddImage(dest, image, nil)
precondition(CGImageDestinationFinalize(dest), "PNG write failed")
print("wrote \(outURL.path) \(image.width)x\(image.height) gap16=\(String(format: "%.2f", gap * scale16))px stroke16=\(String(format: "%.2f", stroke * scale16))px stemLeft16=\(String(format: "%.2f", x0 * scale16))px")
