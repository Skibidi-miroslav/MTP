import Foundation

/// Маленькая GRU-языковая модель (~150 тыс. параметров), полностью офлайн, без зависимостей.
/// Файлы model.bin и meta.json нужно добавить в Xcode-проект (Copy Bundle Resources).
final class MiniChatModel {
    private let E: Int, H: Int, V: Int
    private let chars: [String]
    private var stoi: [String: Int] = [:]
    private let emb: [Float], wx: [Float], wh: [Float], b: [Float], wo: [Float], bo: [Float]

    init?() {
        guard let metaURL = Bundle.main.url(forResource: "meta", withExtension: "json"),
              let binURL = Bundle.main.url(forResource: "model", withExtension: "bin"),
              let metaData = try? Data(contentsOf: metaURL),
              let meta = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any],
              let cs = meta["chars"] as? [String],
              let e = meta["embed"] as? Int, let h = meta["hidden"] as? Int, let v = meta["vocab"] as? Int,
              let blob = try? Data(contentsOf: binURL) else { return nil }
        let floats: [Float] = blob.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        var o = 0
        func take(_ n: Int) -> [Float] { defer { o += n }; return Array(floats[o..<o + n]) }
        E = e; H = h; V = v; chars = cs
        emb = take(v * e); wx = take(e * 3 * h); wh = take(h * 3 * h)
        b = take(3 * h); wo = take(h * v); bo = take(v)
        for (i, c) in cs.enumerated() { stoi[c] = i }
    }

    private func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }

    private func step(_ idx: Int, _ h: [Float]) -> [Float] {
        let G = 3 * H
        var gx = b
        let base = idx * E
        for i in 0..<E {
            let xi = emb[base + i], row = i * G
            for j in 0..<G { gx[j] += xi * wx[row + j] }
        }
        var gh = [Float](repeating: 0, count: G)
        for k in 0..<H {
            let hk = h[k], row = k * G
            for j in 0..<G { gh[j] += hk * wh[row + j] }
        }
        var out = [Float](repeating: 0, count: H)
        for j in 0..<H {
            let z = sigmoid(gx[j] + gh[j])
            let r = sigmoid(gx[H + j] + gh[H + j])
            let n = tanh(gx[2 * H + j] + r * gh[2 * H + j])
            out[j] = (1 - z) * n + z * h[j]
        }
        return out
    }

    func reply(to text: String, temperature: Float = 0.5, maxLength: Int = 160) -> String {
        let cleaned = text.lowercased().replacingOccurrences(of: "ё", with: "е")
            .map { stoi[String($0)] != nil ? String($0) : " " }.joined()
        var h = [Float](repeating: 0, count: H)
        for c in ("> " + cleaned + "\n< ") { h = step(stoi[String(c)] ?? 0, h) }
        var out = ""
        for _ in 0..<maxLength {
            var lg = bo
            for k in 0..<H { let hk = h[k], row = k * V; for v in 0..<V { lg[v] += hk * wo[row + v] } }
            let m = lg.max() ?? 0
            var p = lg.map { exp(($0 - m) / temperature) }
            let s = p.reduce(0, +); p = p.map { $0 / s }
            var r = Float.random(in: 0..<1), pick = V - 1
            for v in 0..<V { r -= p[v]; if r <= 0 { pick = v; break } }
            let ch = chars[pick]
            if ch == "\n" { break }
            out += ch
            h = step(pick, h)
        }
        return out
    }
}
