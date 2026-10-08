import SwiftUI

struct Message: Identifiable { let id = UUID(); let text: String; let mine: Bool }

struct ContentView: View {
    @State private var messages: [Message] = []
    @State private var input = ""
    private let model = MiniChatModel()

    var body: some View {
        VStack {
            ScrollView {
                VStack(spacing: 8) {
                    ForEach(messages) { m in
                        Text(m.text).padding(10)
                            .background(m.mine ? Color.blue.opacity(0.85) : Color.gray.opacity(0.25))
                            .foregroundColor(m.mine ? .white : .primary)
                            .cornerRadius(14)
                            .frame(maxWidth: .infinity, alignment: m.mine ? .trailing : .leading)
                    }
                }.padding()
            }
            HStack {
                TextField("Напиши что-нибудь…", text: $input).textFieldStyle(.roundedBorder).onSubmit(send)
                Button("Отправить", action: send)
            }.padding()
        }
    }

    private func send() {
        let q = input.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return }
        input = ""; messages.append(Message(text: q, mine: true))
        DispatchQueue.global().async {
            let a = model?.reply(to: q) ?? "модель не загрузилась"
            DispatchQueue.main.async { messages.append(Message(text: a, mine: false)) }
        }
    }
}
