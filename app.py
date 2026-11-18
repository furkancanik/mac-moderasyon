"""Maç Sohbeti Moderasyonu: Gradio uygulaması (R1, R5).

Çalıştır:  python app.py   ->  tarayıcıda http://127.0.0.1:7860

- Sohbet ekranı: her mesaj anında temiz / hakaret olarak etiketlenir, güven yüzdesi gösterilir,
  hakaret içeren mesajlar maskelenir.
- Yan panel: seçili model (varsayılan: BERTurk INT8, ONNX Runtime, CPU), her mesajın yanıt
  süresi, 50 ms bütçe kontrolü ve son mesaj için beş modelin yan yana tahmini.
"""
import html
import inspect

import gradio as gr
import pandas as pd

from src.models_hub import ensure_models
from src.predictors import INT8_NAME, MODEL_NAMES, ModelZoo

BUDGET_MS = 50.0
LOG_COLS = ["#", "Model", "Etiket", "Güven", "Süre (ms)"]
CMP_COLS = ["Model", "Etiket", "Güven", "Süre (ms)"]

EXAMPLES = [
    "gg güzel maçtı, ikinci yarı çok iyi oynadık",
    "hakem bugün resmen maçı kaçırdı ya",
    "s1kt1r git hakem",
    "bu kadar beceriksiz bir kaleci görmedim, git başka iş yap",
    "salaaaaak mısın sen ne yazıyorsun",
    "Galatasaray - Fenerbahçe derbisi saat kaçta?",
]

ensure_models()
print("Modeller yükleniyor...")
zoo = ModelZoo()
for name in MODEL_NAMES:
    zoo.get(name)
    for _ in range(3):              # ısınma: ilk çağrıların yavaşlığı ölçümlere girmesin
        zoo.predict(name, "maç başlamak üzere, herkese iyi seyirler")
print("Hazır.")

SHORT = {MODEL_NAMES[0]: "BERTurk INT8 (ONNX)", MODEL_NAMES[1]: "BERTurk FP32",
         MODEL_NAMES[2]: "word2vec + CNN", MODEL_NAMES[3]: "tf-idf + LR",
         MODEL_NAMES[4]: "Kural (liste)"}


def _label(p) -> str:
    return "🚫 hakaret" if p.label else "✅ temiz"


def _stats(rows) -> str:
    if not rows:
        return f"Henüz mesaj yok. Gecikme bütçesi: **{BUDGET_MS:.0f} ms / mesaj** (CPU)."
    times = [r[4] for r in rows]
    last = times[-1]
    avg = sum(times) / len(times)
    hidden = sum(1 for r in rows if "hakaret" in r[2])
    ok = "✅ bütçe içinde" if last <= BUDGET_MS else "❌ bütçe aşıldı"
    return (f"**Son mesaj:** {last:.2f} ms ({ok})  \n"
            f"**Ortalama:** {avg:.2f} ms · **Mesaj:** {len(rows)} · **Gizlenen:** {hidden}  \n"
            f"Bütçe: {BUDGET_MS:.0f} ms / mesaj, CPU")


def on_send(text, username, model_name, history, rows):
    text = (text or "").strip()
    if not text:
        return history, rows, pd.DataFrame(rows, columns=LOG_COLS), _stats(rows), \
            gr.update(), ""
    username = (username or "taraftar").strip()
    p = zoo.predict(model_name, text)

    shown = zoo.mask(text) if p.label else html.escape(text)
    badge = f"{_label(p)} · %{100 * p.confidence:.0f} · {p.ms:.1f} ms"
    history = list(history or []) + [
        {"role": "user", "content": f"**{html.escape(username)}:** {shown}  \n<sub>{badge}</sub>"}
    ]

    rows = list(rows or []) + [[len(rows or []) + 1, SHORT[model_name],
                                _label(p), f"%{100 * p.confidence:.0f}", round(p.ms, 2)]]

    cmp = []
    for name in MODEL_NAMES:
        q = p if name == model_name else zoo.predict(name, text)
        cmp.append([SHORT[name], _label(q), f"%{100 * q.confidence:.0f}", round(q.ms, 2)])

    return (history, rows, pd.DataFrame(rows[::-1], columns=LOG_COLS), _stats(rows),
            pd.DataFrame(cmp, columns=CMP_COLS), "")


def on_clear():
    return [], [], pd.DataFrame(columns=LOG_COLS), _stats([]), pd.DataFrame(columns=CMP_COLS)


with gr.Blocks(title="Maç Sohbeti Moderasyonu") as demo:
    gr.Markdown("# ⚽ Maç Sohbeti Moderasyonu\n"
                "Canlı maç yayını sohbetindeki mesajlar CPU üzerinde anlık olarak "
                "**temiz / hakaret** diye etiketlenir; hakaret içeren mesajlar gizlenir.")
    log_state = gr.State([])
    with gr.Row():
        with gr.Column(scale=3):
            chat_kwargs = {"label": "Canlı sohbet", "height": 460}
            if "group_consecutive_messages" in inspect.signature(gr.Chatbot).parameters:
                chat_kwargs["group_consecutive_messages"] = False   # her mesaj ayrı balon
            chat = gr.Chatbot(**chat_kwargs)
            with gr.Row():
                user = gr.Textbox(value="taraftar", label="Kullanıcı adı", scale=1)
                msg = gr.Textbox(placeholder="Mesajını yaz ve Enter'a bas...", label="Mesaj",
                                 scale=4, autofocus=True)
            with gr.Row():
                send = gr.Button("Gönder", variant="primary")
                clear = gr.Button("Sohbeti temizle")
            gr.Examples(EXAMPLES, inputs=msg, label="Örnek mesajlar")
        with gr.Column(scale=2):
            model_dd = gr.Dropdown(MODEL_NAMES, value=INT8_NAME, label="Moderasyon modeli")
            stats = gr.Markdown(_stats([]))
            log = gr.Dataframe(pd.DataFrame(columns=LOG_COLS), label="Yanıt süresi kaydı",
                               interactive=False)
            cmp = gr.Dataframe(pd.DataFrame(columns=CMP_COLS),
                               label="Son mesaj: beş modelin karşılaştırması", interactive=False)
            gr.Markdown("<sub>Veri: OffensEval-TR 2020. Modeller: kural tabanlı, tf-idf + LR, "
                        "kendi word2vec'imiz + 1D CNN, BERTurk (FP32 ve INT8).</sub>")

    outs = [chat, log_state, log, stats, cmp, msg]
    ins = [msg, user, model_dd, chat, log_state]
    msg.submit(on_send, ins, outs)
    send.click(on_send, ins, outs)
    clear.click(on_clear, None, [chat, log_state, log, stats, cmp])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
