"""FurkanSports · Maç Sohbeti Moderasyonu: Gradio uygulaması (R1, R5).

Çalıştır:  python app.py   ->  tarayıcıda http://127.0.0.1:7860

- Solda maç listesi: bir maça tıklanınca resmi yayıncının YouTube özeti ortada oynar
  (video YouTube'dan gömülür, indirilmez).
- Sağda canlı sohbet: her mesaj anında temiz / hakaret olarak etiketlenir, güven yüzdesi
  gösterilir, hakaret içeren mesajlar maskelenir.
- Videonun altında moderasyon paneli: seçili model (varsayılan: BERTurk INT8, ONNX Runtime,
  CPU), her mesajın yanıt süresi, 50 ms bütçe kontrolü ve beş modelin karşılaştırması.
"""
import base64
import html
import inspect
from pathlib import Path

import gradio as gr
import pandas as pd

from src.models_hub import ensure_models
from src.predictors import INT8_NAME, MODEL_NAMES, ModelZoo

BUDGET_MS = 50.0
LOG_COLS = ["#", "Model", "Etiket", "Güven", "Süre (ms)"]
CMP_COLS = ["Model", "Etiket", "Güven", "Süre (ms)"]

# Maç listesi: resmi yayıncıların YouTube'daki maç özetleri (sadece gömülür).
MATCHES = [
    {"name": "Galatasaray - Fenerbahçe", "league": "Süper Lig · 9. Hafta", "yt": "gpWucIHPMEw",
     "channel": "beIN SPORTS Türkiye"},
    {"name": "Galatasaray - Liverpool", "league": "Şampiyonlar Ligi · 2. Hafta", "yt": "vX2xzCdRVss",
     "channel": "TRT SPOR"},
    {"name": "Aston Villa - Fenerbahçe", "league": "Şampiyonlar Ligi · 2. Hafta", "yt": "-f2_WWUMAdc",
     "channel": "TRT SPOR"},
    {"name": "Lille - Galatasaray", "league": "Şampiyonlar Ligi · 3. Hafta", "yt": "trSabcA4How",
     "channel": "TRT SPOR"},
    {"name": "Fenerbahçe - Beşiktaş", "league": "Süper Lig · 4. Hafta", "yt": "H0BfItmwz7k",
     "channel": "beIN SPORTS Türkiye"},
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Saira+Condensed:wght@600;800;900&display=swap');
.gradio-container { max-width: 100% !important; }
#fs-header { display:flex; align-items:center; gap:24px; padding:6px 4px 14px;
  border-bottom:1px solid #1f2a3a; margin-bottom:6px; }
.fs-logo { font-family:'Saira Condensed', Arial, sans-serif; font-weight:900; font-style:italic;
  font-size:52px; line-height:1; letter-spacing:0.5px; text-transform:uppercase;
  transform:skewX(-8deg); color:#ffffff; position:relative; padding-bottom:6px; }
.fs-logo span { color:#FACC15; }
.fs-logo::after { content:""; position:absolute; left:46%; right:-6px; bottom:0; height:5px;
  background:#FACC15; border-radius:4px; transform:skewX(-20deg); }
.fs-ad { margin-left:auto; display:flex; align-items:center; gap:18px; }
.fs-ad img { height:130px; width:auto; object-fit:contain; }
.fs-ad-text { font-family:'Saira Condensed', Arial, sans-serif; font-weight:800; font-style:normal;
  font-size:24px; line-height:1.1; color:#e2e8f0; display:flex; flex-direction:column; }
.fs-ad-text span { font-size:40px; font-weight:900; color:#FACC15; }
.fs-tag { font-family:'Saira Condensed', Arial, sans-serif; font-weight:600; font-size:20px;
  letter-spacing:2px; color:#94a3b8; text-transform:uppercase; }
#fs-sidebar { background:#0f1724; border-radius:12px; padding:0 0 8px; }
.fs-tab { font-family:'Saira Condensed', Arial, sans-serif; font-weight:800; font-size:20px;
  letter-spacing:2px; color:#FACC15; padding:14px 18px 10px; border-bottom:3px solid #FACC15;
  text-transform:uppercase; }
.fs-match button { justify-content:flex-start !important; text-align:left !important;
  background:#111b2b !important; color:#e2e8f0 !important; border:0 !important;
  border-left:4px solid transparent !important; border-radius:0 !important;
  font-weight:700 !important; font-size:16px !important; padding:16px 14px !important;
  white-space:pre-line !important; line-height:1.35 !important; }
.fs-match button:hover { background:#16233a !important; }
.fs-match.fs-active button { background:#13261c !important; border-left-color:#22c55e !important; }
#fs-now { background:#0f1724; border-radius:10px; padding:10px 16px; color:#cbd5e1; font-size:16px; }
#fs-now b { color:#ffffff; }
.fs-dot { display:inline-block; width:10px; height:10px; border-radius:50%; background:#ef4444;
  margin-right:10px; vertical-align:middle; }
.fs-player iframe { width:100%; aspect-ratio:16/9; border:0; border-radius:12px; background:#000; }
.fs-fallback { font-size:14px; color:#94a3b8; padding:8px 4px 0; }
.fs-fallback a { color:#FACC15 !important; font-weight:700; text-decoration:none; margin-left:6px; }
"""


AD_PATH = Path(__file__).resolve().parent / "assets" / "reklam.png"


def _ad_html() -> str:
    """Sağ üstteki reklam kutusu. Görsel repoya eklenmez; yoksa kutu gösterilmez."""
    if not AD_PATH.exists():
        return ""
    b64 = base64.b64encode(AD_PATH.read_bytes()).decode()
    return ('<div class="fs-ad"><img src="data:image/png;base64,' + b64 + '" alt="2 Big Mac menü">'
            '<div class="fs-ad-text">2 TANE BIG MAC PARASI YERİNE<span>FURKANSPORTS</span></div></div>')


def _kw(component, **kwargs):
    """Gradio sürümünün desteklemediği parametreleri sessizce atlar."""
    params = inspect.signature(component).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _player(i: int) -> str:
    vid = MATCHES[i]["yt"]
    # Standart YouTube gömme adresi + referrer: bazı kanallar, oynatıldığı sayfayı görmeden
    # gömülü oynatmaya izin vermiyor.
    src = (f"https://www.youtube.com/embed/{vid}"
           f"?autoplay=1&mute=1&loop=1&playlist={vid}&rel=0&playsinline=1")
    watch = f"https://www.youtube.com/watch?v={vid}"
    return (f'<div class="fs-player"><iframe src="{src}" '
            f'referrerpolicy="strict-origin-when-cross-origin" '
            f'allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen '
            f'title="{html.escape(MATCHES[i]["name"])}"></iframe>'
            f'<div class="fs-fallback">Video burada açılmıyorsa yayıncı gömülü oynatmaya izin '
            f'vermiyor demektir. <a href="{watch}" target="_blank" rel="noopener">'
            f'YouTube’da izle ↗</a></div></div>')


def _now(i: int) -> str:
    m = MATCHES[i]
    return (f'<div id="fs-now"><span class="fs-dot"></span>Şu anda <b>{html.escape(m["name"])}</b>'
            f' izleniyor · {html.escape(m["league"])} · özet: {html.escape(m["channel"])}</div>')


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


def on_pick(i: int):
    """Maç listesinden seçim: oynatıcı, 'şu anda' çubuğu ve aktif satır vurgusu güncellenir."""
    marks = [gr.update(elem_classes=["fs-match", "fs-active"] if j == i else ["fs-match"])
             for j in range(len(MATCHES))]
    return [_player(i), _now(i), *marks]


blocks_kw = _kw(gr.Blocks, title="FurkanSports · Maç Sohbeti Moderasyonu", css=CSS)
launch_kw = {} if "css" in blocks_kw else _kw(gr.Blocks.launch, css=CSS)

with gr.Blocks(**blocks_kw) as demo:
    gr.HTML('<div id="fs-header"><div class="fs-logo">Furkan<span>Sports</span></div>'
            '<div class="fs-tag">Maç özetleri · Moderasyonlu canlı sohbet</div>' + _ad_html() + '</div>')
    log_state = gr.State([])
    with gr.Row(equal_height=False):
        # SOL: maç listesi
        with gr.Column(scale=1, min_width=240, elem_id="fs-sidebar"):
            gr.HTML('<div class="fs-tab">⚽ Futbol</div>')
            match_btns = []
            for i, m in enumerate(MATCHES):
                match_btns.append(gr.Button(
                    f"{m['name']}\n{m['league']}",
                    elem_classes=["fs-match", "fs-active"] if i == 0 else ["fs-match"]))
        # ORTA: yayın + moderasyon paneli
        with gr.Column(scale=3):
            now = gr.HTML(_now(0))
            player = gr.HTML(_player(0))
            with gr.Row():
                model_dd = gr.Dropdown(MODEL_NAMES, value=INT8_NAME, label="Moderasyon modeli",
                                       scale=1)
                stats = gr.Markdown(_stats([]))
            log = gr.Dataframe(**_kw(gr.Dataframe, value=pd.DataFrame(columns=LOG_COLS),
                                     label="Yanıt süresi kaydı", interactive=False,
                                     max_height=240))
            with gr.Accordion("Son mesaj: beş modelin karşılaştırması", open=False):
                cmp = gr.Dataframe(pd.DataFrame(columns=CMP_COLS), interactive=False)
        # SAĞ: canlı sohbet
        with gr.Column(scale=2):
            chat = gr.Chatbot(**_kw(gr.Chatbot, label="Canlı sohbet", height=560,
                                    group_consecutive_messages=False))
            with gr.Row():
                user = gr.Textbox(value="taraftar", label="Kullanıcı adı", scale=1)
                msg = gr.Textbox(placeholder="Mesajını yaz ve Enter'a bas...", label="Mesaj",
                                 scale=3, autofocus=True)
            with gr.Row():
                send = gr.Button("Gönder", variant="primary")
                clear = gr.Button("Sohbeti temizle")

    for i, b in enumerate(match_btns):
        b.click(lambda i=i: on_pick(i), None, [player, now, *match_btns])

    outs = [chat, log_state, log, stats, cmp, msg]
    ins = [msg, user, model_dd, chat, log_state]
    msg.submit(on_send, ins, outs)
    send.click(on_send, ins, outs)
    clear.click(on_clear, None, [chat, log_state, log, stats, cmp])


if __name__ == "__main__":
    demo.launch(inbrowser=True, **launch_kw)
