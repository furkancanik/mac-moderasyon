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
from datetime import datetime
import inspect
from pathlib import Path

import gradio as gr

from src.models_hub import ensure_models
from src.predictors import INT8_NAME, MODEL_NAMES, ModelZoo

BUDGET_MS = 50.0

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

/* ---- VAR odası: moderasyon paneli ---- */
#fs-models { background:#0f1724; border:1px solid #1e2d45; border-radius:12px; padding:12px 14px; }
#fs-models .wrap, #fs-models fieldset > div { gap:8px !important; }
#fs-models label { background:#111b2b !important; border:1px solid #24344f !important; border-radius:999px !important;
  padding:8px 14px !important; color:#cbd5e1 !important; font-weight:600 !important; }
#fs-models label.selected, #fs-models label:has(input:checked) { background:#FACC15 !important; border-color:#FACC15 !important;
  color:#0b1220 !important; }
.var { background:linear-gradient(180deg,#0c1628 0%,#0a1220 100%); border:1px solid #1e2d45; border-radius:14px;
  padding:18px 18px 12px; color:#e2e8f0; font-family:'Public Sans', Arial, sans-serif; position:relative; overflow:hidden; }
.var::before { content:""; position:absolute; inset:0; pointer-events:none; opacity:.06;
  background:repeating-linear-gradient(90deg,#22c55e 0 60px,#16a34a 60px 120px); }
.var-head { display:flex; align-items:center; gap:12px; margin-bottom:14px; position:relative; }
.var-badge { font-family:'Saira Condensed', Arial, sans-serif; font-weight:900; font-size:20px; letter-spacing:2px;
  background:#FACC15; color:#0b1220; padding:2px 10px; border-radius:6px; }
.var-title { font-family:'Saira Condensed', Arial, sans-serif; font-weight:800; font-size:22px; letter-spacing:1.5px; }
.var-model { margin-left:auto; font-size:13px; color:#94a3b8; border:1px solid #24344f; border-radius:999px; padding:3px 10px; }
.var-live { font-size:13px; font-weight:700; color:#ef4444; letter-spacing:1px; }
.var-live::before { content:"●"; margin-right:5px; animation:varblink 1.4s infinite; }
@keyframes varblink { 50% { opacity:.25; } }
.var-score { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:16px; position:relative; }
.var-tile { background:#0b1220; border:1px solid #1e2d45; border-radius:10px; padding:10px 12px; }
.var-num { font-family:'Saira Condensed', Arial, sans-serif; font-weight:800; font-size:34px; line-height:1; color:#FACC15;
  font-variant-numeric:tabular-nums; }
.var-num small { font-size:16px; color:#cbd5e1; margin-left:3px; }
.var-lbl { font-size:11px; letter-spacing:1.5px; color:#94a3b8; text-transform:uppercase; margin-top:6px; }
.var-sub { font-family:'Saira Condensed', Arial, sans-serif; font-weight:700; font-size:15px; letter-spacing:1.5px;
  color:#94a3b8; text-transform:uppercase; margin:2px 0 8px; position:relative; }
.var-log { max-height:230px; overflow-y:auto; position:relative; }
.var-row { display:grid; grid-template-columns:70px 108px 1fr 62px 150px; align-items:center; gap:10px;
  padding:8px 4px; border-top:1px solid #16233a; font-size:14px; }
.var-row.h { color:#64748b; font-size:11px; letter-spacing:1.2px; text-transform:uppercase; border-top:0; padding-top:0; }
.var-time { color:#94a3b8; font-variant-numeric:tabular-nums; }
.var-card { display:inline-flex; align-items:center; gap:6px; font-weight:700; font-size:12px; letter-spacing:.5px; }
.var-card i { width:11px; height:15px; border-radius:2px; display:inline-block; }
.var-card.ok i { background:#22c55e; } .var-card.ok { color:#4ade80; }
.var-card.red i { background:#ef4444; transform:rotate(8deg); } .var-card.red { color:#f87171; }
.var-who { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.var-p { font-variant-numeric:tabular-nums; color:#cbd5e1; text-align:right; }
.var-bar { display:flex; align-items:center; gap:8px; }
.var-bar span { flex:1; height:8px; background:#16233a; border-radius:99px; overflow:hidden; }
.var-bar b { display:block; height:100%; background:linear-gradient(90deg,#22c55e,#FACC15); border-radius:99px; }
.var-bar.over b { background:#ef4444; }
.var-bar em { font-style:normal; font-variant-numeric:tabular-nums; color:#e2e8f0; min-width:56px; text-align:right; }
.var-empty { color:#64748b; font-size:14px; padding:14px 4px; text-align:center; }
.var-cmp { width:100%; border-collapse:collapse; font-size:14px; }
.var-cmp td, .var-cmp th { padding:8px 6px; border-top:1px solid #16233a; text-align:left; }
.var-cmp th { color:#64748b; font-size:11px; letter-spacing:1.2px; text-transform:uppercase; border-top:0; }
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
RUNTIME = {MODEL_NAMES[0]: "ONNX Runtime · INT8 · CPU", MODEL_NAMES[1]: "PyTorch · FP32 · CPU",
           MODEL_NAMES[2]: "PyTorch · FP32 · CPU", MODEL_NAMES[3]: "scikit-learn · CPU",
           MODEL_NAMES[4]: "Kural · CPU"}


def _label(p) -> str:
    return "🚫 hakaret" if p.label else "✅ temiz"


def _card(label: int) -> str:
    return ('<span class="var-card red"><i></i>HAKARET</span>' if label
            else '<span class="var-card ok"><i></i>TEMİZ</span>')


def _bar(ms: float) -> str:
    pct = min(100.0, 100.0 * ms / BUDGET_MS)
    over = " over" if ms > BUDGET_MS else ""
    return (f'<div class="var-bar{over}"><span><b style="width:{pct:.1f}%"></b></span>'
            f'<em>{ms:.1f} ms</em></div>')


def _console(rows, model_name) -> str:
    """VAR odası: skor tabelası + her mesajın kaydı (yanıt süresi, 50 ms bütçeye göre çubuk)."""
    n = len(rows)
    flagged = sum(r["label"] for r in rows)
    avg = sum(r["ms"] for r in rows) / n if n else 0.0
    ok = sum(1 for r in rows if r["ms"] <= BUDGET_MS)
    head = ('<div class="var-head"><span class="var-badge">VAR</span>'
            '<span class="var-title">MODERASYON ODASI</span>'
            f'<span class="var-model">{html.escape(SHORT[model_name])} · {RUNTIME[model_name]}</span>'
            '<span class="var-live">CANLI</span></div>')
    score = ('<div class="var-score">'
             f'<div class="var-tile"><div class="var-num">{n:03d}</div><div class="var-lbl">Mesaj</div></div>'
             f'<div class="var-tile"><div class="var-num">{flagged:03d}</div><div class="var-lbl">Kırmızı kart</div></div>'
             f'<div class="var-tile"><div class="var-num">{avg:.1f}<small>ms</small></div><div class="var-lbl">Ortalama süre</div></div>'
             f'<div class="var-tile"><div class="var-num">{ok}/{n}</div><div class="var-lbl">Bütçe ≤ {BUDGET_MS:.0f} ms</div></div>'
             '</div>')
    sub = f'<div class="var-sub">Maç raporu · her mesaj · dolu çubuk = {BUDGET_MS:.0f} ms</div>'
    if not rows:
        body = '<div class="var-empty">Düdük bekleniyor: henüz mesaj yok.</div>'
    else:
        lines = ['<div class="var-row h"><span>Saat</span><span>Karar</span><span>Kim · model</span>'
                 '<span style="text-align:right">P(hak.)</span><span>Süre</span></div>']
        for r in reversed(rows):
            lines.append(f'<div class="var-row"><span class="var-time">{r["time"]}</span>{_card(r["label"])}'
                         f'<span class="var-who">{html.escape(r["user"])} · {html.escape(r["model"])}</span>'
                         f'<span class="var-p">{r["prob"]:.2f}</span>{_bar(r["ms"])}</div>')
        body = "".join(lines)
    return f'<div class="var">{head}{score}{sub}<div class="var-log">{body}</div></div>'


def _compare(preds) -> str:
    if not preds:
        return '<div class="var"><div class="var-empty">Mesaj gönderince beş modelin kararı burada görünür.</div></div>'
    rows = "".join(f'<tr><td>{html.escape(SHORT[q.model])}</td><td>{_card(q.label)}</td>'
                   f'<td>%{100 * q.confidence:.0f}</td><td>{_bar(q.ms)}</td></tr>' for q in preds)
    return ('<div class="var"><table class="var-cmp"><tr><th>Model</th><th>Karar</th><th>Güven</th>'
            f'<th>Süre</th></tr>{rows}</table></div>')


def on_send(text, username, model_name, history, rows):
    text = (text or "").strip()
    if not text:
        return history, rows, _console(rows or [], model_name), gr.update(), ""
    username = (username or "taraftar").strip()
    p = zoo.predict(model_name, text)

    shown = zoo.mask(text) if p.label else html.escape(text)
    badge = f"{_label(p)} · %{100 * p.confidence:.0f} · {p.ms:.1f} ms"
    history = list(history or []) + [
        {"role": "user", "content": f"**{html.escape(username)}:** {shown}  \n<sub>{badge}</sub>"}
    ]
    rows = list(rows or []) + [{"time": datetime.now().strftime("%H:%M:%S"), "user": username,
                                "model": SHORT[model_name], "label": p.label, "prob": p.prob,
                                "ms": p.ms}]
    preds = [p if name == model_name else zoo.predict(name, text) for name in MODEL_NAMES]
    return history, rows, _console(rows, model_name), _compare(preds), ""


def on_model(model_name, rows):
    return _console(rows or [], model_name)


def on_clear(model_name):
    return [], [], _console([], model_name), _compare([])


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
        # ORTA: yayın + VAR odası (model seçimi + her mesajın yanıt süresi)
        with gr.Column(scale=3):
            now = gr.HTML(_now(0))
            player = gr.HTML(_player(0))
            model_dd = gr.Radio([(SHORT[n], n) for n in MODEL_NAMES], value=INT8_NAME,
                                label="Moderasyon modeli", elem_id="fs-models")
            console = gr.HTML(_console([], INT8_NAME))
            with gr.Accordion("Son mesaj: beş modelin kararı", open=False):
                cmp = gr.HTML(_compare([]))
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

    outs = [chat, log_state, console, cmp, msg]
    ins = [msg, user, model_dd, chat, log_state]
    msg.submit(on_send, ins, outs)
    send.click(on_send, ins, outs)
    model_dd.change(on_model, [model_dd, log_state], console)
    clear.click(on_clear, model_dd, [chat, log_state, console, cmp])


if __name__ == "__main__":
    demo.launch(inbrowser=True, **launch_kw)
