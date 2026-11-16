"""Capsule endoscopy lesion finder - run:  python app.py   then open http://127.0.0.1:5000
Needs capsule_best.pt (from the training notebook) in the same folder."""
import base64, io, json
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO

HERE = Path(__file__).parent
model = YOLO(str(HERE / "capsule_best.pt"))
try:
    DEFAULT_CONF = float(json.loads((HERE / "model_info.json").read_text())["default_conf"])
except Exception:
    DEFAULT_CONF = 0.25

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@app.post("/predict")
def predict():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="Choose an image first."), 400
    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception:
        return jsonify(error="That file isn’t a readable image. Use a JPG or PNG frame."), 400
    conf = min(max(float(request.form.get("conf", DEFAULT_CONF)), 0.01), 0.95)
    r = model.predict(img, imgsz=640, conf=conf, iou=0.6, verbose=False)[0]
    buf = io.BytesIO()
    Image.fromarray(r.plot()[..., ::-1]).save(buf, "JPEG", quality=90)
    dets = [dict(label=model.names[int(b.cls)], conf=round(float(b.conf), 3)) for b in r.boxes]
    dets.sort(key=lambda d: -d["conf"])
    weak = []
    if not dets:  # show the closest weak guesses so "no lesion" can be told apart from "model not firing"
        lo = model.predict(img, imgsz=640, conf=0.001, iou=0.6, max_det=20, verbose=False)[0]
        weak = sorted((dict(label=model.names[int(b.cls)], conf=round(float(b.conf), 3)) for b in lo.boxes),
                      key=lambda d: -d["conf"])[:3]
    return jsonify(image="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), detections=dets, weak=weak)


@app.get("/")
def index():
    return PAGE.replace("__CONF__", f"{DEFAULT_CONF:.2f}")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Capsule frame review</title>
<style>
:root{--ink:#16242a;--mute:#5b6b70;--paper:#eef2f1;--panel:#fff;--line:#cfd9d8;--accent:#0a6a66;--warn:#fff3cd;--warn-ink:#5c4400}
@media (prefers-color-scheme:dark){:root{--ink:#e6eeee;--mute:#9db0b3;--paper:#0e1719;--panel:#152225;--line:#2a3c40;--accent:#4fc3bb;--warn:#3a3114;--warn-ink:#f2dc95}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
header,main{max-width:1080px;margin:0 auto;padding:0 20px}
header{padding-top:36px}
h1{font:600 clamp(28px,5vw,44px)/1.1 "Iowan Old Style","Palatino Linotype",Georgia,serif;margin:0 0 8px}
header p{margin:0;max-width:60ch;color:var(--mute)}
.note{margin:20px 0;padding:10px 14px;background:var(--warn);color:var(--warn-ink);border-radius:6px;font-size:14px}
main{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,6fr);gap:24px;padding-bottom:48px}
@media (max-width:820px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px}
#drop{display:grid;place-items:center;min-height:240px;border:2px dashed var(--line);border-radius:8px;text-align:center;color:var(--mute);cursor:pointer;overflow:hidden}
#drop.over,#drop:focus-visible{border-color:var(--accent);outline:none}
#drop img{width:100%;display:block}
label{display:flex;justify-content:space-between;margin:18px 0 6px;font-size:14px;color:var(--mute)}
input[type=range]{width:100%;accent-color:var(--accent)}
button{margin-top:16px;width:100%;padding:12px;border:0;border-radius:8px;background:var(--accent);color:#fff;font:600 16px inherit;cursor:pointer}
@media (prefers-color-scheme:dark){button{color:#06201e}}
button:disabled{opacity:.5;cursor:default}
button:focus-visible{outline:3px solid var(--ink);outline-offset:2px}
#out img{width:100%;border-radius:8px;display:block}
ul{list-style:none;margin:14px 0 0;padding:0}
li{display:flex;justify-content:space-between;gap:12px;padding:9px 0;border-top:1px solid var(--line)}
.bar{height:6px;border-radius:3px;background:var(--line);margin-top:6px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}
.msg{color:var(--mute)}.err{color:#b3261e}
</style></head><body>
<header><h1>Capsule frame review</h1>
<p>Upload a single wireless capsule endoscopy frame. The model boxes the lesions it finds and ranks them by confidence.</p>
<div class="note">Research demo. It is not a medical device and must not be used to diagnose or treat anyone.</div></header>
<main>
<section class="card">
<div id="drop" tabindex="0" role="button" aria-label="Choose an image">Drop a frame here or click to choose a file</div>
<input id="file" type="file" accept="image/*" hidden>
<label for="conf"><span>Confidence threshold</span><output id="cv">__CONF__</output></label>
<input id="conf" type="range" min="0.01" max="0.9" step="0.01" value="__CONF__">
<button id="go" disabled>Find lesions</button>
</section>
<section class="card" id="out"><p class="msg">Results appear here.</p></section>
</main>
<script>
const $=s=>document.querySelector(s);let file=null;
const drop=$("#drop"),input=$("#file"),go=$("#go"),out=$("#out");
function pick(f){if(!f||!f.type.startsWith("image/"))return;file=f;go.disabled=false;
  drop.innerHTML="";const im=new Image();im.src=URL.createObjectURL(f);im.alt="Selected frame";drop.append(im);}
drop.onclick=()=>input.click();
drop.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();input.click();}};
input.onchange=()=>pick(input.files[0]);
drop.ondragover=e=>{e.preventDefault();drop.classList.add("over");};
drop.ondragleave=()=>drop.classList.remove("over");
drop.ondrop=e=>{e.preventDefault();drop.classList.remove("over");pick(e.dataTransfer.files[0]);};
$("#conf").oninput=e=>$("#cv").textContent=(+e.target.value).toFixed(2);
go.onclick=async()=>{
  go.disabled=true;go.textContent="Analysing…";
  const fd=new FormData();fd.append("image",file);fd.append("conf",$("#conf").value);
  try{
    const r=await fetch("/predict",{method:"POST",body:fd});const d=await r.json();
    if(!r.ok)throw new Error(d.error||"Request failed");
    const weak=(d.weak||[]).map(x=>`${x.label} ${(x.conf*100).toFixed(0)}%`).join(", ");
    const rows=d.detections.map(x=>`<li><div style="flex:1"><div style="display:flex;justify-content:space-between"><span>${x.label}</span><span>${(x.conf*100).toFixed(0)}%</span></div><div class="bar"><i style="width:${x.conf*100}%"></i></div></div></li>`).join("");
    out.innerHTML=`<img src="${d.image}" alt="Frame with detected lesions">`+
      (rows?`<ul>${rows}</ul>`:`<p class="msg">No lesions above this threshold. ${weak?`Closest weak candidates: ${weak}.`:"Nothing scored above 0.1%."} Lower the slider to see them.</p>`);
  }catch(e){out.innerHTML=`<p class="err">${e.message}</p>`;}
  go.disabled=false;go.textContent="Find lesions";
};
</script></body></html>"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)