"""Build the product-page "Fit insight" prototype (prototype/index.html).

The page embeds the model's per-size predictions and the real review counts
behind them for a few items, so it works offline in any browser.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANDS = ["0–4", "5–8", "9–12", "13–16", "17–20", "21+"]


def dress_svg(body, trim, shape):
    """Original, generic dress illustration (no real product photo is used)."""
    skirts = {
        "aline": "M120 210 L80 420 Q180 445 280 420 L240 210 Z",
        "sheath": "M128 210 L118 420 Q180 432 242 420 L232 210 Z",
        "wrap": "M122 210 L84 410 Q170 440 276 418 L238 210 Z",
    }
    extra = {
        "aline": "",
        "sheath": f'<path d="M150 210 L150 420" stroke="{trim}" stroke-width="3" opacity=".35"/>',
        "wrap": f'<path d="M238 210 L140 330 L120 420" stroke="{trim}" stroke-width="4" fill="none" opacity=".45"/>',
    }
    return f"""<svg viewBox="0 0 360 460" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="dress illustration">
 <defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#efe9f7"/><stop offset="1" stop-color="#f9edf1"/></linearGradient>
 <linearGradient id="fab" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{body}"/><stop offset="1" stop-color="{trim}"/></linearGradient></defs>
 <rect width="360" height="460" rx="8" fill="url(#bg)"/>
 <path d="M180 40 Q186 58 180 70" stroke="#9aa0ad" stroke-width="3" fill="none"/>
 <path d="M150 70 L210 70 L230 90 L240 210 L120 210 L130 90 Z" fill="url(#fab)"/>
 <path d="M150 70 Q180 104 210 70" fill="none" stroke="#ffffff" stroke-width="3" opacity=".7"/>
 <path d="{skirts[shape]}" fill="url(#fab)"/>
 <rect x="120" y="200" width="120" height="14" rx="4" fill="{trim}" opacity=".9"/>
 {extra[shape]}
 <ellipse cx="180" cy="438" rx="110" ry="8" fill="#000" opacity=".06"/>
</svg>"""


ILLUSTRATIONS = {
    "174086": dress_svg("#3d5a99", "#253b6e", "aline"),     # navy A-line
    "136110": dress_svg("#9b2f4a", "#6d1f33", "sheath"),    # burgundy sheath
    "126335": dress_svg("#2f8f6f", "#1f6650", "wrap"),      # emerald wrap
}
MARGIN = 0.10

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fit insight prototype</title>
<style>
 body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#f6f6f7;color:#282c3f}
 header{background:#fff;padding:14px 28px;border-bottom:1px solid #eaeaec;font-weight:700;letter-spacing:.5px}
 main{max-width:980px;margin:24px auto;display:flex;gap:28px;padding:0 16px}
 .img{flex:0 0 360px;height:460px;border-radius:6px;overflow:hidden}
 .img svg{width:100%;height:100%;display:block}
 .info{flex:1}
 h1{font-size:22px;margin:0 0 4px} .sub{color:#7e818c;margin-bottom:18px}
 .sizes{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 18px}
 .sizes button{border:1px solid #d4d5d9;background:#fff;border-radius:24px;padding:9px 16px;font:inherit;cursor:pointer}
 .sizes button.sel{border-color:#ff3f6c;color:#ff3f6c;font-weight:600}
 .fit{border:1px solid #eaeaec;background:#fff;border-radius:6px;padding:14px 16px;min-height:74px}
 .fit .head{font-weight:700;margin-bottom:6px} .up{color:#c05600} .down{color:#2b6cb0} .ok{color:#03a685}
 .ev{color:#535766;font-size:14px} .note{margin-top:14px;font-size:12px;color:#94969f}
 select{font:inherit;padding:6px;margin-bottom:12px}
</style></head><body>
<header>FASHION STORE · PROTOTYPE</header>
<main>
 <div class="img" id="img"></div>
 <div class="info">
  <select id="item"></select>
  <h1 id="title"></h1><div class="sub" id="sub"></div>
  <div><strong>Select size</strong></div>
  <div class="sizes" id="sizes"></div>
  <div class="fit" id="fit"><div class="ev">Pick a size to see how it fits people who wore that size.</div></div>
  <div class="note">Advice comes from a model trained on reviews before 2017. Evidence counts are real
   past reviews of this item at that size. Illustrations are generic drawings, not product photos.</div>
 </div>
</main>
<script>
const ITEMS = __DATA__, BANDS = __BANDS__, MARGIN = __MARGIN__;
const pct = (x) => Math.round(x * 100) + "%";
const sel = document.getElementById("item");
sel.innerHTML = ITEMS.map((it, i) => `<option value="${i}">Item ${it.item_id} (${it.category})</option>`).join("");
function show(i) {
  const it = ITEMS[i];
  document.getElementById("img").innerHTML = it.svg || "";
  document.getElementById("title").textContent = `Item ${it.item_id}`;
  document.getElementById("sub").textContent = `${it.category} · ${it.train_reviews.toLocaleString()} past reviews`;
  const box = document.getElementById("sizes");
  box.innerHTML = BANDS.map((b, k) => it.bands[k] ? `<button data-k="${k}">${b}</button>` : "").join("");
  box.querySelectorAll("button").forEach(btn => btn.onclick = () => pick(it, btn));
  document.getElementById("fit").innerHTML = '<div class="ev">Pick a size to see how it fits people who wore that size.</div>';
}
function pick(it, btn) {
  document.querySelectorAll(".sizes button").forEach(b => b.classList.remove("sel"));
  btn.classList.add("sel");
  const v = it.bands[btn.dataset.k], d = v.p_small - v.p_large;
  let head, cls;
  if (d >= MARGIN) { head = "Runs small at this size: consider one size up"; cls = "up"; }
  else if (-d >= MARGIN) { head = "Runs large at this size: consider one size down"; cls = "down"; }
  else { head = "True to size for most people at this size"; cls = "ok"; }
  const ev = v.hist_n ? `${pct(v.hist_small)} of ${v.hist_n} past renters in sizes ${BANDS[btn.dataset.k]} said too small, ${pct(v.hist_large)} too large.`
                      : "No past reviews at this size yet; advice is based on similar items and sizes.";
  document.getElementById("fit").innerHTML = `<div class="head ${cls}">${head}</div><div class="ev">${ev}</div>`;
}
sel.onchange = () => show(sel.value);
show(0);
</script></body></html>
"""


def main(item_ids=("174086", "136110", "126335")):
    ex = {e["item_id"]: e for e in json.loads((ROOT / "results" / "widget_examples.json").read_text())}
    items = [dict(ex[i], svg=ILLUSTRATIONS.get(i, "")) for i in item_ids if i in ex]
    html = (PAGE.replace("__DATA__", json.dumps(items))
                .replace("__BANDS__", json.dumps(BANDS))
                .replace("__MARGIN__", str(MARGIN)))
    out = ROOT / "prototype" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    return out


if __name__ == "__main__":
    print(main())
