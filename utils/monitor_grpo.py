#!/usr/bin/env python3
"""
训练实时仪表盘 (SFT / GRPO 自适应, JSON API + 前端 SVG 渲染)
============================================================
- 解析训练日志: 指标行 / Reward 审计行 / tqdm 进度条 / Trainer 配置
- 用真实训练步数对齐曲线 (从 epoch * steps_per_epoch 推导)
- 提供 /api/metrics JSON 接口, 前端无闪烁轮询渲染 SVG 图表
-  robust: 日志截断自动重置, 半行防护, 缺失字段容错

用法:
  python3 monitor_grpo.py --log results/grm_judge/train.log
  # 浏览器打开 http://localhost:8989
"""

import re, os, time, ast, argparse, threading, json
from http.server import HTTPServer, BaseHTTPRequestHandler
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))


# ═══════════════════════════════════════════════════════════════════════════════
# 日志解析
# ═══════════════════════════════════════════════════════════════════════════════

_METRIC_RE = re.compile(r"\{[^}]+\}")

# tqdm 进度条:  10%|▉  | 64/672 [07:10<59:26,  5.87s/it]
_PROGRESS_RE = re.compile(
    r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+)(s/it|it/s)\]"
)
# Trainer 启动时打印的配置
_CFG_TOTAL_RE = re.compile(r"Total optimization steps\s*=\s*([\d,]+)")
_CFG_SPE_RE = re.compile(r"Num update steps per epoch\s*=\s*([\d,]+)")
_CFG_EPOCHS_RE = re.compile(r"Num Epochs\s*=\s*(\d+)")

# Reward 审计 (GRPO): [Reward 审计 #10] total=0.318 | acc=0.400(nonzero=2/4) | ...
_AUDIT_RE = re.compile(
    r"\[Reward 审计 #(\d+)\]\s+total=([\d.-]+)\s*\|\s*"
    r"acc=([\d.-]+)\(nonzero=(\d+)/(\d+)\)\s*\|\s*"
    r"fmt=([\d.-]+)\((\d+)/(\d+)\)\s*\|\s*"
    r"cit=([\d.-]+)\((\d+)/(\d+)\)\s*\|\s*"
    r"len=([\d.-]+)\((\d+)/(\d+)\)"
)


def _is_number(s) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def parse_metrics(line: str):
    m = _METRIC_RE.search(line)
    if not m:
        return None
    try:
        d = ast.literal_eval(m.group())
        if not isinstance(d, dict):
            return None
        return {k: float(v) for k, v in d.items() if _is_number(v)}
    except (ValueError, SyntaxError):
        return None


def parse_audit(line: str):
    m = _AUDIT_RE.search(line)
    if not m:
        return None
    g = m.groups()
    return {
        "step": int(g[0]), "total": float(g[1]),
        "acc": float(g[2]), "fmt": float(g[5]),
        "cit": float(g[8]), "len": float(g[11]),
    }


def parse_progress(line: str):
    """返回该行中最后一个完整进度条 (一行可能含多个 \\r 覆盖的进度)"""
    matches = _PROGRESS_RE.findall(line)
    if not matches:
        return None
    pct, cur, total, elapsed, eta, speed, unit = matches[-1]
    return {
        "pct": int(pct), "step": int(cur), "total": int(total),
        "elapsed": elapsed, "eta": eta, "speed": f"{speed}{unit}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 数据采集
# ═══════════════════════════════════════════════════════════════════════════════

def _smooth(data, window=5):
    if len(data) < window:
        return list(data)
    out = []
    for i in range(len(data)):
        s = max(0, i - window + 1)
        out.append(sum(data[s:i + 1]) / (i - s + 1))
    return out


class DataCollector:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.lock = threading.Lock()
        self.last_pos = 0
        # 配置 (来自 Trainer 启动日志)
        self.total_steps = None
        self.steps_per_epoch = None
        self.num_epochs = None
        # 记录: 每条 = {"step": 真实步数, ...指标}
        self.train_records = []   # 含 loss / reward / kl 等训练指标
        self.eval_records = []    # 含 eval_* 指标
        self.audit_records = []   # GRPO reward 审计
        self.progress = None      # 最新 tqdm 进度

    # ---------- 增量读取 ----------
    def read_new_lines(self):
        if not os.path.exists(self.log_path):
            return
        with self.lock:
            size = os.path.getsize(self.log_path)
            if size < self.last_pos:
                # 日志被截断重写 → 清空全部状态从头读
                self._reset_state()
            if size <= self.last_pos:
                return
            with open(self.log_path, "rb") as f:
                f.seek(self.last_pos)
                raw = f.read(size - self.last_pos)
            # 只处理到最后一个换行, 防止半行丢失
            last_nl = raw.rfind(b"\n")
            if last_nl == -1:
                return
            self.last_pos += last_nl + 1
            text = raw[:last_nl + 1].decode("utf-8", errors="replace")
            for line in text.split("\n"):
                line = line.strip()
                if line:
                    self._process_line(line)

    def _reset_state(self):
        self.last_pos = 0
        self.total_steps = self.steps_per_epoch = self.num_epochs = None
        self.train_records, self.eval_records, self.audit_records = [], [], []
        self.progress = None

    # ---------- 单行分发 ----------
    def _process_line(self, line: str):
        # 1) Trainer 配置
        m = _CFG_TOTAL_RE.search(line)
        if m:
            self.total_steps = int(m.group(1).replace(",", ""))
        m = _CFG_SPE_RE.search(line)
        if m:
            self.steps_per_epoch = int(m.group(1).replace(",", ""))
        m = _CFG_EPOCHS_RE.search(line)
        if m:
            self.num_epochs = int(m.group(1))

        # 2) tqdm 进度 (仅接受与 total_steps 匹配的, 过滤加载/格式化进度条)
        #    注意: \r 可能使进度条与指标字典合并到同一行, 故命中后不能提前 return
        p = parse_progress(line)
        if p and self.total_steps and p["total"] == self.total_steps:
            self.progress = p

        # 3) 指标字典
        m = parse_metrics(line)
        if m:
            self._add_metric(m)
            return

        # 4) Reward 审计 (GRPO)
        a = parse_audit(line)
        if a:
            self.audit_records.append(a)

    def _derive_step(self, m) -> int:
        """优先用 epoch * steps_per_epoch 推导真实步数, 否则用序号"""
        if "epoch" in m and self.steps_per_epoch:
            return max(1, round(m["epoch"] * self.steps_per_epoch))
        return len(self.train_records) + len(self.eval_records) + 1

    def _add_metric(self, m: dict):
        step = self._derive_step(m)
        is_eval = any(k.startswith("eval_") for k in m)
        rec = {"step": step}
        rec.update(m)
        if is_eval:
            self.eval_records.append(rec)
        else:
            self.train_records.append(rec)

    # ---------- 组装 API 数据 ----------
    def build_payload(self) -> dict:
        with self.lock:
            tr = [dict(r) for r in self.train_records]
            ev = [dict(r) for r in self.eval_records]
            au = [dict(r) for r in self.audit_records]
            prog = dict(self.progress) if self.progress else None
            total_steps, spe = self.total_steps, self.steps_per_epoch

        def col(records, key):
            xs, ys = [], []
            for r in records:
                if key in r:
                    xs.append(r["step"])
                    ys.append(r[key])
            return xs, ys

        # 训练指标序列
        step, loss = col(tr, "loss")
        _, grad = col(tr, "grad_norm")
        _, lr = col(tr, "learning_rate")
        _, reward = col(tr, "reward")
        _, kl = col(tr, "kl")
        _, entropy = col(tr, "entropy")
        _, comp_len = col(tr, "completions/mean_length")
        _, clipped = col(tr, "completions/clipped_ratio")
        # eval 序列
        eval_step, eval_loss = col(ev, "eval_loss")

        mode = "sft" if loss and not reward else ("grpo" if reward else "auto")

        series = {
            "step": step, "loss": loss, "loss_smooth": _smooth(loss),
            "grad_norm": grad, "learning_rate": lr,
            "reward": reward, "reward_smooth": _smooth(reward),
            "kl": kl, "entropy": entropy,
            "comp_len": comp_len, "clipped": clipped,
            "eval_step": eval_step, "eval_loss": eval_loss,
        }
        audit = {
            "step": [a["step"] for a in au],
            "acc": [a["acc"] for a in au], "fmt": [a["fmt"] for a in au],
            "cit": [a["cit"] for a in au], "len": [a["len"] for a in au],
            "total": [a["total"] for a in au],
        }

        latest = {}
        if tr:
            last = tr[-1]
            for k in ("loss", "grad_norm", "learning_rate", "reward", "kl", "entropy"):
                if k in last:
                    latest[k] = last[k]
        if eval_loss:
            latest["eval_loss"] = eval_loss[-1]

        cur_step = prog["step"] if prog else (step[-1] if step else 0)

        return {
            "mode": mode,
            "total_steps": total_steps,
            "steps_per_epoch": spe,
            "num_epochs": self.num_epochs,
            "progress": prog,
            "current_step": cur_step,
            "series": series,
            "audit": audit,
            "latest": latest,
            "best_eval_loss": min(eval_loss) if eval_loss else None,
            "counts": {"train": len(tr), "eval": len(ev), "audit": len(au)},
            "updated": time.strftime("%H:%M:%S"),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP 服务
# ═══════════════════════════════════════════════════════════════════════════════

class Handler(BaseHTTPRequestHandler):
    collector: DataCollector = None
    html: bytes = b""

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/api/metrics"):
            self.collector.read_new_lines()
            body = json.dumps(self.collector.build_payload(), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.html)


def main():
    parser = argparse.ArgumentParser(description="训练实时仪表盘")
    parser.add_argument("--log", type=str, help="训练日志文件路径")
    parser.add_argument("--port", type=int, default=8989)
    parser.add_argument("--interval", type=int, default=3, help="前端刷新间隔(秒)")
    args = parser.parse_args()

    log_path = args.log
    if not log_path:
        for cand in ("results/grm_judge/train.log", "results/grpo_v4/train.log"):
            if os.path.exists(cand):
                log_path = cand
                break
    if not log_path:
        print("用法: python3 monitor_grpo.py --log results/grm_judge/train.log")
        return

    collector = DataCollector(log_path)
    collector.read_new_lines()
    Handler.collector = collector
    Handler.html = HTML_TEMPLATE.replace("__INTERVAL__", str(args.interval * 1000)).encode()

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"📊 训练实时仪表盘已启动")
    print(f"   🌐 浏览器打开: http://localhost:{args.port}")
    print(f"   📄 读取日志: {log_path}")
    print(f"   🔄 每 {args.interval} 秒自动刷新 (无闪烁)")
    print(f"   (按 Ctrl+C 停止)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n监控已停止")
        server.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# 前端页面 (自包含, 无 CDN 依赖)
# ═══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>训练实时仪表盘</title>
<style>
  :root{
    --bg:#12141f; --card:#1b1e2e; --card2:#20243a; --line:#2c3044;
    --txt:#dfe3ee; --sub:#8b91a7; --accent:#5b8ff9; --good:#4ad6a5; --warn:#f6bd16; --bad:#e8684a;
  }
  *{box-sizing:border-box; margin:0; padding:0;}
  body{background:var(--bg); color:var(--txt); font-family:"Segoe UI",system-ui,-apple-system,sans-serif; padding:18px 22px;}
  .top{display:flex; align-items:center; gap:14px; margin-bottom:14px; flex-wrap:wrap;}
  .top h1{font-size:19px; font-weight:700;}
  .badge{padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; letter-spacing:.5px;}
  .badge.sft{background:#1e3a5f; color:#6db3ff;}
  .badge.grpo{background:#3a2a1e; color:#ffb066;}
  .badge.auto{background:#2a2a3a; color:#999;}
  .live{display:flex; align-items:center; gap:7px; font-size:12px; color:var(--sub);}
  .dot{width:9px; height:9px; border-radius:50%; background:var(--good); animation:pulse 1.6s infinite;}
  @keyframes pulse{0%,100%{opacity:1;}50%{opacity:.25;}}
  .updated{margin-left:auto; font-size:12px; color:var(--sub);}
  .progress-wrap{background:var(--card); border-radius:12px; padding:14px 18px; margin-bottom:14px;}
  .progress-meta{display:flex; justify-content:space-between; font-size:13px; margin-bottom:8px; flex-wrap:wrap; gap:8px;}
  .progress-meta b{color:var(--txt);}
  .progress-meta span{color:var(--sub);}
  .bar{height:10px; background:#262a3c; border-radius:6px; overflow:hidden;}
  .bar-fill{height:100%; background:linear-gradient(90deg,#3d6fe0,#5b8ff9); border-radius:6px; transition:width .6s ease; width:0%;}
  .stats{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:16px;}
  .stat{background:var(--card); border-radius:12px; padding:13px 16px;}
  .stat .k{font-size:11px; color:var(--sub); text-transform:uppercase; letter-spacing:.6px; margin-bottom:5px;}
  .stat .v{font-size:21px; font-weight:700; font-variant-numeric:tabular-nums;}
  .stat .s{font-size:11px; color:var(--sub); margin-top:3px;}
  .v.good{color:var(--good);} .v.warn{color:var(--warn);} .v.accent{color:var(--accent);}
  .charts{display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); gap:14px;}
  .card{background:var(--card); border-radius:12px; padding:14px 16px;}
  .card-title{font-size:13.5px; font-weight:700; color:var(--txt); margin-bottom:4px;}
  .legend{display:flex; gap:16px; flex-wrap:wrap; margin-bottom:6px; min-height:16px;}
  .legend span{display:flex; align-items:center; gap:6px; font-size:11px; color:var(--sub);}
  .legend i{width:14px; height:3px; border-radius:2px; display:inline-block;}
  .chart{width:100%;}
  .chart svg{display:block; width:100%; height:auto;}
  .empty{color:#4a5068; font-size:13px; text-align:center; padding:60px 0;}
</style>
</head>
<body>
  <div class="top">
    <h1>📊 训练实时仪表盘</h1>
    <span class="badge auto" id="modeBadge">…</span>
    <span class="live"><span class="dot"></span>LIVE</span>
    <span class="updated" id="updated"></span>
  </div>

  <div class="progress-wrap">
    <div class="progress-meta">
      <span>进度 <b id="pStep">–</b> / <b id="pTotal">–</b> steps</span>
      <span><b id="pPct">–</b>%</span>
      <span>已用 <b id="pElapsed">–</b></span>
      <span>预计剩余 <b id="pEta">–</b></span>
      <span>速度 <b id="pSpeed">–</b></span>
    </div>
    <div class="bar"><div class="bar-fill" id="barFill"></div></div>
  </div>

  <div class="stats" id="stats"></div>
  <div class="charts" id="charts"></div>

<script>
const REFRESH = __INTERVAL__;
const C = {blue:"#5b8ff9", yellow:"#f6bd16", red:"#e8684a", green:"#4ad6a5",
           purple:"#9270ca", cyan:"#6dc8ec", magenta:"#ff9d9d", gray:"#8b91a7"};

function el(tag, attrs, parent){
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for(const k in attrs) e.setAttribute(k, attrs[k]);
  if(parent) parent.appendChild(e);
  return e;
}
function niceTicks(min, max, n){
  const span = max - min;
  if(span <= 0) return [min];
  const step0 = Math.pow(10, Math.floor(Math.log10(span/n)));
  const err = span/n/step0;
  const step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
  const out = [];
  for(let v = Math.ceil(min/step)*step; v <= max + step*1e-6; v += step) out.push(v);
  return out;
}
function fmt(v){
  if(v === 0) return "0";
  const a = Math.abs(v);
  if(a >= 1e5 || a < 1e-3) return v.toExponential(1);
  if(a < 0.01) return v.toFixed(5);
  if(a < 1) return v.toFixed(3);
  return (Math.round(v*100)/100).toString();
}

function drawChart(host, title, ylabel, seriesList){
  // seriesList: [{x,y,color,label,width,dash,scatter}]
  host.innerHTML = "";
  const legend = document.createElement("div"); legend.className = "legend";
  seriesList.forEach(s=>{
    if(!s.label) return;
    const sp = document.createElement("span");
    sp.innerHTML = '<i style="background:'+s.color+'"></i>'+s.label;
    legend.appendChild(sp);
  });
  const box = document.createElement("div"); box.className = "chart";
  host.appendChild(legend); host.appendChild(box);

  const W = box.clientWidth || 520, H = 230;
  const mL=54, mR=14, mT=12, mB=30, iw=W-mL-mR, ih=H-mT-mB;
  const svg = el("svg",{viewBox:"0 0 "+W+" "+H, width:W, height:H}, box);

  let xs=[], ys=[];
  seriesList.forEach(s=>{ xs=xs.concat(s.x); ys=ys.concat(s.y); });
  if(!xs.length){
    const t = el("text",{x:W/2,y:H/2,fill:"#4a5068","text-anchor":"middle","font-size":13},svg);
    t.textContent = "等待数据…";
    return;
  }
  let xmin=Math.min.apply(null,xs), xmax=Math.max.apply(null,xs);
  let ymin=Math.min.apply(null,ys), ymax=Math.max.apply(null,ys);
  if(xmin===xmax) xmax=xmin+1;
  if(ymin===ymax){ ymax+=1; ymin-=1; }
  const ypad=(ymax-ymin)*0.1; ymin-=ypad; ymax+=ypad;
  const X=v=> mL+(v-xmin)/(xmax-xmin)*iw;
  const Y=v=> mT+ih-(v-ymin)/(ymax-ymin)*ih;

  niceTicks(ymin,ymax,5).forEach(v=>{
    el("line",{x1:mL,y1:Y(v),x2:W-mR,y2:Y(v),stroke:"#262a3c","stroke-width":1},svg);
    el("text",{x:mL-7,y:Y(v)+4,fill:"#6a7190","text-anchor":"end","font-size":10},svg).textContent=fmt(v);
  });
  niceTicks(xmin,xmax,6).forEach(v=>{
    el("line",{x1:X(v),y1:mT,x2:X(v),y2:mT+ih,stroke:"#20243a","stroke-width":1},svg);
    el("text",{x:X(v),y:H-10,fill:"#6a7190","text-anchor":"middle","font-size":10},svg).textContent=Math.round(v);
  });
  el("line",{x1:mL,y1:mT+ih,x2:W-mR,y2:mT+ih,stroke:"#3a3f58","stroke-width":1.2},svg);
  if(ylabel){
    el("text",{x:14,y:mT+ih/2,fill:"#6a7190","font-size":10,
      transform:"rotate(-90 14 "+(mT+ih/2)+")","text-anchor":"middle"},svg).textContent=ylabel;
  }
  seriesList.forEach(s=>{
    if(s.scatter){
      s.x.forEach((xv,i)=> el("circle",{cx:X(xv),cy:Y(s.y[i]),r:3.5,fill:s.color},svg));
    } else {
      const pts = s.x.map((xv,i)=>X(xv).toFixed(1)+","+Y(s.y[i]).toFixed(1)).join(" ");
      el("polyline",{points:pts,fill:"none",stroke:s.color,
        "stroke-width":s.width||2,"stroke-dasharray":s.dash||"","stroke-linejoin":"round"},svg);
    }
  });
}

function statCard(k, v, sub, cls){
  return '<div class="stat"><div class="k">'+k+'</div><div class="v '+(cls||'')+'">'+v+
         '</div><div class="s">'+(sub||'')+'</div></div>';
}

function renderStats(d){
  const L = d.latest, parts = [];
  if(L.loss != null)        parts.push(statCard("Train Loss", L.loss.toFixed(4), "最新训练损失", "accent"));
  if(L.eval_loss != null)   parts.push(statCard("Eval Loss", L.eval_loss.toFixed(4),
        d.best_eval_loss!=null ? "最佳 "+d.best_eval_loss.toFixed(4) : "", "good"));
  if(L.reward != null)      parts.push(statCard("Reward", L.reward.toFixed(3), "最新奖励", "accent"));
  if(L.kl != null)          parts.push(statCard("KL", L.kl.toFixed(5), "KL 散度", ""));
  if(L.entropy != null)     parts.push(statCard("Entropy", L.entropy.toFixed(3), "策略熵", ""));
  if(L.learning_rate != null) parts.push(statCard("Learning Rate", L.learning_rate.toExponential(2), "当前学习率", ""));
  if(L.grad_norm != null)   parts.push(statCard("Grad Norm", L.grad_norm.toFixed(3), "梯度范数", ""));
  parts.push(statCard("指标样本", d.counts.train + " / " + d.counts.eval, "train / eval 记录数", ""));
  document.getElementById("stats").innerHTML = parts.join("");
}

function renderProgress(d){
  const p = d.progress;
  const total = d.total_steps || (p ? p.total : "–");
  const cur = d.current_step || 0;
  document.getElementById("pStep").textContent = cur;
  document.getElementById("pTotal").textContent = total;
  let pct = p ? p.pct : (d.total_steps ? Math.min(100, Math.round(cur/d.total_steps*100)) : 0);
  document.getElementById("pPct").textContent = pct;
  document.getElementById("barFill").style.width = pct + "%";
  document.getElementById("pElapsed").textContent = p ? p.elapsed : "–";
  document.getElementById("pEta").textContent = p ? p.eta : "–";
  document.getElementById("pSpeed").textContent = p ? p.speed : "–";
}

function has(arr){ return arr && arr.length > 0; }

function renderCharts(d){
  const s = d.series, a = d.audit, host = document.getElementById("charts");
  host.innerHTML = "";
  const defs = [];
  if(d.mode === "grpo"){
    defs.push({t:"奖励 Reward", y:"Score", list:[
      {x:s.step,y:s.reward,color:C.blue,label:"reward",width:1.5},
      {x:s.step,y:s.reward_smooth,color:C.yellow,label:"smoothed",width:2.5}]});
    defs.push({t:"奖励分解 Reward Decomposition", y:"Score", list:[
      {x:a.step,y:a.acc,color:C.red,label:"acc"},
      {x:a.step,y:a.fmt,color:C.green,label:"fmt"},
      {x:a.step,y:a.cit,color:C.blue,label:"cit"},
      {x:a.step,y:a.len,color:C.magenta,label:"len"},
      {x:a.step,y:a.total,color:C.gray,label:"total",width:2.5}]});
    defs.push({t:"KL 散度 / 熵", y:"KL", list:[
      {x:s.step,y:s.kl,color:C.purple,label:"kl"},
      {x:s.step,y:s.entropy,color:C.cyan,label:"entropy",dash:"4 3"}]});
    defs.push({t:"生成长度 / 截断率", y:"Tokens", list:[
      {x:s.step,y:s.comp_len,color:C.blue,label:"mean_len"},
      {x:s.step,y:s.clipped,color:C.red,label:"clipped",dash:"4 3"}]});
  } else {
    defs.push({t:"训练损失 Training Loss", y:"Loss", list:[
      {x:s.step,y:s.loss,color:C.blue,label:"loss",width:1.5},
      {x:s.step,y:s.loss_smooth,color:C.yellow,label:"smoothed",width:2.5}]});
    defs.push({t:"训练 vs 验证损失", y:"Loss", list:[
      {x:s.step,y:s.loss_smooth,color:C.blue,label:"train"},
      {x:s.eval_step,y:s.eval_loss,color:C.red,label:"eval",scatter:true}]});
    defs.push({t:"学习率调度 LR Schedule", y:"LR", list:[
      {x:s.step,y:s.learning_rate,color:C.purple,label:"learning_rate"}]});
    defs.push({t:"梯度范数 Grad Norm", y:"Grad Norm", list:[
      {x:s.step,y:s.grad_norm,color:C.cyan,label:"grad_norm"}]});
  }
  defs.forEach(cfg=>{
    const card = document.createElement("div"); card.className = "card";
    card.innerHTML = '<div class="card-title">'+cfg.t+'</div>';
    const holder = document.createElement("div");
    card.appendChild(holder); host.appendChild(card);
    const active = cfg.list.filter(x=>has(x.x) && has(x.y));
    drawChart(holder, cfg.t, cfg.y, active.length ? active : cfg.list);
  });
}

function setMode(mode){
  const b = document.getElementById("modeBadge");
  b.className = "badge " + mode;
  b.textContent = mode === "sft" ? "SFT 监督微调" : (mode === "grpo" ? "GRPO 强化学习" : "等待数据");
}

async function refresh(){
  try{
    const r = await fetch("/api/metrics");
    const d = await r.json();
    setMode(d.mode);
    document.getElementById("updated").textContent = "更新于 " + d.updated;
    renderProgress(d);
    renderStats(d);
    renderCharts(d);
  }catch(e){ console.error(e); }
}
refresh();
setInterval(refresh, REFRESH);
window.addEventListener("resize", refresh);
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
