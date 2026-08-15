"""
ZSBR/SFOC: 零信号预算重分配与信号通量控制 — 调度器与采样器
================================================================
理论: docs/20_zsbr_theory.md (V1/V2) + docs/21_signal_flux_theory.md (SFC)
  - V1 选择器: 贪心 top-M(后验期望P̂_sig) + ε-greedy 覆盖 + 冷却 C
  - V2 多重入选: 水填法按边际收益 Δv=P_sig·(1−P_sig)^k 分配组槽位, k≤k_max
  - SFC 信号通量控制(docs/21 §3): 槽位三分 = 贪婪(收割M池) + 投资ι·m(近边界D题挖掘) + ε探索;
    投资候选=观测含0.1格式分 或 全错但观测少(s=0,n≤2G), 按转化概率 1−(1−p̂_post)^{kG} 水填;
    ι=0 严格退化为 V1(回滚开关)
  - 在线估计: Beta(1,1) 先验 + 时间衰减计数, 乐观冷启动

设计约束(零 TRL 手术, 见计划 Rejected Alternatives):
  - ZSBRSampler 严格复刻 TRL RepeatSampler 的输出结构:
      [chunk(m 个 idx 各连续 G 次)] × repeat_count, chunk 重复副本逐位一致
    唯一差异 = chunk 的 m 个 index 来自 scheduler.select_chunk() 而非 randperm 切片。
  - index 即真实 prompt id → reward kwargs["idx"] 语义不变, 无映射簿记。
  - num_workers=0(TRL 默认)下 DataLoader 同步迭代 → select 天然滞后 reward 更新恰一个 cycle。
"""

from __future__ import annotations

import json
import random
from typing import Optional

from torch.utils.data import Sampler
REPO_ROOT = os.environ.get("LLM_TRAINING_ROOT",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", os.path.expanduser("~/models"))



def p_sig_binary(p: float, G: int = 2) -> float:
    """二值近似信号率 (docs/20 式 1.2)。"""
    return 1.0 - p ** G - (1.0 - p) ** G


class ZSBRScheduler:
    """在线 p̂ 估计 + 组槽位选择 (V1 top+ε / V2 水填)。单进程使用。"""

    def __init__(self, num_prompts: int, G: int = 2, mode: str = "v1",
                 epsilon: float = 0.2, k_max: int = 3, cooldown: int = 3,
                 gamma: float = 0.98, alpha0: float = 1.0, beta0: float = 1.0,
                 invest_ratio: float = 0.25, seed: int = 42):
        assert mode in ("v1", "v2", "sfc")
        self.N = num_prompts
        self.G = G
        self.mode = mode
        self.epsilon = epsilon
        self.k_max = k_max
        self.cooldown = cooldown
        self.gamma = gamma
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.invest_ratio = invest_ratio if mode == "sfc" else 0.0
        self.rng = random.Random(seed)
        # 稀疏统计: idx -> [s, n, last_obs_cycle]
        self.stats: dict[int, list] = {}
        self.last_selected: dict[int, int] = {}
        self.seen_format_score: set[int] = set()   # SFC: 观测过 0.1 格式分的题(近边界D证据)
        self.invested: dict[int, int] = {}          # SFC: 投资题 idx -> 投资时 cycle(转化追踪)
        self.conversion_log: list[dict] = []        # SFC: 投资转化事件(F-e 判定)
        self.cycle = 0                      # select_chunk 调用计数
        self.history: list[dict] = []       # 每 cycle 的选择摘要(落盘复盘用)

    # ---- 在线更新 (reward_function 逐条调用) ----
    def update(self, idx: int, reward: float):
        idx = int(idx)
        st = self.stats.setdefault(idx, [0.0, 0.0, self.cycle])
        # 时间衰减: 距上次观测的 cycle 数
        dt = max(0, self.cycle - st[2])
        if dt > 0:
            w = self.gamma ** dt
            st[0] *= w
            st[1] *= w
        passed = reward >= 0.9
        st[0] += 1.0 if passed else 0.0   # 通过 ⟺ reward∈{0.9,1.0}
        st[1] += 1.0
        st[2] = self.cycle
        # SFC: 近边界证据(格式分) + 投资转化追踪(D→M: 投资题观测到通过或混合即转化)
        if 0.05 <= reward < 0.5:
            self.seen_format_score.add(idx)
        if idx in self.invested:
            if passed:
                self.conversion_log.append({"idx": idx, "invest_cycle": self.invested[idx],
                                            "convert_cycle": self.cycle, "converted": True})
                del self.invested[idx]
            # 未转化不立即删: 同 cycle 后续条还可能通过; 超时清理在 select_chunk

    # ---- 估计 ----
    def p_hat(self, idx: int) -> float:
        st = self.stats.get(idx)
        if st is None:
            return self.alpha0 / (self.alpha0 + self.beta0)   # 未见: 0.5(乐观)
        w = self.gamma ** max(0, self.cycle - st[2])
        return (self.alpha0 + w * st[0]) / (self.alpha0 + self.beta0 + w * st[1])

    def p_sig_hat(self, idx: int) -> float:
        """后验期望评分 E[P_sig|数据] = 2αβ/((α+β)(α+β+1)) (G=2, Beta 后验解析式)。

        v1.1 修复(docs/20 §7.2 诊断): 点估计 P_sig(p̂) 使未见题(0.5)与已证实混合题(0.5)
        完全并列, 未见池≫混合池 → 贪婪槽被探索淹没。后验期望给出正确排序:
        混合(0.40) > 未见(0.333) > 一致(0.30), 且随观测次数加深判别。"""
        st = self.stats.get(idx)
        if st is None:
            a, b = self.alpha0, self.beta0
        else:
            w = self.gamma ** max(0, self.cycle - st[2])
            a = self.alpha0 + w * st[0]
            b = self.beta0 + w * (st[1] - st[0])
        if self.G == 2:
            return 2.0 * a * b / ((a + b) * (a + b + 1.0))
        if self.G == 4:
            # G=4 后验期望闭式 (docs/34 §1.3, A.T2a 推导): E[P_sig|D] = 1 − [a₄+b₄]/(a+b)₄
            def rising4(x: float) -> float:
                return x * (x + 1.0) * (x + 2.0) * (x + 3.0)
            return 1.0 - (rising4(a) + rising4(b)) / rising4(a + b)
        # 其他 G>2 退化为点估计(本轮未用)
        return p_sig_binary(a / (a + b), self.G)

    # ---- 选择 (每 generation cycle 一次) ----
    def select_chunk(self, m: int) -> list[int]:
        """返回本 cycle 的 m 个组槽位对应的 prompt index 列表。
        V1: 无放回 top-(1-ε)m + ε·m 均匀; V2: 水填法允许重复 ≤k_max;
        SFC(docs/21 §3): 贪婪(1-ι-ε)m 收割 + 投资 ι·m 近边界D题挖掘 + ε·m 探索。"""
        self.cycle += 1
        n_eps = int(round(self.epsilon * m))
        n_invest = int(round(self.invest_ratio * m)) if self.mode == "sfc" else 0
        n_greedy = max(0, m - n_eps - n_invest)

        # SFC: 投资超时清理(>15 cycle 未转化 → 记负例, F-e 的分母)
        if self.invested:
            expired = [i for i, c in self.invested.items() if self.cycle - c > 15]
            for i in expired:
                self.conversion_log.append({"idx": i, "invest_cycle": self.invested[i],
                                            "convert_cycle": None, "converted": False})
                del self.invested[i]

        # 候选打分(冷却排除只作用于贪婪槽)。全量打分 O(N), N=7473 可忽略。
        cooled = {i for i, t in self.last_selected.items()
                  if self.cycle - t < self.cooldown}
        # v1.1: 后验期望下未见题(0.333)不再与混合题(≥0.40)并列, 无需大量未见代表;
        # 保留少量未见样本参与(供混合池不足时填充), 探索主要由 ε 槽承担。
        seen = list(self.stats.keys())
        unseen_score = None  # 占位: 下方用 p_sig_hat(任意未见 idx) 统一计算
        scored = [(self.p_sig_hat(i), i) for i in seen if i not in cooled]
        # 未见题: 分数并列时随机代表(避免构造全 N 列表), 取足量随机未见样本参与竞争
        n_unseen_pool = min(self.N - len(seen), 4 * m)
        unseen_pool = []
        if n_unseen_pool > 0:
            u_score = 2.0 * self.alpha0 * self.beta0 / (
                (self.alpha0 + self.beta0) * (self.alpha0 + self.beta0 + 1.0)) \
                if self.G == 2 else p_sig_binary(0.5, self.G)   # Beta(1,1): 1/3
            tries = 0
            got = set()
            while len(got) < n_unseen_pool and tries < 20 * n_unseen_pool:
                c = self.rng.randrange(self.N)
                tries += 1
                if c not in self.stats and c not in cooled and c not in got:
                    got.add(c)
            unseen_pool = [(u_score, i) for i in got]

        pool = scored + unseen_pool
        self.rng.shuffle(pool)                      # 并列随机破
        pool.sort(key=lambda x: x[0], reverse=True)

        chunk: list[int] = []
        if self.mode in ("v1", "sfc"):
            # 贪心槽: 无放回 top(SFC 的收割槽同 V1 路径)
            for _, i in pool:
                if len(chunk) >= n_greedy:
                    break
                if i not in chunk:
                    chunk.append(i)
        else:
            # V2 水填: 全局边际收益降序逐槽分配, k≤k_max
            import heapq
            heap = []   # (-Δv, idx, k_current)
            for s, i in pool[: 8 * m]:              # 截断候选池防 O(N log N) 浪费
                heapq.heappush(heap, (-(s * (1 - s) ** 0), i, 0, s))
            counts: dict[int, int] = {}
            while heap and len(chunk) < n_greedy:
                negdv, i, k, s = heapq.heappop(heap)
                if counts.get(i, 0) != k:
                    continue                        # 过期条目
                chunk.append(i)
                counts[i] = k + 1
                if k + 1 < self.k_max:
                    dv = s * (1 - s) ** (k + 1)
                    heapq.heappush(heap, (-dv, i, k + 1, s))

        # ── SFC 投资槽(docs/21 §3): 近边界 D 题, 按转化概率 1−(1−p̂)^{kG} 水填 k≤k_max ──
        invest_picked: list[int] = []
        if n_invest > 0:
            cand = []
            for i, st in self.stats.items():
                if i in chunk or i in cooled:
                    continue
                w = self.gamma ** max(0, self.cycle - st[2])
                a = self.alpha0 + w * st[0]
                b = self.beta0 + w * (st[1] - st[0])
                p_post = a / (a + b)
                # 近边界 D 判据: 低通过率 且 (曾见格式分 或 观测少的全错)
                if p_post < 0.4 and st[0] < 0.5 and \
                        (i in self.seen_format_score or st[1] <= 2 * self.G):
                    cand.append((p_post, i))
            # 水填: 边际收益 Δ(k)=(1−p)^{kG}·(1−(1−p)^G) 递减 → 全局降序堆
            import heapq as _hq
            self.rng.shuffle(cand)
            heap2 = []
            for p_post, i in cand:
                base = (1 - p_post) ** self.G
                _hq.heappush(heap2, (-((1 - base) * 1.0), i, 0, p_post))  # k=0 边际=1−(1−p)^G
            counts2: dict[int, int] = {}
            while heap2 and len(invest_picked) < n_invest:
                negdv, i, k, p_post = _hq.heappop(heap2)
                if counts2.get(i, 0) != k:
                    continue
                invest_picked.append(i)
                counts2[i] = k + 1
                if k + 1 < self.k_max:
                    base = (1 - p_post) ** self.G
                    dv = (base ** (k + 1)) * (1 - base)
                    _hq.heappush(heap2, (-dv, i, k + 1, p_post))
            for i in set(invest_picked):
                self.invested.setdefault(i, self.cycle)
            chunk.extend(invest_picked)

        # ε 槽: 全域均匀(不含已入 chunk 的, V2 也不重复占 ε 槽)
        guard = 0
        while len(chunk) < m and guard < 50 * m:
            c = self.rng.randrange(self.N)
            guard += 1
            if c not in chunk:
                chunk.append(c)

        for i in chunk:
            self.last_selected[i] = self.cycle
        # 摘要落 history
        uniq = len(set(chunk))
        h = {
            "cycle": self.cycle, "unique": uniq, "dup_slots": m - uniq,
            "mean_psig_hat": round(sum(self.p_sig_hat(i) for i in chunk) / m, 4),
            "n_seen_stats": len(self.stats),
        }
        if self.mode == "sfc":
            h["invest_slots"] = len(invest_picked)
            h["invest_unique"] = len(set(invest_picked))
            if invest_picked:
                h["invest_mean_conv"] = round(sum(
                    1 - (1 - self.p_hat(i)) ** self.G for i in invest_picked) / len(invest_picked), 4)
        self.history.append(h)
        return chunk

    # ---- 落盘 ----
    def dump(self, path: str):
        seen_items = [(i, st[0], st[1], self.p_hat(i)) for i, st in self.stats.items()]
        out = {
            "config": {"mode": self.mode, "G": self.G, "epsilon": self.epsilon,
                       "k_max": self.k_max, "cooldown": self.cooldown,
                       "gamma": self.gamma, "invest_ratio": self.invest_ratio, "N": self.N},
            "cycles": self.cycle,
            "n_prompts_observed": len(self.stats),
            "n_prompts_selected": len(self.last_selected),
            "p_hat_seen": [{"idx": i, "s": round(s, 2), "n": round(n, 2),
                            "p_hat": round(p, 4)} for i, s, n, p in seen_items],
            "history": self.history,
            "conversion_log": self.conversion_log,
            "n_invested_pending": len(self.invested),
        }
        with open(path, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)


class ZSBRSampler(Sampler):
    """复刻 TRL RepeatSampler 输出结构的调度采样器。

    yield 序列 = 对每个 chunk: [i_0×G, i_1×G, ..., i_{m-1}×G] 整体重复 repeat_count 次;
    chunk 内容在首次被拉取时由 scheduler.select_chunk(m) 决定(惰性, 拿到最新 p̂)。
    __len__ 与 RepeatSampler 同公式, epoch 语义不变。
    """

    def __init__(self, num_samples: int, scheduler: ZSBRScheduler,
                 mini_repeat_count: int, batch_size: int, repeat_count: int):
        self.num_samples = num_samples
        self.scheduler = scheduler
        self.mini_repeat_count = mini_repeat_count   # = G
        self.batch_size = batch_size                 # = m (unique 槽位数/chunk)
        self.repeat_count = repeat_count             # = num_iterations × steps_per_generation

    def __iter__(self):
        n_chunks = self.num_samples // self.batch_size
        for _ in range(n_chunks):
            chunk = self.scheduler.select_chunk(self.batch_size)
            assert len(chunk) == self.batch_size, "select_chunk 返回槽位数不符"
            # 结构复刻: 先物化完整 chunk 序列, 再逐位重复 yield(副本必然逐位一致)
            flat = [i for i in chunk for _ in range(self.mini_repeat_count)]
            for _ in range(self.repeat_count):
                yield from flat

    def __len__(self):
        return (self.num_samples // self.batch_size) * self.batch_size \
            * self.mini_repeat_count * self.repeat_count
