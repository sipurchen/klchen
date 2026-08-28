# ============================================================================
# FABLE5 CoT — Five-Stage Reasoning Framework for Skill Prompt Engineering
# Fable5 CoT — 五段式推理框架，供本 repo 各專屬 skill 的 prompt 工程使用
# Purpose: A reusable CoT discipline any klchen skill/agent can adopt for
#          decisions, plans, readiness checks, and multi-source reconciliation.
# ============================================================================

---

## 📋 OVERVIEW / 總覽

### English

Fable5 CoT is a five-stage chain-of-thought discipline — **Frame, Facts, Forks,
Fix Plan, Final Proof**, always in that order, always all five — for turning a
messy situation into a decision worth standing behind. It is domain-agnostic:
it does not assume any particular model, hardware tier, or subsystem in this
repo, so it can be dropped into the prompt layer for any klchen skill/agent
(Spec-Experts controller prompts, agent node prompts, tool_session prompts,
future skills) as a shared reasoning scaffold.

The core insight: the most common way an LLM's analysis goes wrong isn't bad
reasoning — it's silently skipping a stage. Recommending a fix before
confirming the bug is real. Planning next steps without ever naming the
tradeoff that should have been a real decision point. Claiming something
works because the code *looks* right rather than because it was *run*. Each
Fable5 stage exists to block exactly one of these failure modes.

### 中文

Fable5 CoT 是一套五段式思維鏈紀律——**Frame（定位）、Facts（實據）、Forks（分岔）、
Fix Plan（修正計畫）、Final Proof（驗收）**，順序固定、五段都要有——用來把一個
混亂的狀況收斂成一個站得住腳的決定。這套框架不綁定任何特定模型、硬體層級或
本 repo 的子系統，可以直接放進任何 klchen skill／agent 的 prompt 層當共用推理骨架
（Spec-Experts controller prompt、agent node prompt、tool_session prompt，
或未來新增的 skill）。

核心洞察：LLM 分析出錯最常見的原因不是推理能力不夠，而是**悄悄跳過某一段沒發現**。
還沒確認 bug 是真的就先建議修法；規劃下一步卻從沒把該決斷的取捨講出來；
只因為程式碼「看起來對」就宣稱它能用，而不是因為真的跑過。Fable5 的每一段
都是為了擋住其中一種失誤而存在。

---

## 🧩 THE FIVE STAGES / 五個階段

### 1. Frame（定位）

**EN** — State the actual problem or decision at hand, not a restatement of
the request. A good Frame names the tension: what breaks if this isn't
addressed, or what decision is actually blocking progress. If you can't
articulate why this needs a decision *now*, that's a signal the full
framework may be overkill for this request.

**中** — 講清楚實際要解決的問題或決斷，不是把使用者的話重講一遍。好的 Frame
要點出張力：不處理會出什麼問題、或者是什麼決斷卡住了後續進度。如果講不出
「為什麼現在非決斷不可」，代表這個請求可能不需要動用整套框架。

### 2. Facts（實據）

**EN** — Only what you have *directly verified*, with how. Not memory, not
what a doc claims, not an assumption carried over from earlier context. Cite
the command/file/observation each fact came from. If two sources disagree,
report both and flag the discrepancy — never silently pick one. Anything
that matters but hasn't been checked must be listed as unverified, not left
to read like a confirmed fact.

**中** — 只寫**直接查證過**的東西，並附上怎麼查的。不是記憶、不是文件宣稱、
不是沿用前面對話的假設。每條事實都要附來源（指令／檔案／觀察結果）。兩個
來源講的不一樣時，兩邊都寫出來並標註矛盾——不要悄悄選一邊採信。任何重要但
還沒查證的東西，要明講「未驗證」，不能讓它讀起來跟已驗證的事實一樣可信。

### 3. Forks（分岔）

**EN** — The genuine decision points where a reasonable actor could choose
differently, and the choice has real consequences. For each fork: the real
options (not a strawman vs. the obviously-correct pick), the tradeoff, and
a recommendation with the reasoning behind it. If there's really only one
sane option, it isn't a fork — don't manufacture one. This is the stage most
often skipped in favor of jumping straight to a plan; resist that, because a
plan that hides its own judgment calls is one nobody downstream can push
back on.

**中** — 真正需要決斷的地方——換一個理性的人可能會選不同答案，而且選擇有
實際後果。每個分岔要寫：真實的選項（不是稻草人 vs. 明顯正解）、取捨、以及
帶著理由的建議。如果真的只有一個合理選項，那就不是分岔，不要硬生出一個。
這是最常被跳過、直接跳去寫計畫的一段——要抵抗這個衝動，因為一份藏起自己
判斷點的計畫，後面的人根本無從質疑起。

### 4. Fix Plan（修正計畫）

**EN** — Concrete, ordered, actionable steps. Sequenced and dated where
timing matters, with clear ownership if more than one actor is involved.
Every step should be something a specific person/process actually *does* —
not a restatement of the goal. If a step depends on an unresolved Fork, say
so explicitly rather than silently planning around an assumed answer.

**中** — 具體、有順序、可執行的步驟。時程重要時要排序、標日期；牽涉多方時
要標清楚誰負責。每一步都要是「某人／某流程實際會做的動作」，不是把目標
換句話說重講一次。如果某步驟依賴一個還沒決斷的 Fork，要明講，不要悄悄
假設一個答案就繼續規劃下去。

### 5. Final Proof（驗收）

**EN** — How to know the plan actually worked: real, checkable acceptance
criteria — not a feeling. Explicitly state what will **not** be verified or
covered, if anything. "Looks right" is not Final Proof; "ran X, got Y,
matches expected Z" is.

**中** — 怎麼判斷計畫真的做到了：真實、可查驗的驗收判準，不是「感覺應該
可以」。明講這份計畫**沒有**驗到什麼（如果有的話）。「看起來對」不是驗收，
「跑了 X、得到 Y、對得上預期的 Z」才是。

---

## ⚙️ WHEN TO USE / 何時使用

### English

Not every prompt needs the full five-section writeup. Use Fable5 CoT when a
skill/agent in this repo is asked to: make a decision with real consequences,
produce a plan spanning multiple steps, assess whether something is
ready/working/fixed, or reconcile conflicting information from multiple
sources (e.g. multiple monitor signals in the Spec-Experts pipeline
disagreeing, or a controller decision that could route to more than one
expert). For a quick, low-stakes question, skip the ceremony — a short
Fable5 pass is cheap when it's warranted, but padding a simple answer with
five headers just to look thorough defeats the point.

### 中文

不是每個 prompt 都需要完整寫出五段。當本 repo 的某個 skill／agent 被要求：
做一個有實際後果的決斷、產出跨多步驟的計畫、判斷某件事是否就緒／能動／
已修好、或是要調和來自多個來源的矛盾資訊（例如 Spec-Experts pipeline 裡
多個 monitor 訊號互相矛盾、或 controller 的路由決策可能通往一個以上的
expert）——這時候才用 Fable5 CoT。單純、低風險的問題就跳過這套儀式——
真的需要時，Fable5 走一輪成本很低，但只為了看起來嚴謹就硬塞五個標題進
一個簡單答案，反而失去這套框架的意義。

---

## 🔗 INTEGRATION NOTES / 整合筆記

**EN** — This framework is intentionally domain-agnostic and contains no
references to any specific project outside this repo. To wire it into a
specific klchen skill/agent prompt (e.g. `spec_experts/controller.py`'s
routing decisions, or a future `agents/*` node), reference this file from
that skill's own prompt template rather than duplicating the five-stage
text — one canonical copy, many callers.

**中** — 這份框架刻意保持與領域無關，不含任何本 repo 以外的專案內容。要接
進某個特定的 klchen skill／agent prompt（例如 `spec_experts/controller.py`
的路由決策、或未來的 `agents/*` node），從該 skill 自己的 prompt template
指回這份文件即可，不要把五段文字複製貼上到每個地方——一份正本，多處引用。

---

*Added: 2026-08-29*
*Maintainer: SipurChen*
*Source: extracted and formalized from a Claude Code session's consistent
application of this framework across engineering decisions, release-readiness
assessment, and multi-step work planning — see the companion portable version
for use in other AI tools (Perplexity Spaces, Google Antigravity, etc.).*
