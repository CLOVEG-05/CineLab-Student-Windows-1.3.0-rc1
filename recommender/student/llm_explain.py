"""实验四 S4：大模型个性化推荐解释（可核验）。

原理（课程验收要求）：
- 结合用户**真实历史评分**与 **Content/ItemCF/RRF 等算法依据**，调用**外部大模型 API**
  （本组使用 LM Studio 本地部署的 Qwen2.5-7B-Instruct-1M，OpenAI 兼容接口），
  生成个性化推荐理由。
- **可核验**：把"事实"（用户真实高分电影、推荐电影真实片名/类型、算法来源与名次）
  作为受控上下文喂给模型，并约束其不得编造；返回后程序校验解释中《》引用的
  每个电影名都必须出现在事实里，违例自动重试一次，仍违例则标记 verified=false。

模型连接（环境变量可覆盖）：
- LLM_BASE_URL  默认 http://host.docker.internal:1234/v1（容器访问宿主机 LM Studio）
- LLM_API_KEY   默认 lm-studio
- LLM_MODEL     默认 qwen2.5-7b-instruct-1m
"""
import json
import os
import re
import urllib.request
from datetime import datetime, timezone

import recommender.business as business
from recommender.contracts import Algorithm, PROVIDERS, RecommendationRequest
from recommender.data import DATASET, SPLIT
from sqlalchemy import text

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://host.docker.internal:1234/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "lm-studio")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5-7b-instruct-1m")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "90"))

_MOVIE_SQL = text("SELECT movie_id, movielens_title, COALESCE(title_zh,'') AS title_zh, genres "
                  "FROM movies WHERE movie_id=:mid")
_HISTORY_SQL = text("SELECT r.movie_id, r.rating, m.movielens_title, COALESCE(m.title_zh,'') AS title_zh, m.genres "
                    "FROM history_ratings r JOIN movies m ON m.movie_id=r.movie_id "
                    "WHERE r.ml_user_id=:uid ORDER BY r.rating DESC, r.movie_id LIMIT 8")


def _title(row) -> str:
    return row["title_zh"] or row["movielens_title"]


def build_facts(ml_user_id: int, movie_id: int, algorithm: str) -> dict:
    """收集可核验的事实：推荐电影真实信息 + 用户真实高分历史 + 算法来源依据。"""
    db = business._database()
    with db.connection() as conn:
        target = conn.execute(_MOVIE_SQL, {"mid": movie_id}).mappings().first()
        history = conn.execute(_HISTORY_SQL, {"uid": ml_user_id}).mappings().all()
    if target is None:
        raise KeyError(f"movie {movie_id} not found")

    # 算法依据：该电影在对应算法真实推荐列表中的名次；rrf 额外带来源与名次。
    request = RecommendationRequest("ml", ml_user_id, 10, "business", DATASET, SPLIT)
    result = PROVIDERS[Algorithm[algorithm]].recommend(request)
    rank, sources, ranks = None, [], {}
    for item in result.items:
        if item.movie_id == str(movie_id):
            rank = item.rank
            sources = list(item.recall_sources or [])
            ranks = dict(item.recall_ranks or {})
            break

    # 与推荐电影类型重叠的高分历史电影（真实可核验的相似性证据）。
    target_genres = set((target["genres"] or "").split("|")) - {"(no genres listed)"}
    similar = [h for h in history
               if target_genres & (set((h["genres"] or "").split("|")) - {"(no genres listed)"})]
    return {
        "target": {"movie_id": str(movie_id), "title": _title(target),
                   "movielens_title": target["movielens_title"], "genres": target["genres"]},
        "algorithm": algorithm,
        "rank_in_list": rank,
        "recall_sources": sources,
        "recall_ranks": ranks,
        "history": [{"movie_id": str(h["movie_id"]), "title": _title(h),
                     "rating": float(h["rating"]), "genres": h["genres"]} for h in history],
        "similar_history": [{"title": _title(h), "rating": float(h["rating"])} for h in similar],
        "rating_count": len(history),
    }


def _chat(messages: list[dict], temperature: float) -> str:
    body = json.dumps({"model": LLM_MODEL, "messages": messages,
                       "temperature": temperature, "max_tokens": 320}).encode("utf-8")
    req = urllib.request.Request(
        LLM_BASE_URL.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + LLM_API_KEY})
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"].strip()


def _prompt(facts: dict) -> str:
    hist = "；".join(f"《{h['title']}》{h['rating']:.1f}分（{h['genres']}）" for h in facts["history"])
    src = ""
    if facts["recall_sources"]:
        src = "；rrf 融合来源：" + "、".join(
            f"{s}#{facts['recall_ranks'].get(s, '?')}" for s in facts["recall_sources"])
    sim = "；".join(f"《{h['title']}》{h['rating']:.1f}分" for h in facts["similar_history"]) or "（无同类型高分记录）"
    history_titles = "、".join(f"《{h['title']}》" for h in facts["history"][:5])
    return (
        "你是电影推荐解释助手。只能使用【事实】中给出的电影和信息，禁止编造任何电影名、"
        "评分或数据；解释中引用电影必须用《》且只能引用下列【可引用清单】中的电影。\n"
        "【事实】\n"
        f"- 推荐电影：《{facts['target']['title']}》（类型：{facts['target']['genres']}）\n"
        f"- 该电影在本推荐列表中排第 {facts['rank_in_list']} 位{src}\n"
        f"- 用户真实高分历史（共 {facts['rating_count']} 条评分，以下是评分最高的若干条）：{hist}\n"
        f"- 其中与推荐电影同类型的高分电影：{sim}\n"
        f"【可引用清单（必须且只能从中选电影名）】：推荐电影《{facts['target']['title']}》、用户历史{history_titles}\n"
        "请基于以上事实，用不超过 110 字的中文解释\"为什么向这位用户推荐"
        f"《{facts['target']['title']}》\"，必须引用至少一部他真实看过的电影，只输出解释正文。"
    )


_ALLOWED_RE = re.compile(r"《([^》]{1,60})》")


def _norm_title(s: str) -> str:
    """标题归一化：去大小写、去括号与年份、去冠词 the、去标点，仅留字母/数字/中文。"""
    s = s.lower().replace("（", "(").replace("）", ")").replace("【", "(").replace("】", ")")
    s = re.sub(r"\([^)]*\)", "", s)          # 去掉括号内容（年份等）
    s = re.sub(r"，.*", "", s)                # 去掉中文逗号后的附加说明
    s = re.sub(r"\bthe\b", " ", s)            # 去掉独立冠词 the
    return re.sub(r"[^a-z0-9一-鿿]", "", s)


def _verify(text_out: str, facts: dict) -> bool:
    history_norm = {_norm_title(h["title"]) for h in facts["history"] if h["title"]}
    target_norm = _norm_title(facts["target"]["title"])
    cited = _ALLOWED_RE.findall(text_out)     # 模型用《》标注的片名
    if not cited:                             # 未按要求引用任何电影 → 不通过
        return False
    cited_norm = {_norm_title(t) for t in cited}
    # 必须至少引用一部用户真实看过的电影（历史），且不得出现历史/目标之外的电影。
    if not (cited_norm & history_norm):
        return False
    return all(c in (history_norm | {target_norm}) for c in cited_norm)


def explain(ml_user_id: int, movie_id: int, algorithm: str) -> dict:
    facts = build_facts(ml_user_id, movie_id, algorithm)
    messages = [{"role": "user", "content": _prompt(facts)}]
    output = _chat(messages, temperature=0.3)
    verified = _verify(output, facts)
    if not verified:  # 违例重试一次（更严格约束）
        messages.append({"role": "assistant", "content": output})
        messages.append({"role": "user", "content": "你引用了事实之外的电影。请重新回答：只引用事实中出现过的《》电影名，不超过 110 字，只输出正文。"})
        output = _chat(messages, temperature=0.1)
        verified = _verify(output, facts)
    return {
        "algorithm": algorithm,
        "movie_id": str(movie_id),
        "model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "verified": verified,
        "explanation": output,
        "facts": facts,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="microseconds") + "Z",
    }
