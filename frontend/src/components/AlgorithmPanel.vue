<script setup lang="ts">
import { computed, onScopeDispose, ref, watch } from 'vue'
import { Connection } from '@element-plus/icons-vue'
import http, { ApiError, errorMessage, fetchExplanation } from '../api'
import StatePanel from './StatePanel.vue'
import type { AlgorithmResult, ExplanationResult } from '../types'

interface MovieMeta {
  movielens_title?: string
  title_zh?: string
  genres?: string[]
  poster_path?: string | null
  release_year?: number | string
}

const props = withDefaults(defineProps<{ endpoint: string | null; label: string; emptySubject?: string }>(), { emptySubject: '登录后体验' })
const loading = ref(false), error = ref<unknown>(null), unavailable = ref(false)
const result = ref<AlgorithmResult | null>(null)
const movieMap = ref<Record<string, MovieMeta>>({})
let run = 0
onScopeDispose(() => { run++ })

function movieTitle(id: string): string {
  const m = movieMap.value[id]
  if (!m) return '电影 #' + id
  return m.movielens_title || m.title_zh || ('电影 #' + id)
}
function movieGenres(id: string): string[] {
  return movieMap.value[id]?.genres || []
}
function moviePoster(id: string): string | null {
  const p = movieMap.value[id]?.poster_path
  return p ? 'https://image.tmdb.org/t/p/w92' + p : null
}

async function loadMovieMeta(ids: string[]) {
  const map: Record<string, MovieMeta> = {}
  await Promise.all(ids.map(async (id) => {
    try {
      const r = await http.get<MovieMeta>(`/movies/${id}`)
      map[id] = r.data
    } catch { /* 单部缺失不影响其余展示 */ }
  }))
  if (Object.keys(map).length) movieMap.value = map
}

async function load() {
  const seq = ++run
  error.value = null; result.value = null; unavailable.value = false; loading.value = false; movieMap.value = {}
  if (!props.endpoint) return
  loading.value = true
  try {
    const response = await http.get<AlgorithmResult>(props.endpoint)
    if (seq !== run) return
    result.value = response.data
    if (response.data.items?.length) {
      loadMovieMeta(response.data.items.map((i) => String(i.movie_id)))
    }
  } catch (e) {
    if (seq !== run) return
    if (e instanceof ApiError && e.status === 501 && e.code === 'ALGORITHM_NOT_IMPLEMENTED') unavailable.value = true
    else error.value = e
  } finally { if (seq === run) loading.value = false }
}
watch(() => props.endpoint, load, { immediate: true })

// 实验四 S4 + 实验五：历史用户（ml）与登录账号（me）均支持本地大模型推荐解释。
const explainContext = computed(() => {
  const ml = (props.endpoint || '').match(/\/demo\/history-users\/(\d+)\/recommendations\/([a-z]+)/)
  if (ml) return { kind: 'ml' as const, userId: ml[1], algorithm: ml[2] }
  const me = (props.endpoint || '').match(/\/me\/recommendations\/([a-z]+)/)
  return me ? { kind: 'me' as const, userId: '', algorithm: me[1] } : null
})
const explainLoading = ref<Record<string, boolean>>({})
const explainResult = ref<Record<string, ExplanationResult>>({})
const explainError = ref<Record<string, string>>({})

async function onExplain(movieId: string) {
  const ctx = explainContext.value
  if (!ctx || explainLoading.value[movieId]) return
  explainLoading.value = { ...explainLoading.value, [movieId]: true }
  explainError.value = { ...explainError.value, [movieId]: '' }
  try {
    const subject = ctx.kind === 'ml' ? { kind: 'ml' as const, userId: ctx.userId } : { kind: 'me' as const }
    const data = await fetchExplanation(subject, ctx.algorithm, movieId)
    explainResult.value = { ...explainResult.value, [movieId]: data }
  } catch (e) {
    explainError.value = { ...explainError.value, [movieId]: errorMessage(e) }
  } finally {
    explainLoading.value = { ...explainLoading.value, [movieId]: false }
  }
}

function explainBasis(facts: ExplanationResult['facts']): string {
  const top = facts.history.slice(0, 3).map((h: { title: string; rating: number }) => `《${h.title}》${h.rating.toFixed(1)}分`).join('、')
  const favs = (facts.favorites || []).slice(0, 2).map((f: { title: string }) => `收藏《${f.title}》`).join('、')
  const src = facts.recall_sources.length
    ? '；rrf 来源 ' + facts.recall_sources.map((s: string) => `${s}#${facts.recall_ranks[s] ?? '?'}`).join('、')
    : ''
  const parts = [top, favs].filter(Boolean).join('、')
  return `依据：你真实评分 ${parts || '（暂无）'}${src}`
}
</script>
<template>
  <section class="algorithm-panel" :aria-label="label">
    <div class="section-top"><h3>{{ label }}</h3><span class="small-label">RECOMMENDATION</span></div>
    <StatePanel :loading="loading" :error="error" @retry="load">
      <div v-if="!endpoint || unavailable" class="algorithm-empty" role="status">
        <div class="algorithm-symbol"><el-icon aria-hidden="true"><Connection /></el-icon></div>
        <div><span class="status-pill">{{ !endpoint ? emptySubject : '尚未实现' }}</span><h3>{{ !endpoint ? '选择观众，开始探索' : '下一部好电影，等待算法发现' }}</h3><p>{{ !endpoint ? '先选择历史用户或登录注册账号。推荐功能等待算法接入。' : '等待算法接入；当前没有模型版本或推荐结果。' }}</p></div>
        <RouterLink v-if="!endpoint && emptySubject === '登录后体验'" class="text-link" to="/login">前往登录 →</RouterLink>
      </div>
      <div v-else-if="result" class="recommendation-results">
        <p class="meta-line">模型版本 {{ result.model_version }} · 数据集 {{ result.dataset_version }}</p>
        <p v-if="!result.items.length">本次没有推荐结果。</p>
        <ol v-else class="rec-list">
          <li v-for="item in result.items" :key="item.movie_id">
            <img v-if="moviePoster(String(item.movie_id))" class="rec-poster" :src="moviePoster(String(item.movie_id))!" :alt="movieTitle(String(item.movie_id))" />
            <div class="rec-main">
              <span class="rec-rank">#{{ item.rank }}</span>
              <RouterLink class="rec-title text-link" :to="`/movies/${item.movie_id}`">{{ movieTitle(String(item.movie_id)) }}</RouterLink>
              <div class="rec-genres"><span v-for="g in movieGenres(String(item.movie_id))" :key="g" class="genre-chip">{{ g }}</span></div>
              <div class="rec-sources" v-if="item.recall_sources && item.recall_sources.length">
                <span class="source-label">融合来源</span>
                <span v-for="s in item.recall_sources" :key="s" class="source-chip">{{ s }}<template v-if="item.recall_ranks"> #{{ item.recall_ranks[s] }}</template></span>
              </div>
              <div class="rec-explain" v-if="explainContext">
                <el-button size="small" type="primary" plain :loading="explainLoading[String(item.movie_id)]"
                           @click="onExplain(String(item.movie_id))">AI 解释</el-button>
                <p v-if="explainError[String(item.movie_id)]" class="explain-err">{{ explainError[String(item.movie_id)] }}</p>
                <template v-if="explainResult[String(item.movie_id)]">
                  <p class="explain-text">{{ explainResult[String(item.movie_id)].explanation }}</p>
                  <p class="explain-basis">{{ explainBasis(explainResult[String(item.movie_id)].facts) }}</p>
                  <span class="explain-verify" :class="explainResult[String(item.movie_id)].verified ? 'ok' : 'warn'">
                    大模型:{{ explainResult[String(item.movie_id)].model }} · {{ explainResult[String(item.movie_id)].verified ? '已核验（理由均出自真实评分与算法依据）' : '注意：解释含事实外引用，请核对' }}
                  </span>
                </template>
              </div>
            </div>
            <div class="rec-score" v-if="item.score !== null">模型分数 {{ item.score }}</div>
          </li>
        </ol>
      </div>
    </StatePanel>
  </section>
</template>

<style scoped>
.recommendation-results .meta-line { color: var(--text-dim, #9aa0a6); font-size: 13px; margin-bottom: 10px; }
.rec-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; }
.rec-list li { display: flex; align-items: center; gap: 12px; padding: 8px; border: 1px solid var(--border, #2a2a2a); border-radius: 10px; background: rgba(255,255,255,0.02); }
.rec-poster { width: 38px; height: 57px; object-fit: cover; border-radius: 4px; background: #1a1a1a; flex: 0 0 auto; }
.rec-main { flex: 1 1 auto; min-width: 0; }
.rec-rank { font-weight: 700; color: var(--accent, #c1121f); margin-right: 6px; }
.rec-title { font-size: 15px; font-weight: 600; }
.rec-genres { margin-top: 4px; display: flex; flex-wrap: wrap; gap: 4px; }
.genre-chip { font-size: 11px; padding: 1px 7px; border-radius: 999px; background: rgba(193,18,31,0.15); color: #e0a3a8; border: 1px solid rgba(193,18,31,0.35); }
.rec-sources { margin-top: 6px; display: flex; flex-wrap: wrap; align-items: center; gap: 5px; }
.source-label { font-size: 11px; color: var(--text-dim, #9aa0a6); }
.source-chip { font-size: 11px; padding: 1px 8px; border-radius: 999px; background: rgba(64,120,242,0.16); color: #a9c4ff; border: 1px solid rgba(64,120,242,0.45); }
.rec-score { flex: 0 0 auto; font-size: 13px; color: var(--text-dim, #9aa0a6); white-space: nowrap; }
.rec-explain { margin-top: 8px; display: flex; flex-direction: column; gap: 6px; }
.explain-err { color: #e0a3a8; font-size: 12px; margin: 0; }
.explain-text { margin: 0; font-size: 13px; line-height: 1.6; color: #e8eaed; }
.explain-basis { margin: 0; font-size: 12px; color: var(--text-dim, #9aa0a6); }
.explain-verify { font-size: 11px; }
.explain-verify.ok { color: #79c0a0; }
.explain-verify.warn { color: #e7b87a; }
</style>
