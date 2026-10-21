<script setup lang="ts">
import { ArrowRight } from '@element-plus/icons-vue'
import type { Movie } from '../types'
import MoviePoster from './MoviePoster.vue'
defineProps<{ movie: Movie }>()
</script>
<template>
  <article class="movie-card">
    <RouterLink :to="`/movies/${movie.movie_id}`" :aria-label="`查看 ${movie.title_zh || movie.movielens_title} 详情`">
      <div class="poster-wrap">
        <MoviePoster :path="movie.poster_path" :title="movie.movielens_title" />
        <span class="movie-id">#{{ movie.movie_id }}</span>
        <span v-if="movie.rating" class="my-rating">我的评分 {{ movie.rating }}</span>
        <div class="card-overlay" aria-hidden="true">
          <span class="card-title">{{ movie.title_zh || movie.movielens_title }}</span>
          <span class="card-meta">{{ movie.release_year || '年份未收录' }}<span> · </span>{{ movie.genres.slice(0, 2).join(' / ') || '类型未收录' }}</span>
          <span class="card-cta">查看详情<el-icon><ArrowRight /></el-icon></span>
        </div>
      </div>
      <h3>{{ movie.title_zh || movie.movielens_title }}</h3>
      <p>{{ movie.release_year || '年份未收录' }}<span aria-hidden="true"> · </span>{{ movie.genres.slice(0, 2).join(' / ') || '类型未收录' }}</p>
    </RouterLink>
  </article>
</template>

<style scoped>
.poster-wrap { position: relative; overflow: hidden; border-radius: 8px; }
.movie-poster { transition: transform .4s ease, box-shadow .35s ease, border-color .35s ease; }
.movie-card a:hover .movie-poster,
.movie-card a:focus-visible .movie-poster { transform: translateY(-8px) scale(1.04); box-shadow: 0 22px 46px -20px rgba(225,29,42,.7); border-color: var(--accent); }
.my-rating { top: 12px; right: 12px; left: auto; }
.card-overlay {
  position: absolute; inset: 0; display: flex; flex-direction: column; justify-content: flex-end; gap: 6px;
  padding: 16px; opacity: 0; transform: translateY(10px);
  background: linear-gradient(to top, rgba(10, 6, 6, .94) 8%, rgba(10, 6, 6, .55) 46%, transparent 76%);
  transition: opacity .3s ease, transform .3s ease;
}
.movie-card a:hover .card-overlay,
.movie-card a:focus-visible .card-overlay { opacity: 1; transform: translateY(0); }
.card-title {
  font-size: 14px; font-weight: 600; color: #fff; line-height: 1.4;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
.card-meta { font-size: 11px; color: #d4ddd8; }
.card-meta span { opacity: .6; }
.card-cta {
  margin-top: 6px; width: fit-content; display: inline-flex; align-items: center; gap: 6px;
  background: var(--accent); color: var(--ink); font-size: 12px; font-weight: 600;
  padding: 6px 12px; border-radius: 5px;
}
</style>
