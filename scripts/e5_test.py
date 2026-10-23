"""实验五端到端测试：注册账号评分 -> rrf 重算 -> AI 解释依据含新评分（可核验）。"""
import sys
sys.path.insert(0, '/app')
from pathlib import Path
import os
os.environ['APP_DB_PASSWORD'] = Path('/run/secrets/app_password').read_text(encoding='utf-8').strip()
os.environ['APP_JWT_SECRET'] = Path('/run/secrets/jwt_secret').read_text(encoding='utf-8').strip()
from recommender import business
from recommender.contracts import Algorithm, PROVIDERS, RecommendationRequest
from recommender.data import DATASET, SPLIT
from recommender.student import llm_explain
from sqlalchemy import text

LOGIN = '__e5_test__'
db = business._database()

with db.connection() as conn:
    print('account_ratings columns:', [c[0] for c in conn.execute(text(
        "SELECT column_name FROM information_schema.columns WHERE table_name='account_ratings'"))])
    row = conn.execute(text("SELECT account_id FROM accounts WHERE login_name=:n"), {"n": LOGIN}).first()
    if row is None:
        from backend.app.security import PASSWORDS
        hashed = PASSWORDS.hash('E5TestPass#2026')
        conn.execute(text("INSERT INTO accounts(login_name, password_hash) VALUES (:n, :h)"),
                     {"n": LOGIN, "h": hashed})
        conn.commit()
    aid = conn.execute(text("SELECT account_id FROM accounts WHERE login_name=:n"), {"n": LOGIN}).first()[0]
print('account_id =', aid)

with db.connection() as conn:
    movies = [r[0] for r in conn.execute(text(
        "SELECT movie_id FROM movies WHERE genres LIKE '%Drama%' AND movie_id < 1000 LIMIT 6"))]
print('candidate movies:', movies)


def rate(mids):
    with db.connection() as conn:
        for m in mids:
            conn.execute(text("INSERT INTO account_ratings(account_id, movie_id, rating) "
                              "VALUES (:a, :m, :r) ON DUPLICATE KEY UPDATE rating=VALUES(rating)"),
                         {"a": aid, "m": m, "r": 5.0})
        conn.commit()


def rrf_top():
    req = RecommendationRequest("app", aid, 10, "business", DATASET, SPLIT)
    res = PROVIDERS[Algorithm.rrf].recommend(req)
    return [str(i.movie_id) for i in res.items]


with db.connection() as conn:
    existing = conn.execute(text("SELECT COUNT(*) FROM account_ratings WHERE account_id=:a"),
                            {"a": aid}).scalar()
print('existing ratings:', existing)

cold = rrf_top()
print('RRF with zero ratings (cold start):', cold[:5])

rate(movies[:3])
after1 = rrf_top()
print('RRF after rating 3 movies:', after1[:5])

new_movie = movies[3]
rate([new_movie])
after2 = rrf_top()
print(f'RRF after ALSO rating movie {new_movie}:', after2[:5])
print('new rating changed the result?', after1 != after2)

target = int(after2[0])
facts = llm_explain.build_facts('app', aid, target, 'rrf')
print('facts: rating_count', facts['rating_count'], 'favorite_count', facts['favorite_count'],
      'subject_kind', facts['subject_kind'])
print('history(top, newest first):', [h['title'] for h in facts['history'][:4]])
out = llm_explain.explain('app', aid, target, 'rrf')
print('VERIFIED', out['verified'], '| MODEL', out['model'])
print('EXPLAIN', out['explanation'])

print()
print('演示账号保留：login_name=__e5_test__  password=E5TestPass#2026 （可用于实验五页面演示）')
