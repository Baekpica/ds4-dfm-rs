#!/usr/bin/env python3
"""Merge generate_compare.py outputs of two recipes into metrics.json and docs/GENERATION_COMPARE.md."""
import argparse
import json
from pathlib import Path

import generate_compare as gc

CATS = ['ko_chat', 'ko_longctx', 'ko_think', 'en_chat', 'code', 'math_think', 'tool', 'ja_chat', 'zh_chat']
CAT_KO = {'ko_chat': '한국어 chat', 'ko_longctx': '긴 영어 문맥 + 한국어 질문', 'ko_think': '한국어 think',
          'en_chat': '영어 chat', 'code': '코드', 'math_think': '수학 think', 'tool': 'tool call',
          'ja_chat': '일본어', 'zh_chat': '중국어'}


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def fmt(v, pct=False, nd=3):
    if v is None:
        return '–'
    return f'{100 * v:.0f}%' if pct else f'{v:.{nd}f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--r1', type=Path, default=Path('/workspace/axk2/work/gen-r1/R1.jsonl'))
    ap.add_argument('--r2', type=Path, default=Path('/workspace/axk2/work/gen/R2.jsonl'))
    ap.add_argument('--m1', type=Path, default=Path('/workspace/axk2/work/gen-r1/metrics.json'))
    ap.add_argument('--m2', type=Path, default=Path('/workspace/axk2/work/gen/metrics.json'))
    ap.add_argument('--out', type=Path, default=Path('/workspace/axk2/work/gen/metrics.json'))
    ap.add_argument('--md', type=Path, default=Path('/workspace/axk2/docs/GENERATION_COMPARE.md'))
    a = ap.parse_args()
    r1, r2 = load(a.r1), load(a.r2)
    m1, m2 = json.loads(a.m1.read_text()), json.loads(a.m2.read_text())
    merged = {'R1': m1['R1'], 'R2': m2['R2']}
    merged['R1']['aggregate'] = gc.aggregate(r1)
    merged['R2']['aggregate'] = gc.aggregate(r2)
    a.out.write_text(json.dumps(merged, ensure_ascii=False, indent=1) + '\n')
    A1, A2 = merged['R1']['aggregate'], merged['R2']['aggregate']
    L = []
    L.append('# A.X-K2 VQ 생성 비교: R1(하단 1–7층 10-bit) vs R2(16·20·…·40층 10-bit)\n')
    L.append('두 레시피는 같은 바이트(10-bit 7층 + 9-bit 53층, 전문가 imatrix|ls|split3|col_power 0.75, '
             '같은 Q4_K/Q8_0 body)이고 10-bit 층의 위치만 다릅니다. 이 인스턴스(RTX PRO 6000 Blackwell, '
             'sm_120, CUDA 13.2)에서 `tools/generate_compare.py`로 32개 프롬프트를 greedy 생성했습니다. '
             '수치는 GPU 아키텍처, CUDA 버전, 커널, 배치 구성에 따라 달라집니다(근접 동률 argmax가 뒤집히면 이후 '
             '생성이 갈라짐). 그래서 아래 비교는 통계적 비교이고, DGX Spark 엔진 출력과 토큰 단위로 같을 것을 '
             '기대하면 안 됩니다.\n')
    s = merged['R2'].get('sanity_prefill_vs_goldens', {})
    d = merged['R2'].get('sanity_step_vs_prefill', {})
    L.append('## 하니스 sanity check (R2)\n')
    L.append(f"- teacher-forced prefill argmax vs `eval_e2e.py` goldens(같은 릴리스 바이트, held-out {s.get('cases')}개 "
             f"케이스 ≤ 2,048토큰, {s.get('positions')}개 위치): **{fmt(s.get('argmax_agreement'), True)} 일치**. "
             '불일치는 배치 구성(8케이스씩 vs 22케이스 전체)에 따른 커널·누적 순서 차이가 근접 동률을 뒤집고, '
             '깊은 층에서 증폭된 것입니다(통과 기준이 아니라 동작 확인용).')
    L.append(f"- 한 번의 incremental decode step vs 같은 위치 prefill logits: 평균 절대 차이 "
             f"{fmt(d.get('mean_abs_logit_diff'))}, 최대 {fmt(d.get('max_abs_logit_diff'))} "
             f"(logit 표준편차 {fmt(d.get('logit_scale_std'), nd=2)}), argmax 동일: {d.get('same_argmax')}. "
             'decode 단계는 packed index에서 바로 계산하는 GEMV 커널(같은 BF16 가중치 반올림, 다른 누적 순서)과 '
             '단일 쿼리 attention을 씁니다.\n')
    L.append('## 요약 지표\n')
    L.append('| 지표 | R1 | R2 |')
    L.append('|---|---:|---:|')
    rows = [('EOS로 끝난 비율', 'finished', True), ('think 닫힘(`</think>`) 비율 (think 프롬프트)', 'think_closed', True),
            ('반복 4-gram 비율 (평균)', 'rep4', False), ('같은 8-gram 최대 반복 횟수 (평균)', 'max_8gram_count', False),
            ('한글 비율 (한국어 프롬프트 답변)', 'hangul_share', True), ('U+FFFD 개수 (평균)', 'replacement_chars', False),
            ('tool call 전부 유효 (tool 프롬프트)', 'tool_ok', True), ('평균 생성 토큰 수', 'tokens', False)]
    for label, key, pct in rows:
        L.append(f"| {label} | {fmt(A1['ALL'][key], pct, 1 if key == 'tokens' else 3)} | "
                 f"{fmt(A2['ALL'][key], pct, 1 if key == 'tokens' else 3)} |")
    L.append('')
    L.append('### 범주별 (EOS 종료 / 반복 4-gram / 범주 특화 지표)\n')
    L.append('| 범주 | n | R1 EOS | R2 EOS | R1 rep4 | R2 rep4 | R1 특화 | R2 특화 |')
    L.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for c in CATS:
        if c not in A1:
            continue
        x, y = A1[c], A2[c]
        spec = 'think_closed' if c in ('ko_think', 'math_think') else 'tool_ok' if c == 'tool' else \
            'hangul_share' if c.startswith('ko') else None
        L.append(f"| {CAT_KO[c]} | {x['n']} | {fmt(x['finished'], True)} | {fmt(y['finished'], True)} | "
                 f"{fmt(x['rep4'])} | {fmt(y['rep4'])} | "
                 f"{(spec + ' ' + fmt(x[spec], True)) if spec else '–'} | {(spec + ' ' + fmt(y[spec], True)) if spec else '–'} |")
    L.append('')
    for name in ('R1', 'R2'):
        t0, t1 = merged[name].get('timing_think_False', {}), merged[name].get('timing_think_True', {})
        L.append(f"- {name} 속도(배치 전체, 이 인스턴스): non-think {fmt(t0.get('tokens_per_s'), nd=1)} tok/s "
                 f"({t0.get('steps')} steps), think {fmt(t1.get('tokens_per_s'), nd=1)} tok/s ({t1.get('steps')} steps). "
                 'DGX Spark 엔진 속도와는 무관합니다.')
    L.append('')
    if 'sampled_korean' in merged['R2']:
        samp = merged['R2']['sampled_korean']
        fin = sum(int(x['metrics']['finished']) for x in samp) / max(1, len(samp))
        rep = sum(x['metrics']['rep4'] for x in samp) / max(1, len(samp))
        L.append(f"R2를 권장 샘플링(temperature 0.9, top-p 0.95, seed 1234)으로 한국어 {len(samp)}개 프롬프트에 다시 돌린 결과: "
                 f"EOS 종료 {fmt(fin, True)}, 반복 4-gram 평균 {fmt(rep)} (출력은 `work/gen/metrics.json`의 `sampled_korean`).\n")
    L.append('## 프롬프트별 출력 (각 1,200자에서 자름)\n')
    by2 = {r['id']: r for r in r2}
    for x in r1:
        y = by2[x['id']]
        L.append(f"### {x['id']} · {CAT_KO.get(x['category'], x['category'])} · {'think' if x['think'] else 'non-think'}\n")
        L.append(f"프롬프트: {x['summary']}\n")
        for name, r in (('R1', x), ('R2', y)):
            m = r['metrics']
            tag = ', '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' for k, v in m.items())
            text = r['output'].replace('<|im_end|>', '⏎<|im_end|>')
            if len(text) > 1200:
                text = text[:1200] + ' …(생략)'
            L.append(f"**{name}** ({tag})\n")
            L.append('```text\n' + text.replace('```', "'''") + '\n```\n')
    a.md.write_text('\n'.join(L) + '\n')
    print(json.dumps({'R1': A1['ALL'], 'R2': A2['ALL']}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
