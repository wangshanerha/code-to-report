#!/usr/bin/env python3
"""Render reviewed report JSON to offline HTML. Python 3.9+, standard library only."""
import argparse
import html
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def esc(value):
    return html.escape(str(value), quote=True)


def pretty(value):
    if isinstance(value, list) and len(value) <= 16:
        if all(isinstance(v, (int, float, bool, str, type(None))) for v in value):
            return json.dumps(value, ensure_ascii=False)
        if all(isinstance(row, list) and len(row) <= 16 and all(isinstance(v, (int, float, bool, type(None))) for v in row) for row in value):
            return '[\n' + ',\n'.join('  ' + json.dumps(row, ensure_ascii=False) for row in value) + '\n]'
    return json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value


def code(value):
    return '<pre><code>' + esc(pretty(value)) + '</code></pre>'


def paras(value):
    return ''.join('<p>' + esc(s) + '</p>' for s in (value if isinstance(value, list) else [value]))


def validate(data):
    """Check structure and referential integrity, never semantic correctness."""
    errors = []
    for key in ('title', 'summary', 'scope', 'version', 'flows', 'examples', 'modules', 'sources', 'verification'):
        if key not in data or not data[key]:
            errors.append('Missing non-empty field: ' + key)
    if errors:
        return errors
    all_ids = set()
    source_ids = {s.get('id') for s in data['sources']}
    nodes = {}
    graph_ids = set()
    for flow in data['flows']:
        fid = flow.get('id', '')
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', fid) or fid in all_ids:
            errors.append('Invalid/duplicate flow ID: ' + str(fid))
        all_ids.add(fid); graph_ids.add(fid)
        local = {n.get('id') for n in flow.get('nodes', [])}
        occupied = set()
        for n in flow.get('nodes', []):
            nid = n.get('id', '')
            if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', nid) or nid in all_ids:
                errors.append('Invalid/duplicate node ID: ' + str(nid))
            all_ids.add(nid); nodes[nid] = n
            for k in ('title', 'input', 'output', 'details', 'example', 'sources'):
                if not n.get(k):
                    errors.append(f'{nid}: missing {k}')
            for source in n.get('sources', []):
                if source not in source_ids:
                    errors.append(f'{nid}: unknown source {source}')
            for k in ('col', 'row'):
                if not isinstance(n.get(k), int) or n[k] < 0:
                    errors.append(f'{nid}: {k} must be a nonnegative integer')
            cell = (n.get('col'), n.get('row'))
            if cell in occupied:
                errors.append(f'{nid}: duplicate layout cell')
            occupied.add(cell)
        out_degree = {nid: 0 for nid in local}
        for edge in flow.get('edges', []):
            if edge.get('from') not in local or edge.get('to') not in local:
                errors.append(fid + ': edge must connect local nodes')
            else:
                out_degree[edge['from']] += 1
            if not edge.get('label'):
                errors.append(fid + ': every edge requires condition/data label')
            for side in ('from_side', 'to_side'):
                if edge.get(side, 'bottom' if side == 'from_side' else 'top') not in ('top', 'bottom', 'left', 'right'):
                    errors.append(fid + ': invalid port')
        for n in flow.get('nodes', []):
            degree = out_degree.get(n['id'], 0)
            if n.get('kind') == 'decision' and degree < 2:
                errors.append(n['id'] + ': decision needs at least two labeled exits')
            if n.get('kind') != 'terminal' and not degree:
                errors.append(n['id'] + ': nonterminal node has no exit')
    example_ids = set()
    for example in data['examples']:
        eid = example.get('id', '')
        if eid in example_ids or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', eid):
            errors.append('Invalid/duplicate example ID: ' + str(eid))
        example_ids.add(eid)
        if not example.get('provenance'):
            errors.append(eid + ': example must disclose provenance')
        for fid, visits in example.get('paths', {}).items():
            if fid not in graph_ids:
                errors.append(eid + ': unknown flow ' + fid); continue
            flow = next(f for f in data['flows'] if f['id'] == fid)
            local = {n['id'] for n in flow['nodes']}
            pairs = {(e['from'], e['to']) for e in flow['edges']}
            ids = [v.get('node') for v in visits]
            if not ids:
                errors.append(eid + ': empty path')
            for visit in visits:
                if visit.get('node') not in local or 'input' not in visit or 'output' not in visit:
                    errors.append(eid + ': visit needs local node and input/output')
            for a, b in zip(ids, ids[1:]):
                if (a, b) not in pairs:
                    errors.append(f'{eid}/{fid}: nonexistent edge {a} -> {b}')
    for m in data['modules']:
        if not m.get('input') or not m.get('output') or not m.get('details'):
            errors.append('Module requires input/output/details: ' + m.get('name', '?'))
    return errors


def source_links(ids, sources):
    lookup = {s['id']: s for s in sources}
    return ' · '.join('<a href="#source-' + esc(i) + '">' + esc(lookup[i]['label']) + '</a>' for i in ids)


def graph(flow):
    nodes = {n['id']: n for n in flow['nodes']}
    width = max(n['col'] for n in nodes.values()) * 330 + 340
    height = max(n['row'] for n in nodes.values()) * 156 + 150

    def box(n):
        return (40 + n['col'] * 330, 24 + n['row'] * 156)

    def port(n, side):
        x, y = box(n)
        return {'top': (x + 130, y), 'bottom': (x + 130, y + 94), 'left': (x, y + 47), 'right': (x + 260, y + 47)}[side]

    fid = flow['id']
    svg = [f'<svg class="flow-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title-{fid}">',
           f'<title id="title-{fid}">{esc(flow["title"])}：所有节点均可点击查看输入输出</title>',
           f'<defs><marker id="arrow-{fid}" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="context-stroke"/></marker></defs>']
    for i, edge in enumerate(flow['edges']):
        a = port(nodes[edge['from']], edge.get('from_side', 'bottom'))
        b = port(nodes[edge['to']], edge.get('to_side', 'top'))
        if edge.get('via'):
            points = [a] + [tuple(p) for p in edge['via']] + [b]
        elif a[0] == b[0] or a[1] == b[1]:
            points = [a, b]
        else:
            middle = (a[1] + b[1]) / 2
            points = [a, (a[0], middle), (b[0], middle), b]
        path = 'M' + ' L'.join(f'{x},{y}' for x, y in points)
        segments = [(math.dist(p, q), p, q) for p, q in zip(points, points[1:])]
        _, p, q = max(segments, key=lambda s: s[0])
        lx, ly = edge.get('label_at', [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2 - 9])
        svg.append(f'<g class="edge" data-from="{esc(edge["from"])}" data-to="{esc(edge["to"])}"><path d="{path}" marker-end="url(#arrow-{fid})"/><text x="{lx}" y="{ly}" text-anchor="middle">{esc(edge["label"])}</text></g>')
    for n in nodes.values():
        x, y = box(n); nid = n['id']; decision = n.get('kind') == 'decision'
        shape = f'<path d="M{x+18} {y}H{x+242}L{x+260} {y+47}L{x+242} {y+94}H{x+18}L{x} {y+47}Z"/>' if decision else f'<rect x="{x}" y="{y}" width="260" height="94" rx="{30 if n.get("kind")=="terminal" else 12}"/>'
        svg.append(f'<a href="#step-{esc(nid)}" class="node {esc(n.get("kind", "step"))}" data-node="{esc(nid)}" aria-label="{esc(nid + " " + n["title"])}">{shape}<text class="node-id" x="{x+20}" y="{y+24}">{esc(nid)}</text><text class="node-title" x="{x+20}" y="{y+49}">{esc(n["title"])}</text><text class="node-sub" x="{x+20}" y="{y+74}">{esc(n.get("subtitle", ""))}</text></a>')
    svg.append('</svg>')
    return ''.join(svg)


def render(data):
    nav = '<a href="#overview">01 项目与边界</a><a href="#technology">02 技术背景</a>'
    nav += ''.join(f'<a href="#{esc(f["id"])}">{i+3:02} {esc(f["short_title"])}</a>' for i, f in enumerate(data['flows']))
    nav += '<a href="#examples">贯穿实例</a><a href="#modules">模块覆盖</a><a href="#verification">验证与限制</a><a href="#sources">源码依据</a>'
    body = [f'<section id="overview"><p class="kicker">01 / PROJECT & SCOPE</p><h2>先知道它做什么，再看它怎样做</h2>{paras(data["summary"])}<div class="callout"><strong>分析边界</strong>{paras(data["scope"])}</div>']
    if data.get('config'):
        body.append('<h3>贯穿全篇的教学配置</h3>' + code(data['config']))
    body.append('</section><section id="technology"><p class="kicker">02 / TECHNICAL BACKGROUND</p><h2>技术背景，落到代码里</h2><div class="tech-grid">')
    for t in data.get('technologies', []):
        body.append(f'<article class="tech"><span class="tag">{esc(t["name"])}</span><h3>{esc(t["problem"])}</h3>{paras(t["principle"])}<p><strong>本项目：</strong>{esc(t["usage"])}</p><p class="muted">{esc(t.get("limit", ""))}</p></article>')
    body.append('</div></section>')
    example_options = '<option value="">总览 · 显示所有路径</option>' + ''.join(f'<option value="{esc(e["id"])}">{esc(e["title"])}</option>' for e in data['examples'])
    for flow in data['flows']:
        fid = flow['id']
        body.append(f'<section id="{esc(fid)}" class="flow-section"><p class="kicker">EXECUTION MAP / {esc(fid.upper())}</p><h2>{esc(flow["title"])}</h2>{paras(flow["intro"])}<div class="flow-panel"><div class="toolbar"><label>选择路径 <select class="case-select" data-flow="{esc(fid)}">{example_options}</select></label><div class="zoom-controls"><button type="button" data-zoom="-1" aria-label="缩小流程图">−</button><button type="button" data-zoom="reset">适合宽度</button><button type="button" data-zoom="1" aria-label="放大流程图">＋</button></div></div><div class="legend"><span>圆角框：处理</span><span>切角框：判断</span><span>高亮线：本例路径</span><span>点击节点：查看下方详情</span><span>长图可滚动；小屏可点＋放大</span></div><div class="graph-scroll">{graph(flow)}</div><p class="path-status" aria-live="polite">完整流程 · 每条箭头已标注条件或数据</p></div><h3 class="detail-heading">逐步拆解 · 输入、处理、输出与实例</h3><div class="step-list">')
        for n in flow['nodes']:
            linked = n.get('subflow')
            extra = f'<a class="subflow-link" href="#{esc(linked)}">展开子流程 →</a>' if linked else ''
            detail_list = ''.join('<li>' + esc(s) + '</li>' for s in n['details'])
            body.append(f'<article class="step-card" id="step-{esc(n["id"])}"><div class="step-title"><span class="step-number">{esc(n["id"])}</span><h3>{esc(n["title"])}</h3>{extra}</div><p>{esc(n.get("purpose", n.get("subtitle", "")))}</p><div class="io-grid"><div><span class="field-label">INPUT / 输入</span>{code(n["input"])}</div><div><span class="field-label">OUTPUT / 输出</span>{code(n["output"])}</div></div><h4>内部如何处理</h4><ol>{detail_list}</ol><div class="example-note"><strong>具体例子</strong>{paras(n["example"])}</div><div class="live-io" aria-live="polite"></div><p class="evidence">源码依据：{source_links(n["sources"], data["sources"])}</p></article>')
        body.append('</div></section>')
    body.append('<section id="examples"><p class="kicker">WALKTHROUGHS</p><h2>把同一个例子，从头走到尾</h2><p>每个案例保留具体输入输出。上方流程图可以选择对应案例查看路径；下面的数据即使关闭 JavaScript 也可阅读。</p>')
    for e in data['examples']:
        body.append(f'<article class="example-card" id="example-{esc(e["id"])}"><span class="tag">{esc(e["provenance"])}</span><h3>{esc(e["title"])}</h3>{paras(e["description"])}<div class="io-grid"><div><h4>案例输入</h4>{code(e["input"])}</div><div><h4>案例输出 / 可确定结果</h4>{code(e["output"])}</div></div>')
        for fid, visits in e['paths'].items():
            trace = ' → '.join(v['node'] for v in visits)
            rows = ''.join(f'<tr><td><a href="#step-{esc(v["node"])}">{esc(v["node"])}</a></td><td>{code(v["input"])}</td><td>{code(v["output"])}</td></tr>' for v in visits)
            body.append(f'<details><summary>{esc(fid)} · {esc(trace)}</summary><div class="table-scroll"><table><thead><tr><th>步骤</th><th>本步输入</th><th>本步输出</th></tr></thead><tbody>{rows}</tbody></table></div></details>')
        body.append('</article>')
    if data.get('math_notes'):
        body.append('<div class="callout"><h3>可以手算的数值例子</h3>')
        for item in data['math_notes']:
            body.append(f'<h4>{esc(item["title"])}</h4>{paras(item["explanation"])}{code(item["values"])}')
        body.append('</div>')
    body.append('</section><section id="modules"><p class="kicker">MODULE INVENTORY</p><h2>模块职责与覆盖范围</h2>')
    for m in data['modules']:
        body.append(f'<article class="module-card"><div class="step-title"><h3>{esc(m["name"])}</h3><span class="tag">{esc(m["status"])}</span></div><p class="muted">{esc(m["location"])}</p><div class="io-grid"><div><strong>输入</strong>{paras(m["input"])}</div><div><strong>输出</strong>{paras(m["output"])}</div></div>{paras(m["details"])}<p><strong>例子：</strong>{esc(m["example"])}</p></article>')
    body.append('</section>')
    if data.get('paper_mapping'):
        body.append('<section id="paper"><h2>论文与实现对照</h2>' + paras(data['paper_mapping']) + '</section>')
    body.append('<section id="verification"><p class="kicker">EVIDENCE & LIMITATIONS</p><h2>哪些已确认，哪些仍需运行</h2><div class="verification-grid">')
    for v in data['verification']:
        body.append(f'<article><span class="tag">{esc(v["status"])}</span><h3>{esc(v["title"])}</h3>{paras(v["detail"])}</article>')
    body.append('</div><h3>适用边界</h3>' + paras(data.get('limitations', [])) + '</section><section id="sources"><p class="kicker">SOURCE INDEX</p><h2>可核对的源码依据</h2><p>代码为固定版本的原文摘录，保留原行号。所有外部链接仅作引用；阅读本报告无需联网。</p>')
    for s in data['sources']:
        url = s.get('url', '')
        link = f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">查看固定版本源码 ↗</a>' if url.startswith('https://') else ''
        body.append(f'<details class="source-card" id="source-{esc(s["id"])}"><summary>{esc(s["label"])}</summary>{paras(s.get("note", ""))}{code(s["code"])}{link}</details>')
    body.append('</section>')
    payload = json.dumps({'examples': data['examples']}, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    mapping = {
        'TITLE': esc(data['title']), 'SUBTITLE': esc(data.get('subtitle', '从源码到理解')), 'VERSION': esc(data['version']),
        'NAV': nav, 'BODY': ''.join(body), 'PAYLOAD': payload,
        'COUNTS': esc(f'{len(data["flows"])} 张流程图 · {sum(len(f["nodes"]) for f in data["flows"])} 个步骤 · {len(data["examples"])} 组实例'),
        'FOOTER': esc(data.get('footer', 'Generated with code-to-report · Structure checks do not prove semantic correctness.')),
    }
    template = (ROOT / 'assets' / 'report-template.html').read_text(encoding='utf-8')
    # Substitute once: report content may legitimately contain template-looking text.
    return re.sub(r'@@([A-Z]+)@@', lambda m: mapping[m[1]], template)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text(encoding='utf-8-sig'))
        errors = validate(data)
        if errors:
            raise ValueError('\n'.join(errors))
        if args.validate_only:
            print('Structure and references: valid. Semantics require source review.')
            return
        if args.output is None:
            parser.error('Use --output FILE or --validate-only')
        result = render(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding='utf-8')
        print('Generated: ' + str(args.output))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, 'Report error: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
