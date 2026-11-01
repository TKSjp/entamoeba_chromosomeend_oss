"""Step09: write readme_<genome>.html (methods, parameters, tool versions, result summary, file index)
and the index page readme.html that links the readme / viewer of every configured genome.

All numbers in the page are read from result/<genome>/ so the page always matches the latest run.
"""
import datetime
import html
import os

import numpy as np
import pandas as pd

import yaml

from common import CONFIG_DIR, Logger, available_genomes, load_config, path, read_fasta, tool_version, window_label

cfg = load_config()
log = Logger(cfg, "STEP09_readme")
log.start("readme_<genome>.html and readme.html (index)")
R = path(cfg, cfg["result_dir"])
ROOT = cfg["_root"]
labs = [window_label(k) for k in cfg["windows_kb"]]
primary = window_label(cfg["cluster"]["similarity_scale_kb"])
E = html.escape
CLS_JA = {"3p_terminal": "3′末端", "5p_terminal": "5′末端", "interior": "染色体内部"}

stats = pd.read_csv(f"{R}/01_qc/chr_stats.tsv", sep="\t")
bnd = pd.read_csv(f"{R}/02_kmer/terminal_boundary.tsv", sep="\t")
types = pd.read_csv(f"{R}/04_cluster/end_types.tsv", sep="\t")
msai = pd.read_csv(f"{R}/04_cluster/msa_index.tsv", sep="\t")
feats = pd.read_csv(f"{R}/05_annotation/features.tsv", sep="\t")
on = pd.read_csv(f"{R}/06_terminal_element/element_on_ends.tsv", sep="\t", keep_default_na=False)
irs = pd.read_csv(f"{R}/06_terminal_element/element_inverted_repeats.tsv", sep="\t")
egh = pd.read_csv(f"{R}/06_terminal_element/element_genome_hits.tsv", sep="\t")
ih = pd.read_csv(f"{R}/03_similarity/internal_hits.tsv", sep="\t")
element = list(read_fasta(f"{R}/06_terminal_element/terminal_element.fna").values())[0]
ec = cfg["terminal_element"]
mic = cfg["micro"]
mseg = pd.read_csv(f"{R}/07_micro/segments.tsv", sep="\t", keep_default_na=False)
mtyp = pd.read_csv(f"{R}/07_micro/micro_types.tsv", sep="\t", keep_default_na=False)
mjs = __import__("json").load(open(f"{R}/07_micro/micro_viewer.json"))

N = len(stats)
has_el = on[on.has_element == 1].copy()
has_el["overhang_bp"] = has_el.overhang_bp.astype(int)
capped = has_el[has_el.overhang_bp <= ec["capped_max_overhang"]]
ext = has_el[has_el.overhang_bp > ec["capped_max_overhang"]]
no_el = on[on.has_element == 0]
cons = bnd[bnd.conserved_len > 0]
full = egh[(egh.qcov >= ec["full_copy_qcov"]) & (egh.pident >= ec["full_copy_min_pident"])]
divh = egh[(egh.qcov >= ec["full_copy_qcov"]) & (egh.pident < ec["full_copy_min_pident"])]  # diverged full-length homologs
trna = feats[feats.type == "tRNA"].copy()
trna["aa"] = trna.name.str.extract(r"(tRNA-[^\s]+)")[0]
trna_by = trna.groupby(["chrom", "aa"]).size().reset_index(name="n")

versions = {
    "python": tool_version("python --version"),
    "BLAST+": tool_version("blastn -version"),
    "MAFFT": tool_version("mafft --version"),
    "TRF": tool_version("conda list -n " + cfg["conda_env"] + " 2>/dev/null | awk '$1==\"trf\"{print \"trf \" $2}'"),
    "tRNAscan-SE": tool_version("tRNAscan-SE -h 2>&1 | sed -n 2p"),
    "numpy / pandas / scipy": f"{np.__version__} / {pd.__version__} / {__import__('scipy').__version__}",
}


def rel(p):
    return os.path.relpath(p, ROOT)


def link(p, text=None):
    r = rel(p) if os.path.isabs(p) else p
    return f'<a href="{E(r)}">{E(text or r)}</a>'


def table(df, num=()):
    h = "<table><thead><tr>" + "".join(f"<th>{E(str(c))}</th>" for c in df.columns) + "</tr></thead><tbody>"
    for r in df.itertuples(index=False):
        h += "<tr>" + "".join(f'<td class="{"num" if c in num else ""}">{E(str(v))}</td>' for c, v in zip(df.columns, r)) + "</tr>"
    return h + "</tbody></table>"


def type_counts(lab):
    c = pd.read_csv(f"{R}/04_cluster/clusters_{lab}.tsv", sep="\t").type.value_counts()
    ks = sorted(c.index, key=lambda t: (t == "U", len(t), t))
    return ", ".join(f"{k}={c[k]}" for k in ks)


def members(t):
    return ", ".join(types[types.end_type == t].chrom)


type_rows = pd.DataFrame([(lab, type_counts(lab)) for lab in labs], columns=["窓", "タイプ（本数）"])
prim_rows = pd.DataFrame([(t, (types.end_type == t).sum(), members(t))
                          for t in sorted(types.end_type.unique(), key=lambda t: (t == "U", len(t), t))],
                         columns=["タイプ", "本数", "染色体"])
el_tbl = on[["chrom", "end_type", "has_element", "dist_from", "dist_to", "pident", "qcov", "overhang_bp", "overhang_seq"]] \
    .rename(columns={"chrom": "染色体", "end_type": f"タイプ({primary})", "has_element": "エレメント", "dist_from": "3′端から(始)",
                     "dist_to": "3′端から(終)", "pident": "一致率%", "qcov": "被覆率", "overhang_bp": "はみ出し(bp)",
                     "overhang_seq": "はみ出し配列"})
msa_tbl = msai[["window", "group", "n_seq", "aln_length", "mean_majority_frac", "runtime_s"]] \
    .rename(columns={"window": "窓", "group": "グループ", "n_seq": "配列数", "aln_length": "整列長", "mean_majority_frac": "平均保存度",
                     "runtime_s": "計算時間(s)"})
bnd_tbl = bnd.merge(types[["chrom", "end_type"]], on="chrom")[
    ["chrom", "end_type", "terminal_offset", "conserved_end", "conserved_len", "max_share", "interior_copy_frac_in_region"]] \
    .rename(columns={"chrom": "染色体", "end_type": "タイプ", "terminal_offset": "オフセット", "conserved_end": "内側端",
                     "conserved_len": "保存長", "max_share": "最大共有本数", "interior_copy_frac_in_region": "内部コピー率"})
ir_tbl = irs[["stem_len", "loop_len", "dist_from", "dist_to", "hairpin_seq"]] \
    .rename(columns={"stem_len": "ステム", "loop_len": "ループ", "dist_from": "3′端から(始)", "dist_to": "3′端から(終)", "hairpin_seq": "配列"})


def files_of(step):
    d = f"{R}/{step}"
    out = []
    for root, _, fs in os.walk(d):
        for f in sorted(fs):
            if f.endswith((".log", ".raw")) or f.startswith("."):
                continue
            p = os.path.join(root, f)
            out.append(p)
    return out


def file_list(step, limit=12):
    fs = files_of(step)
    li = "".join(f"<li>{link(p)} <span class='muted'>({os.path.getsize(p) / 1e3:,.0f} kB)</span></li>" for p in fs[:limit])
    if len(fs) > limit:
        li += f"<li class='muted'>…ほか {len(fs) - limit} ファイル（{E(rel(f'{R}/{step}'))}/）</li>"
    return f"<ul class='files'>{li}</ul>"


def cfg_block(key):
    v = cfg[key]
    if isinstance(v, dict):
        return "<code>" + E(", ".join(f"{k}={vv}" for k, vv in v.items())) + "</code>"
    return f"<code>{E(str(v))}</code>"


el_marked = element
oc = cfg.get("orient", {})
ori_fn = f"{R}/00_orient/orientation.tsv"
ori = pd.read_csv(ori_fn, sep="\t", keep_default_na=False) if oc.get("enabled") and os.path.exists(ori_fn) else None
ir0 = irs.iloc[0] if len(irs) else None

STEPS = [
    ("00", "除外配列の除去とコンティグの向き判定", "00_orient_contigs.py", "00_orient",
     f"exclude_contigs に指定した配列（{', '.join(cfg.get('exclude_contigs') or []) or 'なし'}、プラスミド）を取り除き、以降のステップでは使わない。"
     f"基準染色体 {oc.get('ref_chrom', '-')} の 3′末端 {oc.get('ref_bp', '-')} bp を末端モチーフとし（基準の向きを正とする）、各コンティグの両端 "
     f"{oc.get('search_window_kb', '-')} kb（5′側は逆相補にして 3′末端の向きにそろえる）を BLASTN で検索。モチーフの {oc.get('min_qcov', '-')} 以上を覆い、"
     f"3′端が末端から {oc.get('max_dist_bp', '-')} bp 以内の同方向ヒットがあれば「その端にモチーフあり」。3′側のみ = そのまま、5′側のみ = 反転、両側 = 両端、"
     f"なし = なし。apply = {oc.get('apply')}（true のとき反転判定のコンティグを逆相補にする）。除外または反転を行った FASTA を Step01 以降で使用。",
     ["exclude_contigs", "orient"]),
    ("01", "QC と 3′末端窓の切り出し", "01_qc_extract_ends.py", "01_qc",
     f"各配列の長さ・GC・N・ソフトマスク（小文字）率を集計し、3′末端から {', '.join(labs)} の窓を切り出す（染色体が窓より短い場合は全長）。"
     "以降の座標は <b>3′末端からの距離（最末端の塩基 = 1）</b> で表す。", ["windows_kb"]),
    ("02", "染色体間・染色体内の類似（k-mer）と末端保存領域", "02_kmer_sharing.py", "02_kmer",
     f"最大窓（{labs[-1]}）内の各塩基について、その塩基を覆う canonical {cfg['kmer']['k']}-mer（両鎖合算・完全一致）の出現を数える。"
     "<b>染色体間の類似</b>（inter_chroms）＝その k-mer をどこかにもつ他の染色体の本数、<b>染色体内の類似</b>（intra_copies）＝同じ染色体の別の位置（末端窓内の反復も含む全長）にあるコピー数。"
     f"3-mer の種類が {cfg['kmer']['low_complexity_min_trimers']} 未満の低複雑度 k-mer（ポリT・2 塩基反復など）はこの 2 つから除外。各塩基の値は重なる k-mer の最大値。"
     "あわせて他の何本の染色体の 3′末端窓に存在するか（end_share、染色体間の類似のうち相手が 3′末端のもの）を数え、end_share ≥ min_other_chrs が "
     "min_run bp 以上連続する領域が 3′末端から max_offset 以内に始まれば「末端保存領域」とし、max_gap を超える非保存区間が現れるまで内側へ伸ばす。",
     ["kmer"]),
    ("03", "総当たり類似度（BLASTN）とゲノム内コピー", "03_pairwise_similarity.py", "03_similarity",
     "各窓サイズで 3′末端窓どうしを all-vs-all BLASTN（megablast）。ペア (i, j) の類似度 = i の窓のうち j への HSP で覆われる同一塩基数 / i の窓長、を双方向平均した値。"
     f"さらに末端 {cfg['blast']['internal_query_kb']} kb をゲノム全体に BLASTN し、ヒット先を「他の 3′末端」「5′末端」（各 {cfg['blast']['internal_exclude_kb']} kb 以内）「内部」に分類。",
     ["blast"]),
    ("04", "末端タイプ分けと MAFFT 多重整列", "04_cluster_msa.py", "04_cluster",
     f"距離 = 1 − 類似度 で平均連結法の階層クラスタリング（最適葉順序）。距離 {cfg['cluster']['distance_cut']} で切って 2 本以上のクラスタを T1, T2…（大きい順）、単独を U とする。"
     f"主タイプは {primary} 窓の結果。各窓サイズで「全染色体」と「主タイプごと」に MAFFT で整列（向きは染色体のまま、3′末端が右端）。",
     ["cluster", "mafft"]),
    ("05", "末端領域の構成要素の注釈", "05_annotate_elements.py", "05_annotation",
     f"最大窓に対し TRF（縦列反復）、tRNAscan-SE（真核モデル）、ソフトマスク区間、自己 BLASTN による直列／逆位反復、GC（{cfg['annot']['gc_bin_bp']} bp ビン）、"
     f"最末端 {cfg['annot']['terminal_motif_bp']} bp の塩基組成を求め、ゲノム座標と 3′末端距離の両方で出力。", ["trf", "trnascan", "annot"]),
    ("06", "共通末端エレメントの定義とマッピング", "06_terminal_element.py", "06_terminal_element",
     f"最大タイプの {window_label(ec['seed_window_kb'])} 整列のコンセンサス 3′側 {ec['seed_bp']} bp（仮のエレメント）でエレメントをもつ末端を探し、"
     f"その末端 {ec['core_probe_kb']} kb を仮のエレメントの 3′端で揃えて MAFFT 整列。変異サイト（Step07 と同じ基準）の出現を 3′端から見て、"
     f"2 区間の二項尤度が最大になる位置（変化点、各区間 ≥{ec['core_min_seg_bp']} bp）までを「保存された中心部」とし、そのコンセンサスを末端エレメントとする。"
     f"内部の完全一致逆向き反復（ステム ≥{ec['min_stem']} bp、ループ ≤{ec['max_loop']} bp）を列挙。"
     f"エレメントを各 3′末端 {ec['search_window_kb']} kb とゲノム全体に BLASTN（blastn task, dust off）し、エレメントの 3′端を含み、エレメントの "
     f"{ec.get('element_min_qcov', 0):.0%} 以上を覆うヒットがある末端を「エレメントあり」とする。エレメント 3′端より外側の配列長を「はみ出し」として算出。",
     ["terminal_element"]),
    ("07", f"ミクロ構造（保存された中心部 {mjs['coreBp']:,} bp＋余白のモジュール分割と分類）", "07_micro_structure.py", "07_micro",
     f"エレメントをもつ末端を、エレメント 3′端で揃えた {mjs['windowBp']:,} bp（中心部＋{mic['window_margin_bp']} bp）で MAFFT L-INS-i 整列（エレメントより外側の配列は extension として別扱い）。"
     "中心部の中だけをモジュールと可変領域に分け、中心部より内側は「外側の領域」（O）としてまとめる。"
     f"列支持率（多数派塩基をもつ配列の割合）< {mic['core_support']} の列を変異サイトとし（ただし塩基をもつ末端が {mic['ignore_insertion_below']:.0%} 未満の挿入列は変異サイトにせず、その末端の挿入として扱う）、{mic['link_bp']} bp 以内のサイトをつないで可変領域（V1, V2…。{mic.get('min_variable_columns', 1)} 列未満にしか及ばない孤立した SNP は区切りにせずモジュール内の置換として扱う）、"
     f"その間の {mic['min_module_bp']} bp 以上の区間を保存モジュール（T1, T2…）とする（番号は 3′側から）。{mic['shared_search_min_bp']} bp 以上の可変領域と外側の領域の中で平滑化支持率 ≥{mic['shared_support']} の区間は"
     f"部分共有ブロック（S1, S2…）、同一塩基 {mic['homopolymer_min']} 以上の連続はホモポリマー（PolyT など）。各染色体×セグメントで長さ・一致率・置換/挿入/欠失を求め、"
     f"配列を MSA 一致率 ≥{mic['variant_identity']} でバリアント（a, b…）にまとめる。中心部の一致率でファミリー（A, B…、閾値 {mic['family_identity']}）、"
     "中心部のセグメントのバリアントの組合せでミクロ型（A1, A2…）を定義。エレメントのない末端ではモジュール配列を局所アラインメントで探索。", ["micro"]),
    ("08", "インタラクティブ HTML ビューア", "08_build_viewer.py", "08_viewer",
     "Step01–07 の結果を 1 つの JSON にまとめ gzip + base64 で埋め込んだ単一 HTML（外部ライブラリ・ネットワーク不要）。", []),
    ("09", "readme の生成", "09_build_readme.py", None, f"この文書（readme_{cfg['genome_label']}.html）と全ゲノムの目次 readme.html。数値はすべて {cfg['result_dir']}/ から読み込んで生成。", []),
]

steps_html = ""
for num, title, script, d, desc, keys in STEPS:
    params = "".join(f"<div class='param'><span class='k'>{E(k)}</span> {cfg_block(k)}</div>" for k in keys)
    steps_html += f"""
<section class="step" id="step{num}">
  <h3><span class="num">Step {num}</span>{E(title)}</h3>
  <p class="script">スクリプト: {link('script/' + script)}　出力: {E(rel(f'{R}/{d}') + '/') if d else E(f"readme_{cfg['genome_label']}.html")}</p>
  <p>{desc}</p>
  {params}
  {file_list(d) if d else ''}
</section>"""

findings = f"""
<ol class="findings">
<li><b>共通の末端エレメントがある。</b> {N} 本中 <b>{len(has_el)}</b> 本の 3′末端 {ec['search_window_kb']} kb に {len(element)} bp の末端エレメント（一致率 {has_el.pident.astype(float).min():.0f}–{has_el.pident.astype(float).max():.0f}%）が同じ向きで存在し、
うち <b>{len(capped)}</b> 本ではエレメントの 3′端から {ec['capped_max_overhang']} bp 以内（大半は 0–2 bp）で配列が終わる。</li>
<li><b>このエレメントは末端に特異的。</b> エレメントの {ec['full_copy_qcov']:.0%} 以上を一致率 {ec['full_copy_min_pident']}% 以上で覆う完全長コピーは {len(full)} 個で、
{"・".join(f"{CLS_JA[k]} {v} 個" for k, v in full.subject_class.value_counts().items())}。
{f"ほかに一致率 {divh.pident.min():.0f}–{divh.pident.max():.0f}% の分岐した完全長の類縁配列が {len(divh)} 個（{divh.sseqid.nunique()} 本の染色体、" + "・".join(f"{CLS_JA[k]} {v} 個" for k, v in divh.subject_class.value_counts().items()) + "）ある。" if len(divh) else ""}
部分ヒットは {len(egh) - len(full) - len(divh)} 個
（クエリ位置 {int(egh[egh.qcov < ec['full_copy_qcov']].qstart.min()) if (egh.qcov < ec['full_copy_qcov']).any() else '-'}–{int(egh[egh.qcov < ec['full_copy_qcov']].qend.max()) if (egh.qcov < ec['full_copy_qcov']).any() else '-'} bp）。</li>
<li><b>エレメント内のヘアピン候補。</b> 完全一致の逆向き反復（ステム ≥{ec['min_stem']} bp、ループ ≤{ec['max_loop']} bp）を {len(irs)} 個検出。
{"最長はエレメント 3′端から " + str(ir0.dist_from) + "–" + str(ir0.dist_to) + " bp の、ステム " + str(ir0.stem_len) + " bp・ループ " + str(ir0.loop_len) + " bp の <code>" + E(ir0.hairpin_seq) + "</code>。" if ir0 is not None else ""}
{"最も 3′端に近いものは " + str(irs.dist_from.min()) + " bp から。" if len(irs) else ""}</li>
<li><b>外側に配列が続く末端。</b> {", ".join(f"{r.chrom}（{r.overhang_bp} bp）" for r in ext.itertuples()) or "なし"} では、エレメントの外側にさらに配列が続く。</li>
<li><b>エレメントのない末端は {len(no_el)} 本</b>（{", ".join(no_el.chrom)}）。
{"tRNA 縦列アレイを含むもの: " + ", ".join(f"{r.chrom}（{r.aa} ×{r.n}）" for r in trna_by[trna_by.n >= 3].itertuples()) + "。" if (trna_by.n >= 3).any() else ""}</li>
<li><b>ミクロ構造。</b> 保存された中心部 {mjs['coreBp']:,} bp は、
保存モジュール {", ".join(mseg[(mseg.kind == "module")].name[::-1])} と可変領域からなり、{len(mjs["families"])} ファミリー・{len(mjs["types"])} ミクロ型に分類された
（最大ファミリー {mjs["families"][0]["family"]}: {len(mjs["families"][0]["members"])} 本）。</li>
<li><b>サブテロメア領域も広く共有される。</b> k-mer 共有で定義した末端保存領域は {len(cons)} 本で検出され、長さの中央値は {int(cons.conserved_len.median()):,} bp（最大 {int(cons.conserved_len.max()):,} bp）。
{primary} 窓の主タイプは {", ".join(f"{t} {(types.end_type == t).sum()} 本" for t in sorted(types.end_type.unique(), key=lambda t: (t == "U", len(t), t)))}。</li>
<li><b>末端 {cfg['blast']['internal_query_kb']} kb 全体はゲノム内に多数のコピーをもつ</b>（内部 {int((ih.subject_class == 'interior').sum()):,}、5′末端 {int((ih.subject_class == '5p_terminal').sum()):,}、他の 3′末端 {int((ih.subject_class == '3p_terminal').sum()):,} ヒット）。
エレメントより内側には分散反復配列が多く含まれる。</li>
</ol>
<p class="muted">※ 上記はアセンブリ配列だけから得た観察です。末端の「はみ出し」やエレメントの欠如は、組み立ての途切れ（コンティグ末端）や誤りでも生じます。リードの被覆やロングリードで末端を確認することを推奨します。</p>
"""

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--accent:#2a78d6;--wash:rgba(42,120,214,.08)}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--accent:#3987e5;--wash:rgba(57,135,229,.15)}}
:root[data-theme=dark]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--accent:#3987e5;--wash:rgba(57,135,229,.15)}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.7 system-ui,-apple-system,"Hiragino Sans","Segoe UI",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:40px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--grid)}h3{font-size:16px;margin:0 0 6px}
a{color:var(--accent)}code,.mono{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:13px}
.muted{color:var(--muted);font-size:13px}.lead{color:var(--ink2)}
nav.toc{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:10px 18px;margin:18px 0}
nav.toc a{margin-right:14px;white-space:nowrap}
.cta{display:inline-block;background:var(--ink);color:var(--page);padding:9px 16px;border-radius:9px;text-decoration:none;font-weight:600;margin:6px 0}
.flow{display:flex;flex-wrap:wrap;gap:6px;align-items:stretch;margin:12px 0}
.flow .box{background:var(--surface);border:1px solid var(--ring);border-radius:9px;padding:8px 10px;font-size:12.5px;flex:1 1 120px;min-width:120px}
.flow .box b{display:block;font-size:11.5px;color:var(--muted)}
.flow .arr{align-self:center;color:var(--muted)}
.step{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 18px;margin:12px 0}
.step .num{display:inline-block;background:var(--ink);color:var(--page);border-radius:6px;padding:0 7px;font-size:12px;margin-right:8px;vertical-align:2px}
.step .script{color:var(--ink2);font-size:13px;margin:0 0 6px}
.param{font-size:13px;color:var(--ink2);margin:2px 0}.param .k{display:inline-block;min-width:110px;color:var(--muted)}
ul.files{font-size:13px;margin:8px 0 0;padding-left:20px;columns:2;column-gap:28px}@media (max-width:760px){ul.files{columns:1}}
.tw{overflow-x:auto;border:1px solid var(--grid);border-radius:8px;margin:8px 0 16px;max-height:520px}
table{border-collapse:collapse;font-size:12.5px;width:100%}th,td{padding:4px 9px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}
th{position:sticky;top:0;background:var(--surface);color:var(--ink2);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}
.findings li{margin:6px 0}.seq{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;word-break:break-all;background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:10px 12px}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:2px 16px;font-size:13.5px}.kv div:nth-child(odd){color:var(--muted)}
pre{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:10px 12px;overflow-x:auto;font-size:13px}
"""

if ori is not None:
    oc_cnt = ori.call.value_counts()
    flips = ori[ori.call == "flip"].chrom.tolist()
    orient_summary = (f"<p><b>向き判定（Step00）:</b> 基準 {E(oc['ref_chrom'])} の 3′末端モチーフで、そのまま {oc_cnt.get('as_is', 0)}・"
                      f"反転 {oc_cnt.get('flip', 0)}・両端 {oc_cnt.get('both', 0)}・なし {oc_cnt.get('none', 0)} 本。"
                      + (f"反転したコンティグ: {E(', '.join(flips))}。" if oc.get('apply') and flips else
                         ("反転は適用していない（判定のみ）。" if not oc.get('apply') else "反転が必要なコンティグはなかった。")) + "</p>")
    ot = ori[["chrom", "length", "call", "flipped", "3p_found", "3p_pident", "3p_qcov", "3p_dist_to_end", "5p_found", "5p_pident", "5p_qcov", "5p_dist_to_end"]].rename(columns={
        "chrom": "コンティグ", "length": "長さ", "call": "判定", "flipped": "反転適用", "3p_found": "3′側", "3p_pident": "3′一致率%", "3p_qcov": "3′被覆",
        "3p_dist_to_end": "3′端までの距離", "5p_found": "5′側", "5p_pident": "5′一致率%", "5p_qcov": "5′被覆", "5p_dist_to_end": "5′端までの距離"})
    orient_section = f"""<h2 id="orient">コンティグの向き判定（Step00）</h2>
<p>基準 {E(oc['ref_chrom'])} の 3′末端 {oc['ref_bp']} bp（{link(f'{R}/00_orient/reference_motif.fna', 'FASTA')}）を各コンティグの両端で検索した結果。
判定: as_is = 3′側のみ、flip = 5′側のみ（逆相補にすると 3′側）、both = 両端、none = なし。距離はモチーフの 3′端からコンティグ末端までの塩基数。</p>
<div class="tw">{table(ot, num=("長さ", "3′一致率%", "3′被覆", "3′端までの距離", "5′一致率%", "5′被覆", "5′端までの距離"))}</div>"""
else:
    orient_summary = orient_section = ""

flow = "".join(
    (f'<div class="arr">→</div>' if i else "") + f'<div class="box"><b>Step {n}</b>{E(t)}</div>'
    for i, (n, t, *_rest) in enumerate(STEPS))

viewer = f"{R}/08_viewer/chrom_end_viewer.html"
page = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{E(cfg['genome_label'])} 3′末端構造解析 readme</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>{E(cfg['genome_label'])} 染色体 3′末端構造解析</h1>
<p class="lead">テロメア（テロメラーゼ型反復）をもたないと考えられる微生物ゲノムについて、染色体 3′末端の配列がどのように共通・類似しているかを調べ、末端構造を可視化・配列取得できるようにするパイプライン。</p>
<p><a class="cta" href="{E(rel(viewer))}">インタラクティブビューアを開く →</a></p>
<nav class="toc"><a href="readme.html">← 目次</a><a href="#summary">結果の要約</a><a href="#orient">向き判定</a><a href="#element">末端エレメント</a><a href="#micro">ミクロ構造</a><a href="#types">末端タイプ</a><a href="#pipeline">パイプライン</a><a href="#steps">各ステップ</a><a href="#viewer">ビューアの使い方</a><a href="#run">実行方法</a><a href="#env">環境</a><a href="#notes">注意点</a></nav>

<div class="kv">
<div>入力</div><div><code>{E(cfg['input_fasta'])}</code>（解析対象 {N} 配列, {int(stats.length.sum()):,} bp, GC {stats.length.dot(stats.GC) / stats.length.sum():.3f}, N {int(stats.N.sum())}）</div>
<div>除外</div><div>{E(", ".join(cfg.get("exclude_contigs") or [])) or "なし"}（プラスミド。Step00 で取り除き、全ステップの解析対象外）</div>
<div>対象</div><div>各配列の 3′末端（{"Step00 の向き判定で反転とされたコンティグは逆相補にした向き" if oc.get("enabled") and oc.get("apply") else "FASTA に書かれた向きのまま"}。5′末端は解析対象外）</div>
<div>設定</div><div>{link(cfg['_config_path'])}　目次: {link('readme.html')}</div>
<div>窓サイズ</div><div>{", ".join(labs)}</div>
<div>生成日</div><div>{datetime.date.today().isoformat()}　ログ: {link('log.txt')}</div>
</div>

<h2 id="summary">結果の要約</h2>
{orient_summary}
{findings}

{orient_section}
<h2 id="element">共通末端エレメント（Step06）</h2>
<p>エレメントをもつ末端を 3′端で揃えた整列で、変異サイトの密度が急に上がる位置（変化点）までの保存された中心部 {len(element):,} bp のコンセンサス（右端 = 3′末端）。{link(f'{R}/06_terminal_element/terminal_element.fna', 'FASTA')}</p>
<div class="seq">{E(el_marked)}</div>
<h3 style="margin-top:14px">エレメント内の逆向き反復（ヘアピン候補）</h3>
<div class="tw">{table(ir_tbl, num=("ステム", "ループ", "3′端から(始)", "3′端から(終)"))}</div>
<h3>各染色体の 3′末端でのエレメント</h3>
<p class="muted">はみ出し = エレメント 3′端より外側（末端側）にある塩基数。負の値は、エレメント末尾の数 bp を欠いて配列が終わることを示す。</p>
<div class="tw">{table(el_tbl, num=("3′端から(始)", "3′端から(終)", "一致率%", "被覆率", "はみ出し(bp)"))}</div>

<h2 id="micro">ミクロ構造（Step07, 保存された中心部 {mjs['coreBp']:,} bp）</h2>
<p>エレメントをもつ {len(mtyp)} 本の末端を、保存された中心部（エレメント 3′端から {mjs['coreBp']:,} bp）のセグメント構成で分類した。
ビューアの「ミクロ構造」タブで図（SVG 保存可）と配列を確認できる。</p>
<div class="seq">{" — ".join(E(f"{r.name} {r.len_min if r.len_min == r.len_max else str(r.len_min) + '–' + str(r.len_max)} bp") for r in mseg[mseg.cons_dist_from <= mjs['typeRegionBp']].itertuples())} — [3′端] — extension</div>
<h3 style="margin-top:14px">セグメント（3′側から）</h3>
<div class="tw">{table(mseg.iloc[::-1][["name", "kind", "cons_dist_from", "cons_dist_to", "cons_len", "len_min", "len_max", "n_present", "mean_identity", "n_variants", "cons_seq"]].rename(columns={"name": "セグメント", "kind": "種類", "cons_dist_from": "3′端から(始)", "cons_dist_to": "3′端から(終)", "cons_len": "コンセンサス長", "len_min": "最短", "len_max": "最長", "n_present": "保有本数", "mean_identity": "平均一致率", "n_variants": "バリアント数", "cons_seq": "コンセンサス配列"}), num=("3′端から(始)", "3′端から(終)", "コンセンサス長", "最短", "最長", "保有本数", "平均一致率", "バリアント数"))}</div>
<h3>ファミリーとミクロ型</h3>
<div class="tw">{table(pd.DataFrame([(T["type"], len(T["members"]), ", ".join(T["members"]), T["code"]) for T in mjs["types"]], columns=["ミクロ型", "本数", "染色体", "セグメントのバリアント（3′側から）"]), num=("本数",))}</div>
<p class="muted">ファミリーの平均一致率: {", ".join(f"{F['family']} {F['identity']}（n={len(F['members'])}）" for F in mjs["families"])}。
エレメントのない末端でのモジュール候補: {", ".join(f"{h['chrom']} {h['segment']}({h['strand']}) {h['dist_from']}–{h['dist_to']} bp" for h in mjs["nonElement"]["hits"]) or "なし"}。</p>

<h2 id="types">末端タイプと保存領域</h2>
<h3>主タイプ（{primary} 窓）</h3>
<div class="tw">{table(prim_rows, num=("本数",))}</div>
<h3>窓サイズごとのタイプ構成</h3>
<div class="tw">{table(type_rows)}</div>
<h3>末端保存領域（Step02, k-mer 共有）</h3>
<div class="tw">{table(bnd_tbl, num=("オフセット", "内側端", "保存長", "最大共有本数", "内部コピー率"))}</div>
<h3>MAFFT 整列の一覧（Step04）</h3>
<div class="tw">{table(msa_tbl, num=("配列数", "整列長", "平均保存度", "計算時間(s)"))}</div>

<h2 id="pipeline">パイプラインの全体像</h2>
<div class="flow">{flow}</div>
<p>パラメータはゲノムごとの設定ファイル {link(cfg['_config_path'])} にあり、各スクリプトはそれを読む（run_all.sh が環境変数 CHR_END_GENOME でゲノムを指定）。結果は {E(cfg['result_dir'])}/ に出力。ログは全ゲノム共通の {link('log.txt')} に、各行にゲノム名を付けてステップ単位で開始・終了・パラメータ・実行コマンド・要約統計が追記される。</p>

<h2 id="steps">各ステップの方法と出力</h2>
{steps_html}

<h2 id="viewer">ビューアの使い方</h2>
<p>{link(viewer)}（単一ファイル。ブラウザで直接開く。オフライン可）</p>
<ul>
<li><b>スケール</b>（{" / ".join(labs)}）は全タブ共通。「表示する染色体」で、すべて／末端エレメントをもつ染色体のみ（Step06 でエレメント 3′端を含むヒットがあるもの。スケールに依存しない）／単独型 (U) 以外（現スケールの型）を切替（全タブ共通）。トラック表示範囲・類似度行列・MAFFT 整列・タイプ色がそのスケールに切り替わる。</li>
<li><b>末端トラック</b>: 右端が 3′末端。青の帯＝染色体間の類似（その配列をもつ他の染色体の本数）、黄緑の帯＝染色体内の類似（同じ染色体内の別コピー数）。青が高く黄緑が低い＝染色体間で共通・染色体内では単コピー、両方高い＝散在反復、両方低い＝ユニーク。行（染色体）をドラッグで範囲選択 → 下のパネルで配列を FASTA 表示・コピー・保存。「同じ範囲を全染色体で」「同タイプで」「3′末端から N bp」で一括取得。目盛りのドラッグ／⌥+ホイールでズーム、ダブルクリックで全体。拡大すると塩基が表示される。「隣り合う行の相同領域をリボンで結ぶ」をオンにすると、現スケールの総当たり BLASTN の HSP（最小一致率・最小長を指定）で上下の行を帯で結ぶ（紫＝同方向、緑の交差＝逆方向、濃いほど高一致率）。結ぶ相手は並び順で決まる。</li>
<li><b>ミクロ構造</b>: エレメント 3′端で揃えた末端（{mjs['windowBp']:,} bp）の模式図。上段がコンセンサス構造（セグメント名と長さ範囲、50 bp などのスケールバー、extension は非比例の枠）、下段がファミリー・ミクロ型ごとの各染色体（実際の長さで描画、橙＝置換・紫＝挿入・黒線＝欠失）。表示範囲（ミクロ型領域／500 bp／全体 など）を切替、セグメントをクリックで配列取得、「SVG を保存」で図を保存。</li>
<li><b>MAFFT アラインメント</b>: スケール × グループ（全染色体／タイプ別）を選択。表示は「ミクロ構造で色分け」（既定: 各塩基を 3′末端からの距離で Step07 のセグメントに対応づけ、ミクロ構造タブと同じ色で塗る。エレメント 3′端から {mjs['windowBp']:,} bp（Step07 の解析窓）＋extension が対象で、それより内側とエレメントのない末端は未分類、ギャップは空白）／「コンセンサスとの差」／「塩基で色分け」を切替。横軸は基準染色体（config の viewer.msa_ref_chrom、既定 {E(cfg["viewer"]["msa_ref_chrom"])}。グループに含まれない場合は名前順で最初のエレメント保有染色体）の bp 座標で、末端エレメント 3′端 = 0、内側が負、外側が正（基準に塩基のない外側の列は列数で数える）。整列は挿入・欠失のギャップ列を含むため列数は窓長（例: 1 kb）より長くなる。「基準配列のギャップ列を詰める」をオンにすると基準の bp 座標に投影して表示（他配列の挿入は非表示）。行の高さ・行間は数値で指定。概観（保存度＋各配列）をドラッグして詳細表示の列範囲を選ぶ。コンセンサスとの差／塩基の色分けを切替。選択列（ギャップ付き・なし）、コンセンサス、整列全体を FASTA で取得。</li>
<li><b>類似性マトリクス</b>: クラスタ順のヒートマップ＋デンドログラム（破線 = タイプ分けの閾値）。セルをクリックで HSP 一覧とドットプロット。</li>
<li><b>最末端モチーフ</b>: 末端エレメント配列（ヘアピン候補に下線）と、各染色体の最末端 N bp を末端揃えで表示。赤い縦線がエレメントの 3′端。</li>
<li><b>ゲノム内コピー</b>: 末端 {cfg['blast']['internal_query_kb']} kb のヒットを全染色体上に表示。クエリ染色体・クエリ上の位置・区分で絞り込み、TSV 保存。</li>
<li><b>要約表</b>: 並べ替え・TSV 保存。行クリックでその染色体の末端 1 kb をトラック上で選択。</li>
<li>URL ハッシュで状態を指定可能: 例 <code>chrom_end_viewer.html#tab=msa&amp;scale=1kb&amp;group=T1</code>、<code>#tab=sim&amp;pair=Chr03,Chr23</code>、<code>#tab=tracks&amp;view=1-500</code>。</li>
</ul>

<h2 id="run">実行方法</h2>
<pre>conda activate {E(cfg['conda_env'])}
bash script/run_all.sh {E(cfg['genome_label'])}            # Step00-09 を順に実行（log.txt に追記）
bash script/run_all.sh {E(cfg['genome_label'])} 07 08 09   # 指定ステップのみ（MAFFT 整列は入力が同じなら再利用）
CHR_END_GENOME={E(cfg['genome_label'])} python script/02_kmer_sharing.py   # 個別実行も可

# 別のゲノムを追加する: 設定をコピーして input_fasta / genome_label / result_dir（と orient）を書き換え
cp script/config/{E(cfg['genome_label'])}.yaml script/config/NEW.yaml
bash script/run_all.sh NEW</pre>
<p class="muted">所要時間の目安（Apple M4, 8 スレッド）: Step04 の MAFFT（{labs[-1]} 全染色体）が最も長く、その他は各 1 分以内。</p>

<h2 id="env">環境とツールのバージョン</h2>
<pre>conda create -n {E(cfg['conda_env'])} -c conda-forge -c bioconda python=3.11 biopython numpy pandas scipy pyyaml blast mafft trf trnascan-se</pre>
<div class="kv">{"".join(f"<div>{E(k)}</div><div><code>{E(v)}</code></div>" for k, v in versions.items())}</div>

<h2 id="notes">注意点・限界</h2>
<ul>
<li>入力 FASTA の向きをそのまま用いる（ファイル名の rc は 3′末端が揃うよう逆相補済みと想定）。各配列の 5′端は解析していない。</li>
<li>末端構造の推定はアセンブリ配列だけに基づく。末端の欠落・はみ出し・キメラは組み立ての影響を受けうる。</li>
<li>類似度は BLASTN（megablast, e ≤ {cfg['blast']['evalue']}, HSP ≥ {cfg['blast']['min_hsp_len']} bp）で検出できる相同性に限られ、分岐した配列は過小評価される。</li>
<li>「全染色体」の MAFFT 整列には相同でない末端も含まれるため、保存度の低い列は解釈に注意。タイプ別整列の方が信頼できる。</li>
<li>k-mer は canonical（両鎖合算）の完全一致なので、染色体間・染色体内の類似はほぼ同一（おおむね 95% 以上）のコピーだけを数え、分岐したコピーは数えない。ほぼ同一の反復がアセンブリで 1 つにまとめられている場合はコピー数が過小になる。末端保存領域の境界は min_other_chrs・max_gap に依存する。</li>
<li>tRNAscan-SE は真核モデルの既定設定。tRNA 縦列アレイ（STR を伴う）の解釈は既知の知見と照合すること。</li>
</ul>
</div></body></html>
"""
fn = path(cfg, f"readme_{cfg['genome_label']}.html")
with open(fn, "w", encoding="utf-8") as fh:
    fh.write(page)
log(f"wrote {os.path.basename(fn)} ({len(page) / 1e3:.0f} kB)")

# ---------- index page: every configured genome
rows_idx = []
for g in available_genomes():
    gc_ = yaml.safe_load(open(os.path.join(CONFIG_DIR, f"{g}.yaml")))
    gR = path(cfg, gc_["result_dir"])
    rd, vw = path(cfg, f"readme_{g}.html"), f"{gR}/08_viewer/chrom_end_viewer.html"
    stats_fn, el_fn, o_fn = f"{gR}/01_qc/chr_stats.tsv", f"{gR}/06_terminal_element/element_on_ends.tsv", f"{gR}/00_orient/orientation.tsv"
    n = len(pd.read_csv(stats_fn, sep="\t")) if os.path.exists(stats_fn) else "-"
    if os.path.exists(el_fn):  # element hit that contains the element's 3' edge (= Step06 "found" count)
        ee = pd.read_csv(el_fn, sep="\t", keep_default_na=False)
        nel = int(((ee.has_element.astype(str) == "1") & (ee.has_3p_edge.astype(str) == "1")).sum())
    else:
        nel = "-"
    if os.path.exists(o_fn):
        oo = pd.read_csv(o_fn, sep="\t", keep_default_na=False)
        ostr = f"そのまま {(oo.call == 'as_is').sum()} / 反転 {(oo.call == 'flip').sum()} / 両端 {(oo.call == 'both').sum()} / なし {(oo.call == 'none').sum()}" + \
               ("（反転を適用）" if gc_.get("orient", {}).get("apply") else "（判定のみ）")
    else:
        ostr = "-"
    rows_idx.append(f"<tr><td><b>{E(g)}</b></td><td><code>{E(gc_['input_fasta'])}</code></td><td class='num'>{n}</td><td class='num'>{nel}</td><td>{E(ostr)}</td>"
                    f"<td>{link(rd, 'readme') if os.path.exists(rd) else '未実行'}</td><td>{link(vw, 'ビューア') if os.path.exists(vw) else '-'}</td>"
                    f"<td>{link(os.path.join(CONFIG_DIR, g + '.yaml'), 'config')}</td></tr>")
index = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>染色体 3′末端構造解析 目次</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>染色体 3′末端構造解析</h1>
<p class="lead">ゲノムごとの解析結果（readme とインタラクティブビューア）。方法の詳細は各 readme を参照。</p>
<div class="tw"><table><thead><tr><th>ゲノム</th><th>入力</th><th>配列数</th><th>末端エレメントあり</th><th>向き判定（Step00）</th><th>readme</th><th>ビューア</th><th>設定</th></tr></thead>
<tbody>{"".join(rows_idx)}</tbody></table></div>
<h2>実行方法</h2>
<pre>bash script/run_all.sh &lt;ゲノム&gt;            # 例: bash script/run_all.sh NA442
bash script/run_all.sh &lt;ゲノム&gt; 07 08 09   # 指定ステップのみ
bash script/run_all.sh                     # 設定済みゲノムの一覧</pre>
<p>ゲノムを追加するには <code>script/config/</code> の yaml をコピーして <code>input_fasta</code>・<code>genome_label</code>・<code>result_dir</code>（必要なら <code>orient</code>）を書き換える。ログ: {link('log.txt')}</p>
<p class="muted">更新: {datetime.date.today().isoformat()}</p>
</div></body></html>
"""
with open(path(cfg, "readme.html"), "w", encoding="utf-8") as fh:
    fh.write(index)
log(f"wrote readme.html (index of {len(rows_idx)} genomes)")
log.end()
