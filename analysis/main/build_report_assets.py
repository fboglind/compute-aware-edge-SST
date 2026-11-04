"""Rebuild tables and figures from saved metric/log snapshots; no model calls.

Run in a terminal: python build_report_assets.py
Requires NumPy and Matplotlib. The data/ directory is supplied in the package.
No resegmentation or metric-model inference is repeated. Corpus chrF2 is taken
from the saved, verified scorer output; all new comparisons are derived here.
"""
import csv
import hashlib
import json
import statistics as stats
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA, TABLES, FIGURES = ROOT / "data", ROOT / "tables", ROOT / "figures"
ORDER = [(d, m, c) for d in ("pi", "laptop") for m in ("tiny.en", "base.en") for c in (4, 8)]
COLORS = {("tiny.en", 4): "#0072B2", ("tiny.en", 8): "#009E73",
          ("base.en", 4): "#D55E00", ("base.en", 8): "#8A4F8A"}


def read(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def key(row):
    return row["device"], row["asr"], row["chunk_s"]


def label(model, seconds):
    return f"{model} / {seconds}s"


def save_csv(name, rows):
    with (TABLES / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def save_table(name, headers, rows, caption, widths=None):
    (TABLES / (name + ".md")).write_text(markdown(headers, rows) + "\n\n" + caption + "\n")
    def esc(value):
        return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")
    layout = widths or ("l" * 2 + "r" * (len(headers) - 2))
    tex = ["\\begin{table}[htbp]", "\\centering", "\\small", "\\begin{tabular}{" + layout + "}", "\\toprule",
           " & ".join(map(esc, headers)) + " " + chr(92) * 2, "\\midrule"]
    tex += [" & ".join(map(esc, row)) + " " + "\\" * 2 for row in rows]
    tex += ["\\bottomrule", "\\end{tabular}", "\\caption{" + esc(caption) + "}",
            "\\label{tab:" + name + "}", "\\end{table}"]
    (TABLES / (name + ".tex")).write_text("\n".join(tex) + "\n")


def figure_save(fig, name):
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build():
    TABLES.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    summary, scores, comet = read("main-summary.json"), read("main-scores.json"), read("comet-scores.json")
    audit, asr = read("run-audit.json"), read("asr-errors.json")
    assert summary["complete_grid"] and summary["comet_complete"]
    assert len(scores["runs"]) == 96
    assert comet["scores_json_sha256"] == hashlib.sha256((DATA / "main-scores.json").read_bytes()).hexdigest()
    assert audit["scores_sha256"] == comet["scores_json_sha256"]
    assert asr["scored_inputs_sha256"] == comet["scores_json_sha256"]
    runs = scores["runs"]
    run_audit = {r["key"]: r for r in audit["runs"]}
    asr_runs = {r["run_id"]: r for r in asr["runs"]}
    configurations = {key(r): r for r in summary["configurations"]}
    assert set(configurations) == set(ORDER)
    agreement = read("output-agreement.json")
    within = [c for r in agreement for c in r["comparisons"] if c["a"].split('/')[0] == c["b"].split('/')[0]]
    assert len(within) == 48 and all(c["same_asr"] and c["same_translation"] and c["same_cu_schedule"] for c in within)
    main, operational, uncertainty, segments, talk_rows, pairs = [], [], [], [], [], []
    for identity in ORDER:
        device, model, seconds = identity
        cfg = configurations[identity]
        mm = cfg["metrics"]
        rr = [r for r in runs if key(r) == identity]
        assert len(rr) == 12
        primary = [r for r in rr if r["repetition"] == 1]
        counts = {name: sum(asr_runs[r['key']][name] for r in primary)
                  for name in ('reference_words', 'substitutions', 'deletions', 'insertions')}
        assert counts['reference_words'] == 4686
        assert abs(100 * sum(counts[n] for n in ('substitutions', 'deletions', 'insertions')) / counts['reference_words'] - mm['wer_pct']['mean']) < 1e-9
        for rep in (1, 2):
            sub = [r for r in rr if r['repetition'] == rep]
            p = next(p for p in summary['passes'] if key(p) == identity and p['repetition'] == rep)
            assert abs(sum(r['total_processing_s'] for r in sub) / sum(r['audio_duration_s'] for r in sub) - p['rtf_processing']) < 1e-10
            com = [v for r in sub for v in comet['runs'][r['key']]['segment_scores']]
            assert len(com) == 266 and abs(stats.mean(com) - p['comet']) < 1e-10
            common = [(a,b) for r in sub for a,b in zip(r['latency']['cu']['segment_scores_s'], r['latency']['ca']['segment_scores_s']) if a is not None and b is not None]
            segments.append({'device':device, 'asr':model, 'chunk_s':seconds, 'repetition':rep,
                             'quality_segments':266, 'cu_eligible':p['latency_eligible_segments_cu'],
                             'ca_eligible':p['latency_eligible_segments_ca'], 'common_eligible':len(common),
                             'matched_ca_minus_cu_s':stats.mean(b-a for a,b in common),
                             'cu_late_words':p['words_at_or_after_wav_end_cu'], 'ca_late_words':p['words_at_or_after_wav_end_ca']})
        common_gap = stats.mean(r['matched_ca_minus_cu_s'] for r in segments if key(r) == identity)
        main.append({'device':device, 'asr':model, 'chunk_s':seconds,
                     **{name:mm[name]['mean'] for name in ('chrf2','comet','wer_pct','longyaal_cu_s','longyaal_ca_s','rtf_processing','asr_rtf','mt_rtf')},
                     'ca_minus_cu_s':mm['longyaal_ca_s']['mean'] - mm['longyaal_cu_s']['mean'],
                     'common_segment_ca_minus_cu_s':common_gap,
                     'ca_increase_pct':100*(mm['longyaal_ca_s']['mean']/mm['longyaal_cu_s']['mean']-1), **counts})
        cc = [c for r in rr for c in run_audit[r['key']]['chunks']]
        full = [c for c in cc if abs(c['chunk_duration_s'] - seconds) < 1e-6]
        operational.append({'device':device, 'asr':model, 'chunk_s':seconds,
            'max_run_rtf':max(r['rtf_processing'] for r in rr),
            'mean_first_output_s':mm['mean_first_output_s']['mean'],
            'mean_final_tail_s':mm['mean_final_tail_s']['mean'],
            'max_final_tail_s':max(r['final_tail_s'] for r in rr),
            'max_queue_s':max(c['queue_wait_s'] for c in cc),
            'max_full_chunk_queue_s':max(c['queue_wait_s'] for c in full),
            'full_chunks_queue_over_100ms':sum(c['queue_wait_s'] > .1 for c in full),
            'full_chunks':len(full), 'all_chunks':len(cc),
            'peak_sampled_replay_rss_mib':max(r['sampled_peak_replay_rss_mib'] for r in rr),
            'max_sampled_zone0_temperature_c':max(r['temperature_zone0_max_c'] for r in rr),
            'mt_limit_chunks_r1':sum(r['mt_limit_chunks'] for r in primary),
            'empty_asr_chunks_r1':sum(r['empty_asr_chunks'] for r in primary)})
        for metric, values in mm.items():
            uncertainty.append({'device':device,'asr':model,'chunk_s':seconds,'metric':metric,**values})
        for r in primary:
            talk_rows.append({'device':device,'asr':model,'chunk_s':seconds,'talk':r['talk'],
                **{name:r[name] for name in ('chrf2','wer_pct','rtf_processing','longyaal_cu_s','longyaal_ca_s')},
                'comet':stats.mean(comet['runs'][r['key']]['segment_scores']),
                **{name:asr_runs[r['key']][name] for name in counts}})
    by_key = {key(r):r for r in main}
    for model in ('tiny.en','base.en'):
        for seconds in (4,8):
            pi, lap = by_key['pi',model,seconds], by_key['laptop',model,seconds]
            pairs.append({'asr':model, 'chunk_s':seconds,
                          'pi_to_laptop_processing_ratio':pi['rtf_processing']/lap['rtf_processing'],
                          'pi_minus_laptop_ca_s':pi['longyaal_ca_s']-lap['longyaal_ca_s'],
                          'pi_minus_laptop_chrf2':pi['chrf2']-lap['chrf2'],
                          'pi_minus_laptop_comet':pi['comet']-lap['comet']})
    for name, rows in [('main-results.csv',main),('operational-results.csv',operational),
                       ('repeat-ranges.csv',uncertainty),('latency-eligibility.csv',segments),
                       ('per-talk-results.csv',talk_rows),('device-comparisons.csv',pairs)]: save_csv(name,rows)
    table_rows = [[r['device'],label(r['asr'],r['chunk_s']),f"{r['chrf2']:.2f}",f"{r['comet']:.3f}",f"{r['wer_pct']:.2f}",
                   f"{r['longyaal_cu_s']:.2f}",f"{r['longyaal_ca_s']:.2f}",f"{r['rtf_processing']:.3f}"] for r in main]
    caption = ('Main experiment over six talks (33.63 minutes, 266 reference segments). Quality is corpus chrF2, '
               'segment-mean COMET and pooled WER. Timing is the mean of two complete passes; RTF is audio-weighted '
               'processing time, excluding deliberate playback waits. CU and CA are LongYAAL in seconds, with '
               'clock-specific eligible segments. The repeated outputs were identical within each device/configuration.')
    save_table('main-results',['Device','ASR / chunk','chrF2','COMET','WER (%)','CU (s)','CA (s)','RTF'],table_rows,caption)
    save_table('operational-results',['Device','ASR / chunk','First (s)','Tail max (s)','Queue max (s)','RSS (MiB)'],
               [[r['device'],label(r['asr'],r['chunk_s']),f"{r['mean_first_output_s']:.2f}",f"{r['max_final_tail_s']:.2f}",
                 f"{r['max_queue_s']:.2f}",f"{r['peak_sampled_replay_rss_mib']:.1f}"] for r in operational],
               'First is the mean first-output time from recording onset. Tail and queue values are maxima across all 12 runs '
               'per configuration, including shorter final chunks. RSS is the maximum sampled replay process RSS; resources were '
               'sampled every five seconds plus start/end, so brief memory peaks may be missed. Loading and warm-up are outside replay timing.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','pdf.fonttype':42,'savefig.pad_inches':.12})
    fig, axes = plt.subplots(2,2,figsize=(10.4,7.0),sharex=True)
    for col,device in enumerate(('pi','laptop')):
        for row,(metric,ytitle,limits) in enumerate([('chrf2','chrF2 (higher is better)',(44.7,53.6)),('comet','COMET (higher is better)',(.54,.685))]):
            ax=axes[row,col]
            for r in main:
                if r['device']!=device: continue
                color=COLORS[r['asr'],r['chunk_s']]; cu,ca,y=r['longyaal_cu_s'],r['longyaal_ca_s'],r[metric]
                ax.annotate('',xy=(ca,y),xytext=(cu,y),arrowprops={'arrowstyle':'->','color':color,'lw':1.5,'shrinkA':5,'shrinkB':5})
                ax.scatter([cu],[y],s=55,facecolors='white',edgecolors=color,linewidths=1.6,zorder=3)
                ax.scatter([ca],[y],s=55,color=color,zorder=3)
                offset = -17 if device == 'pi' and metric == 'chrf2' and r['asr'] == 'base.en' and r['chunk_s'] == 4 else 8
                ax.annotate(label(r['asr'],r['chunk_s']),(ca,y),xytext=(0,offset),textcoords='offset points',ha='center',fontsize=9,color=color)
            ax.set_ylim(*limits);ax.set_xlim(1.65,7.9);ax.grid(alpha=.17)
            if row==0:ax.set_title('Raspberry Pi 5' if device=='pi' else 'ThinkPad laptop')
            if col==0:ax.set_ylabel(ytitle)
            if row==1:ax.set_xlabel('LongYAAL latency (s; lower is better)')
    legend=[Line2D([],[],marker='o',linestyle='none',markerfacecolor='white',markeredgecolor='#333333',label='CU: audio availability clock'),
            Line2D([],[],marker='o',linestyle='none',color='#333333',label='CA: measured completion clock')]
    fig.legend(handles=legend,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.025))
    fig.tight_layout(pad=1.6);figure_save(fig,'quality-latency')
    fig,ax=plt.subplots(figsize=(9.5,4.4))
    x=np.arange(8);ar=[r['asr_rtf'] for r in main];mt=[r['mt_rtf'] for r in main]
    ax.bar(x,ar,color='#4477AA',label='ASR stage')
    ax.bar(x,mt,bottom=ar,color='#66CCAA',label='MT stage')
    for i,r in enumerate(main):ax.text(i,r['rtf_processing']+.018,f"{r['rtf_processing']:.3f}",ha='center',fontsize=9)
    ax.axhline(1,color='#777777',ls='--',lw=1);ax.text(7.4,1.015,'RTF = 1',ha='right',fontsize=9,color='#555555')
    ax.set_ylim(0,1.09);ax.set_ylabel('Processing seconds / audio second')
    ax.set_xticks(x,[label(r['asr'],r['chunk_s']) for r in main],rotation=25,ha='right')
    ax.axvline(3.5,color='#BBBBBB',lw=.8);ax.text(1.5,.9,'Raspberry Pi 5',ha='center',weight='bold');ax.text(5.5,.9,'ThinkPad laptop',ha='center',weight='bold')
    ax.legend(loc='upper left',bbox_to_anchor=(.0,.77),frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();figure_save(fig,'processing-rtf')
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.8),sharey=True)
    for ax,model,seconds,talk,span in zip(axes,('tiny','base'),(4,8),('xhjBHGVOyQ','UOlPKyCVgg'),((212,264),(184,256))):
        for dev,color in [('pi','#D55E00'),('laptop','#0072B2')]:
            cc=run_audit[f'{dev}/{talk}-{model}-{seconds}s-r1']['chunks']
            ax.plot([c['source_end_s'] for c in cc],[c['queue_wait_s'] for c in cc],marker='o',ms=3,lw=1.4,label='Pi' if dev=='pi' else 'Laptop',color=color)
        ax.set_xlim(*span);ax.set_ylim(-.3,10.0);ax.set_title(f'{model}.en / {seconds}s — {talk}',fontsize=10)
        ax.set_xlabel('Chunk audio end (s)');ax.grid(alpha=.17)
    axes[0].set_ylabel('Queue wait before processing (s)');axes[0].legend(frameon=False)
    fig.tight_layout();figure_save(fig,'queue-recovery')
    fig,ax=plt.subplots(figsize=(9.5,4.5));bottom=np.zeros(8)
    for field,color,name in [('substitutions','#4477AA','Substitutions'),('deletions','#EEAA33','Deletions'),('insertions','#CC6677','Insertions')]:
        vals=np.array([100*r[field]/r['reference_words'] for r in main]);ax.bar(x,vals,bottom=bottom,color=color,label=name);bottom+=vals
    for i,r in enumerate(main):ax.text(i,bottom[i]+.8,f"{r['wer_pct']:.2f}",ha='center',fontsize=9)
    ax.set_xticks(x,[label(r['asr'],r['chunk_s']) for r in main],rotation=25,ha='right');ax.set_ylim(0,48)
    ax.set_ylabel('Errors per 100 reference words');ax.axvline(3.5,color='#BBBBBB',lw=.8)
    ax.text(1.5,45,'Raspberry Pi 5',ha='center',weight='bold');ax.text(5.5,45,'ThinkPad laptop',ha='center',weight='bold')
    ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.17),frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.tight_layout();figure_save(fig,'asr-error-components')
    checks={'grid_complete':True,'runs':len(runs),'within_device_identical_pairs':len(within),
            'input_score_hashes_match':True,'pooled_wer_matches':True,'rtf_reaggregation_matches':True,
            'comet_reaggregation_matches':True,'all_whole_talk_rtfs_below_one':all(r['rtf_processing']<1 for r in runs),
            'api_or_model_calls_made':0,'quality_scoring_repeated':False,
            'python_packages':{'matplotlib':matplotlib.__version__,'numpy':np.__version__}}
    (ROOT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    build()
