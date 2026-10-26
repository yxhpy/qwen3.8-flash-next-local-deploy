"""Render the published measurements. Requires matplotlib; no GPU or model calls."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'benchmarks/2026-10-26'
OUT=ROOT/'docs/assets'
OUT.mkdir(exist_ok=True)
for candidate in ['/System/Library/Fonts/Hiragino Sans GB.ttc',
                  '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    if Path(candidate).exists():
        fm.fontManager.addfont(candidate)
        plt.rcParams['font.family']=fm.FontProperties(fname=candidate).get_name()
        break
plt.rcParams.update({'axes.unicode_minus':False,'font.size':12,'svg.fonttype':'path','svg.hashsalt':'strata-20261026'})
BG='#F6F4EF'; INK='#182B36'; MUTED='#5D707B'; GRAY='#9AAAB2'; TEAL='#007F83'; AMBER='#B45A25'; GRID='#DDE2DF'
s=json.loads((DATA/'summary.json').read_text())
records=[json.loads(x) for x in (DATA/'records.jsonl').read_text().splitlines()]


def canvas(kicker,title,subtitle):
    fig=plt.figure(figsize=(12,7.5),facecolor=BG)
    fig.text(.065,.94,kicker,fontsize=11,color=TEAL,weight='bold')
    fig.text(.065,.872,title,fontsize=25,color=INK,weight='bold')
    fig.text(.065,.824,subtitle,fontsize=11,color=MUTED)
    return fig


def axis(fig,box,xmax):
    ax=fig.add_axes(box,facecolor=BG)
    for sp in ax.spines.values():sp.set_visible(False)
    ax.set_xlim(0,xmax)
    ax.tick_params(axis='both',length=0,labelcolor=MUTED)
    ax.grid(axis='x',color=GRID,lw=.8,zorder=0)
    ax.set_axisbelow(True)
    return ax


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=180,facecolor=BG)
    svg=OUT/f'{name}.svg'
    fig.savefig(svg,facecolor=BG,metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


f=canvas('01  /  调度优化实测 · 2026-10-26','同一模型配置，生成速度提升 6.6%–11.3%',
         'RTX 4080 SUPER 16 GB · i7-12700 · 64 GB DDR4-3200 · Strata 0.1.44.1')
a=axis(f,[.19,.30,.66,.43],105)
kinds=['short','code','8k']
for j,kind in enumerate(kinds):
    y=2-j
    d=s['production_configuration_comparison'][kind]
    for arm,offset,color in [('baseline',.15,GRAY),('optimized',-.15,TEAL)]:
        med=d[arm]['median_tokens_per_second']
        a.barh(y+offset,med,height=.22,color=color,zorder=2)
        prefix='natural-base-' if arm=='baseline' else 'natural-combined-'
        values=[x['timings']['predicted_per_second'] for x in records if x['run'].startswith(prefix) and x['kind']==kind and not x['warmup']]
        a.plot([min(values),max(values)],[y+offset]*2,color=INK,lw=1.1,zorder=3)
        a.scatter(values,[y+offset]*len(values),s=13,c=INK,alpha=.75,zorder=4)
        a.text(max(values)+1.3,y+offset,f'{med:.2f}',va='center',color=INK,fontsize=12)
    a.text(104,y,f"+{(d['ratio_of_median_speed']-1)*100:.1f}%",va='center',ha='left',color=TEAL,fontsize=18,weight='bold')
a.set_yticks([2,1,0],['短文本生成','代码生成','8K 输入后生成'])
a.set_ylim(-.5,2.5)
a.set_xticks([0,20,40,60,80,100]);a.set_xlabel('生成速度  token/s  ·  越高越快',color=MUTED,labelpad=10)
f.text(.195,.765,'● 优化前',color=GRAY,fontsize=12)
f.text(.32,.765,'● 优化后',color=TEAL,fontsize=12)
f.text(.51,.765,'每侧 4 次正式测量 · 12 / 12 配对更快',color=INK,fontsize=12)
f.text(.065,.174,'仅调整 CPU 任务粒度与复制调度，保留原模型、FP16 KV 和 256K 上下文。',color=INK,fontsize=12)
f.text(.065,.127,'数值核对：固定相同执行顺序的 11 组样本，文本、首步全词表数值与完整状态一致。',color=MUTED,fontsize=10.5)
f.text(.065,.084,'柱为中位数，点为单次测量，细线为最小至最大；不表示置信区间。',color=MUTED,fontsize=10.5)
f.text(.065,.035,'yxhpy / qwen3.8-flash-next-local-deploy   ·   v1.2.0   ·   数据及复核脚本见 GitHub',color=MUTED,fontsize=9)
save(f,'decode-scheduling-20261026')

f=canvas('02  /  长对话缓存对照','复用已保存上下文，缩短续问等待',
         '同一续问：完整重读全文，对比恢复缓存后继续 · 两侧模型均已加载')
a=axis(f,[.19,.30,.72,.43],70)
for j,d in enumerate(s['long_context']):
    y=1-j
    cold=d['cold_first_token_seconds'];rs=d['optimized_restore_seconds'];first=d['optimized_first_token_after_restore_seconds']
    total=rs+first
    a.barh(y+.16,cold,height=.23,color=GRAY,zorder=2)
    a.barh(y-.16,rs,height=.23,color=AMBER,zorder=2)
    a.barh(y-.16,first,left=rs,height=.23,color=TEAL,zorder=2)
    a.text(cold+1,y+.16,f'{cold:.2f} 秒',color=INK,va='center',fontsize=13)
    a.text(total+1,y-.16,f'合计 {total:.2f} 秒（其中恢复 {rs:.2f} 秒）',color=TEAL,va='center',fontsize=12)
a.set_yticks([1,0],['约 3.3 万 token','约 10.3 万 token']);a.set_ylim(-.52,1.52)
a.set_xlabel('收到第一个输出的等待时间 / 秒  ·  越低越快',color=MUTED,labelpad=10)
f.text(.195,.765,'● 完整重读',color=GRAY,fontsize=12)
f.text(.36,.765,'● 缓存恢复耗时',color=AMBER,fontsize=12)
f.text(.565,.765,'● 恢复后等首字',color=TEAL,fontsize=12)
f.text(.065,.174,'这是缓存复用的收益，不能归因于调度参数带来十几倍生成速度。',color=INK,fontsize=12)
f.text(.065,.127,'每种长度 1 次同输入续问对照；包含恢复耗时，不含模型加载。首次读入长文本无此收益。',color=MUTED,fontsize=10.5)
f.text(.065,.084,'精确输入长度：32,932 / 102,565 token。使用合成文档，没有重放业务会话。',color=MUTED,fontsize=10.5)
f.text(.065,.035,'yxhpy / qwen3.8-flash-next-local-deploy   ·   v1.2.0   ·   数据及复核脚本见 GitHub',color=MUTED,fontsize=9)
save(f,'cache-reuse-20261026')

u=json.loads((DATA/'gpu-unlock-controlled.json').read_text())['comparisons']
mma=(u['mma_TF_s']['f16/f32']['speedup']-1)*100
gemm=[(v['speedup']-1)*100 for v in u['gemm_TF_s'].values()]
pf=[v['prompt_per_second']['change_pct'] for k,v in u['llm'].items() if k.startswith('cold ')]
dec=[v['predicted_per_second']['change_pct'] for v in u['llm'].values()]
f=canvas('03  /  GPU 解锁单独实验','张量算力接近翻倍，模型生成未稳定提速',
         'RTX 4080 SUPER · 同卡同驱动 · 模型对照固定实际专家缓存容量')
a=axis(f,[.28,.30,.61,.43],115)
a.set_xlim(-12,115)
groups=[('FP16 / FP32 Tensor',[mma],7),('FP16 / BF16 GEMM',gemm,3),('Qwen 长输入读取',pf,2),('Qwen 生成',dec,2)]
for j,(label,vals,n) in enumerate(groups):
    y=3-j;lo=min(vals);hi=max(vals)
    color=TEAL if j<2 else AMBER
    if len(vals)==1:
        a.barh(y,hi,height=.25,color=color,zorder=2)
        text=f'+{hi:.1f}%'
    else:
        a.plot([lo,hi],[y,y],lw=13,color=color,alpha=.6,solid_capstyle='round',zorder=2)
        a.scatter(vals,[y]*len(vals),color=color,s=26,zorder=3)
        text=f'{lo:+.1f}% 至 {hi:+.1f}%'
    a.text(hi+3,y,text,color=INK,va='center',fontsize=12)
a.axvline(0,color=INK,lw=1)
a.set_yticks([3,2,1,0],[x[0] for x in groups]);a.set_ylim(-.55,3.5)
a.set_xticks([0,25,50,75,100]);a.set_xlabel('相对未解锁基线的速度变化 / %',color=MUTED,labelpad=10)
f.text(.065,.174,'本次最终保留正常 GPU 启动状态，随后单独验证并应用软件调度优化。',color=INK,fontsize=12)
f.text(.065,.127,'区间表示不同矩阵形状或上下文结果，非置信区间；Tensor / GEMM / LLM 每项每侧 7 / 3 / 2 次。',color=MUTED,fontsize=10.5)
f.text(.065,.084,'单卡实验；不据此推断其他模型、显卡或长期运行效果，也不声称解锁前后输出逐字一致。',color=MUTED,fontsize=10.5)
f.text(.065,.035,'yxhpy / qwen3.8-flash-next-local-deploy   ·   v1.2.0   ·   数据及复核脚本见 GitHub',color=MUTED,fontsize=9)
save(f,'gpu-unlock-20261026')
print('Rendered 3 PNG charts and 3 SVG counterparts in',OUT)
