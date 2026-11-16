"""Build the 8-page Chinese report using only recorded results."""
import json,csv,sys,os
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[1]

def main():
    font=os.environ.get('PROJECT_CJK_FONT','C:/Windows/Fonts/simhei.ttf')
    if not Path(font).exists():raise SystemExit('Set PROJECT_CJK_FONT to a Chinese TrueType .ttf font.')
    pdfmetrics.registerFont(TTFont('CJK',font))
    s=json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    body=ParagraphStyle('Body',fontName='CJK',fontSize=10,leading=16,spaceAfter=8,textColor=colors.HexColor('#253448'),wordWrap='CJK')
    small=ParagraphStyle('Small',parent=body,fontSize=8,leading=12,spaceAfter=5)
    heading=ParagraphStyle('Heading',parent=body,fontSize=17,leading=23,spaceAfter=15,textColor=colors.HexColor('#176b89'))
    sub=ParagraphStyle('Sub',parent=body,fontSize=11.5,leading=17,spaceBefore=5,spaceAfter=7)
    title=ParagraphStyle('Title',parent=body,fontSize=22,leading=32,spaceAfter=18)
    story=[]
    def clean(text):
        for a,b in [('²³','^23'),('⁻¹⁰','^(-10)'),('⁻⁴','^(-4)'),('⁻⁸','^(-8)'),('²','^2'),('⁶','^6'),('ᵢ','_i'),('₁','1'),('₂','2'),('⊙','*'),('←','='),('−','-')]:
            text=text.replace(a,b)
        return text
    def p(text,style=body):story.append(Paragraph(clean(text),style))
    def h(text):p(text,heading)
    def section(text):p(text,sub)
    def table(rows,widths=None):
        cells=[[Paragraph(clean(escape(str(c))),small) for c in row] for row in rows]
        t=Table(cells,colWidths=widths or [505/len(rows[0])]*len(rows[0]),hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f1f5')),('VALIGN',(0,0),(-1,-1),'TOP'),
                              ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#176b89')),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#c4d1da')),
                              ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.extend([t,Spacer(1,9)])
    def fig(name,height=175):
        from PIL import Image as PILImage
        path=ROOT/f'figures/{name}.png';im=PILImage.open(path);w,h=im.size
        scale=min(505/w,height/h);story.extend([Image(str(path),width=w*scale,height=h*scale),Spacer(1,5)])
    def page():story.append(PageBreak())
    def f(x,n=3):return f'{float(x):.{n}f}'
    def sci(x):return f'{float(x):.2e}'

    p('基于粒子群算法的汽油辛烷值损失优化',title)
    p('NumPy 向量化实现、性能比较与代理模型可靠性分析',heading)
    p('数据科学中的计算机技能 · 期末项目',sub)
    p('孙慧鑫　2026001409　单人项目，独立负责全部模块',body)
    p('2026 年 11 月 15 日',small)
    section('摘要')
    p(f'本项目研究汽油生产中操作参数的离线优化，固定每个样本的 4 项原料性质，采用粒子群优化（PSO）搜索 23 个操作变量。项目用 NumPy 实现两个独立的 27-50-1 ReLU 预测网络、同步全局最优 PSO，以及逐粒子和向量化两个计算版本。相同随机数和评价预算下，两个版本的最优目标值轨迹最大绝对差为 {sci(s["max_equivalence_error"])}。在 32、128、300 个粒子的测试中，向量化版本的中位耗时加速分别为 '+ '、'.join(f'{r["speedup_median"]:.1f} 倍' for r in s['timings'])+'。')
    p('应用结果同时暴露了模型局限：按时间留出的 65 个样本上，硫含量预测的 R² 为负；PSO 在代理模型中将损失推到近零，但候选点通常远离训练数据。因此，本项目支持“批量计算显著降低本实现的计算开销”，不支持“真实装置已获得同等幅度的辛烷值改善”。')
    section('本项目的实质性工作')
    table([['模块','完成内容'],['数据与模型','按仪表位号映射变量，核对范围；时间划分、训练集标准化、独立测试'],['算法实现','自行编写前向/反向传播、Adam、同步 PSO、可行性优先的目标评价'],['实验与工程','实现等价性验证、性能计时、随机搜索基线、多次独立运行、进程开销实验和可复现入口']],[86,419])
    p('研究范围：离线的约束优化与计算实现，重点评估计算效率、解质量与代理模型可靠性。模型预测不等同于实际生产测量。',small)

    page();h('1　问题定义、数据与变量设置')
    p('设 r 为一个样本固定的原料性质，u∈R²³ 为可调操作变量，f(r,u) 和 g(r,u) 分别为产品 RON 和硫含量的预测模型。目标是最小化 L(u)=RON原料−f(r,u)，并满足操作范围、0≤g(r,u)≤5 μg/g、0≤f(r,u)≤RON原料。采用闭区间数值边界，允许 10⁻¹⁰ 的约束计算容差。')
    p('数据来自用户提供的“附件一：325个样本数据.xlsx”和“附件四：354个操作变量信息.xlsx”。按样本编号读取前者 Sheet1 第 4—328 行，共 325 条；提取原料硫含量、RON、饱和烃比例、密度，以及预先固定的 23 个操作变量。产品 RON、产品硫含量仅作为目标，RON 损失列不作为输入。选中变量无缺失；未重做全体变量的随机森林筛选。')
    section('变量映射与实现设置')
    table([['设置','本项目采用的方案'],['按仪表位号匹配，避免中文同名变量混淆','FT_1001.PV 范围为 40–145；FT_9401.PV 范围为 25–900，均由附件四读取'],['EH-103 存在两个同名温度测点','本项目明确采用 TXE_3201A.DACA，其对应范围为 300–550'],['RON 与硫含量分别建立预测器','采用两个独立的 27-50-1 ReLU 网络，统一输入顺序并分别训练'],['固定算法参数与评价预算','c1=c2=0.5，w=0.8；主应用为 128 粒子、120 轮，全部参数公开'],['保留模型及完整运行入口','提供 NumPy 模型、PSO 源码、最终权重与从头训练脚本']],[195,310])
    section('划分与搜索边界')
    dates=s['split']
    p(f'按时间升序排列，较早 260 条用于训练（{dates["training_dates"][0]} 至 {dates["training_dates"][1]}），较新 65 条作最终留出测试。训练段内部最后 52 条作调参验证；全部标准化参数仅从当次训练段计算。训练段选定种子和轮数后，在 260 条上重新拟合，不用留出集选择模型。')
    p('主实验边界为“附件物理范围”与“260 条训练数据逐列最小/最大值”的交集，避免直接在极宽量程上搜索。此边界仅约束各列范围，不保证联合工况受数据支持。变量完整位号、原始范围字符串、单位与 Excel 列号保存在 data/variables.json。附件中的可疑单位按原文保存，不据此建立额外物理关系。',small)

    page();h('2　预测模型与独立测试')
    p('两个网络分别预测 RON 和硫含量。每个网络为 27 维输入、50 个 ReLU 隐单元、1 个线性输出；输入和目标均用训练段均值与标准差标准化。优化目标为标准化均方误差加权重 L2 项（系数 10⁻⁴，不惩罚偏置）。采用全批 Adam，学习率 0.003，β1=0.9、β2=0.999、ε=10⁻⁸。前向、梯度和参数更新均由 NumPy 实现。')
    p('每个目标尝试种子 11、23、47，最多训练 1500 轮，每 25 轮计算验证误差，选择验证误差最低的种子与轮数，再用全部训练段重训。基准包括标准化岭回归（α=10，截距不惩罚）和训练均值预测。模型结构固定，不根据最终测试结果切换优化器使用的模型。')
    table([['目标','方法','留出 RMSE','留出 MAE','留出 R²']]+[[('RON' if r['target']=='ron' else '硫含量'),r['method'],f(r['rmse']),f(r['mae']),f(r['r2'])] for r in s['models']],[64,110,110,110,111])
    fig('prediction',170)
    p('图 1　时间留出集上的实测值与 MLP 预测值，虚线为理想一致线。RON 的 MLP 误差明显低于训练均值，但岭回归在本次划分上更好。硫含量的 MLP 和岭回归均未超过测试集均值这一 R² 基准，表明模型对较新工况的泛化不足。',small)
    p('不能用 RON 约 90 的分母使相对误差看起来很小，就推断对约 1 个单位的损失改善估计很准。硫含量预测误差尤其会影响 5 μg/g 附近的可行性判断。后续 PSO 仍使用固定 MLP，以检验计算实现，并把该问题作为结果解释的前提。')

    page();h('3　PSO 实现与正确性验证')
    p('为避免各变量量纲差异影响速度，先将操作变量线性映射到 [0,1]²³。在归一化空间更新：')
    p('vᵢ←clip[ωvᵢ+c₁r₁⊙(pbestᵢ−zᵢ)+c₂r₂⊙(gbest−zᵢ), −0.2, 0.2]<br/>zᵢ←clip(zᵢ+vᵢ, 0, 1)',sub)
    p('r₁、r₂ 为按粒子和维度独立生成的均匀随机数。越界维度裁剪并将该维速度置零。所有粒子评价后同步更新个人最优与全局最优；每个任务固定随机种子。首个粒子加入原工况在搜索框中的投影，其余随机初始化。')
    p('评价函数对可行点返回 RON 损失；对不可行点返回 10⁶ 加约束违反量，包含 RON 超过原料值、RON 为负、硫为负和硫超过 5 的违反量。因此不可行点不会胜过正常可行点，同时可以按违反程度移动，避免所有不可行点取同一常数。遇到 NaN、无穷值或错误输出形状立即报错。')
    section('验证方法与结果')
    p('11 项自动测试覆盖：Sphere 已知极小值、最优轨迹不增、循环/批量等价、边界最优、零迭代、非法输入、约束评价、MLP 独立标量前向参照、学习与序列化、岭回归已知线性关系、随机搜索预算，以及数据映射。另对三种解析函数各做 20 个种子的实验。')
    table([['解析函数','维度','最好目标中位数','种子间标准差']]+[[r['function'],r['dimension'],sci(r['median']),sci(r['sd'])] for r in s['analytic']],[140,60,155,150])
    p('表中预算为 96×201=19,296 次评价，200 次更新，理论全局最小值均为 0。Sphere 和二维函数可接近最优；10 维 Rastrigin、Rosenbrock 仍有明显残差，说明当前参数和预算下的 PSO 不保证全局最优。二维 Rastrigin 输出为 0 受浮点舍入影响，不表示精确数学证明。',small)
    p('两个实现使用同形状、同顺序的随机数及相同同步更新规则。循环版逐粒子计算速度、位置和模型预测；向量化版一次更新整个种群并批量前向计算。两者没有改变问题、模型、搜索预算或终止条件。')

    page();h('4　计算瓶颈与向量化性能')
    p('每轮需评价 P 个粒子、两个 27-50-1 网络。循环版反复进行小矩阵运算和 Python 函数调用；批量版将输入拼为 P×27 矩阵，以矩阵乘法完成整群预测。两者主要算术量同为 O(TP(qH+D))，其中 q=27、H=50、D=23。向量化减少解释器和小调用开销，不改变算法复杂度；额外暂存量主要为 O(P(q+H+D))。')
    p('计时固定第一个留出样本，P∈{32,128,300}，T=20；每种实现先预热一次，再各测 5 次，交替执行顺序，用 perf_counter 测量完整搜索调用。计时包含初始化、随机数、PSO 更新和模型预测，不包含训练、文件读写、绘图或验证。两个版本使用同一已加载模型与种子，BLAS 线程数固定为 1。')
    table([['粒子数','循环中位数 ms','批量中位数 ms','中位数比值']]+[[r['population'],f(r['loop']['median']*1000),f(r['vector']['median']*1000),f(r['speedup_median'],1)+' 倍'] for r in s['timings']],[75,145,145,140])
    fig('timing',190)
    p(f'图 2　左图为对数耗时，右图为同规模中位耗时比值。所有条件下最好目标值一致；轨迹最大绝对差 {sci(s["max_equivalence_error"])}。误差验证标准为绝对误差与相对误差均不超过 10⁻¹⁰。粒子数增大时，批量计算对调用开销的摊薄更明显。',small)
    p('这组速度比针对本项目清晰可读的 Python 逐粒子基准。它不能直接推广为“PSO 比其他优化算法快”，也不能推广到已有高度优化的 C++ 实现。毫秒级计时易受系统调度影响，原始五次记录全部保留；跨机器应重新运行。')
    e=s['environment']
    p(f'实验环境：{e["platform"]}；{e["processor"]}；系统报告 {e["logical_cpu_count"]} 个逻辑处理器；Python 3.12.14，NumPy {e["numpy"]}，OpenBLAS 0.3.30，CPU 计算，无 GPU。环境信息与完整 NumPy 构建信息见 results/environment.json。',small)

    page();h('5　等预算解质量与 325 样本应用')
    p('在 65 条留出数据中，按时间顺序等间距固定选取 5 条，每条使用种子 0—19；PSO 与独立均匀随机搜索各 100 次。两者包含相同初始投影点，每次严格使用 96×81=7,776 次目标评价。PSO 更新 80 轮；随机搜索按 512 个候选批量计算。不以算法调用次数代替候选评价次数。')
    table([['方法','代理可行次数','预测损失中位数','预测损失标准差','耗时中位数 ms']]+[[r['method'],str(r['feasible'])+'/'+str(r['runs']),sci(r['loss']['median']),sci(r['loss']['sd']),f(r['seconds']['median']*1000)] for r in s['quality']],[75,100,115,115,100])
    fig('quality',175)
    p('图 3　左图汇总 5 个样本各 20 次运行的分布，属于跨样本混合描述，不是 100 个独立工况的置信区间；逐样本种子记录保存在 quality.csv。右图为第一个留出样本的中位轨迹与四分位带。PSO 在固定评价预算下取得更低代理损失，但其协调更新有额外成本，随机搜索在本次实现中的实际耗时更短。',small)
    section('全体样本的端到端演示')
    a=s['application']['all'];b=s['application']['holdout']
    p(f'对全部 325 个样本各运行一次 128 粒子、120 轮的向量化 PSO，每条预算 15,488 次评价，种子为 20261114+样本编号。共 {a["feasible"]}/325 条满足代理模型约束，搜索耗时合计 {a["seconds_sum"]:.2f} 秒，预测损失中位数为 {sci(a["predicted_loss"]["median"])}。其中 65 条留出数据的预测损失中位数为 {sci(b["predicted_loss"]["median"])}。训练样本上的结果仅作应用展示，不用作模型泛化证据。')
    p('所有 23 项最优操作参数逐样本导出到 optimized_controls.csv。application.csv 同时记录实测产品值、原模型预测、初始投影预测、优化预测和边界投影维度数。22 个原工况超出主搜索框，改善率的分母使用投影后初始工况的预测损失，并明确标注为“相对模型的改善”；不把实测初值与优化预测混算为实测收益。')

    page();h('6　模型可靠性与结果解释边界')
    p('几乎为零的预测损失并不意味着装置损失几乎消失。PSO 优化的对象是学习得到的曲面；当模型存在外推误差时，搜索会偏向模型给出的过度乐观区域。尤其硫含量模型在时间留出集上 R²&lt;0，预测的“可行”不能保证真实硫含量合格。')
    fig('reliability',180)
    p('图 4　用训练集均值/标准差标准化全部 27 项特征，计算候选点到最近训练样本的欧氏距离。该距离是支持程度诊断，不是误差保证，也不是经过校准的异常阈值。',small)
    d=s['holdout_original_nn_distance'];ad=s['application']['holdout']['distance']
    p(f'原始留出观测的最近训练距离中位数为 {d["median"]:.2f}，优化后留出候选的距离中位数为 {ad["median"]:.2f}。即使每列都在训练范围内，23 项操作变量的自由组合仍可能偏离训练样本的联合分布；主边界只能减少显然的单变量外推。')
    table([['搜索边界','重复次数','代理可行次数','最近训练距离中位数']]+[[('训练范围交集' if r['box']=='training_box' else '附件全部物理范围'),r['runs'],r['feasible'],f(r['distance']['median'],2)] for r in s['sensitivity']],[155,75,95,180])
    p('边界敏感性实验固定 5 个留出样本、每个 5 个种子、96 粒子与 80 轮，仅改变搜索边界。完整物理范围包含例如火炬气排放流量 20—20,000,000 的极宽量程，因此更需防止模型外推。两种边界都可找到近零预测损失，不能据此判定真实收益；需要局部信赖域、预测不确定性或额外工况数据进一步验证。')
    p('单条收敛轨迹只能说明该次运行中的目标改善，不能证明全局最优。搜索迭代与装置操作调整属于不同过程；本项目仅保存实际搜索轮次。当前约束还未纳入设备联动关系和动态响应，因而这些候选点仅用于离线计算分析。')

    page();h('7　并行开销、复现与结论')
    fig('parallel',150)
    pp=s['parallel']
    p(f'图 5　12 个留出样本各做 96 粒子、80 轮搜索，单进程、2 进程、4 进程各测 3 次，中位端到端耗时分别为 {pp[0]["median"]:.3f}、{pp[1]["median"]:.3f}、{pp[2]["median"]:.3f} 秒。计时包含新建进程池、模型载入与通信，所有样本结果完全一致。任务太短时进程启动开销占主导，并行更慢；原始记录保留首次运行异常偏高值，不删除不利数据。',small)
    section('复现方法与工程组织')
    p('运行 python -m unittest discover -s tests -v 验证核心实现；运行 python scripts/demo.py 查看单样本小例子；运行 python scripts/run_experiments.py 重新训练并生成完整记录；再运行 summarize.py、plot_results.py、build_report.py 重建摘要、图和报告。README 提供依赖、版本、命令、Git 提交说明及 AI 声明，模型权重以 NPZ 保存，无任意代码反序列化。')
    p(f'本机完整数值实验耗时约 {s["environment"]["experiment_seconds"]:.1f} 秒（不含依赖安装、绘图与文档生成）。建议 CPU、2 GB 可用内存，无 GPU 要求；这只是保守运行建议，未测量峰值内存。不同环境计时不保证一致。所有配置、原始 CSV、训练/留出编号、模型权重与重建脚本一并提交。')
    section('结论')
    p('NumPy 批量计算在保持同步 PSO 语义与数值一致性的前提下，显著降低了当前逐粒子 Python 实现的耗时。相同评价预算下 PSO 比随机搜索更接近代理目标下界，但随机搜索的实际耗时更低。短任务多进程未获得加速。硫含量泛化不足及优化点远离训练数据，限制了工业结论；本项目的主要贡献是可验证的实现、性能证据和适用边界分析。')
    section('参考资料与 AI 使用说明')
    p('[1] 用户提供：《汽油辛烷值优化建模》，方法与变量配置参考资料，作者及年份未注明。<br/>[2] 马维：《数据科学中的计算机技能》，final/final-2026.md（Draft，2026-10-25）。<br/>[3] 用户提供：“附件一：325个样本数据.xlsx”“附件四：354个操作变量信息.xlsx”；文件 SHA-256 见 data/provenance.json。',small)
    p('本项目使用 OpenAI Codex 协助数据核对、代码编写、实验执行、文档生成与检查。AI 编写与运行的内容均已明确披露；提交人需在提交及答辩前自行审阅并能解释核心代码与结果。方法及变量配置的参考来源列于参考资料；核心 NumPy 实现由本项目编写，所有实验结论均来自本项目运行记录。',small)

    def footer(c,doc):
        c.setStrokeColor(colors.HexColor('#c5d6df'));c.line(45,36,550,36)
        c.setFont('CJK',8);c.setFillColor(colors.HexColor('#667b8a'))
        c.drawString(45,23,'孙慧鑫 2026001409 | 数据科学中的计算机技能')
        c.drawRightString(550,23,f'{doc.page} / 8')
    doc=SimpleDocTemplate(str(ROOT/'report.pdf'),pagesize=(595.28,841.89),rightMargin=45,leftMargin=45,topMargin=42,bottomMargin=49,
                          title='基于粒子群算法的汽油辛烷值损失优化',author='孙慧鑫 2026001409')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('Built report.pdf')

if __name__=='__main__':main()
