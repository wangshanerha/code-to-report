#!/usr/bin/env python3
"""Rebuild the nanoGPT source-analysis demo, without importing or executing nanoGPT."""
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('report_builder', ROOT / 'scripts/build_report.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
PROVENANCE = json.loads((HERE / 'source/provenance.json').read_text(encoding='utf-8-sig'))
SHA = PROVENANCE['revision']


def node(id, title, subtitle, row, input, output, details, example, source, col=1, kind='step', subflow=None):
    data = dict(id=id, title=title, subtitle=subtitle, row=row, col=col, input=input, output=output,
                details=details, example=example, sources=[source], kind=kind)
    if subflow:
        data['subflow'] = subflow
    return data


def edge(a, b, label, **kwargs):
    return dict({'from': a, 'to': b, 'label': label}, **kwargs)


forward = dict(id='forward', short_title='模型完整前向', title='01 · 从 token ID 到 logits / loss', intro=[
    '这是 GPT.forward 的完整主路径。输入已经是 token ID，不是原始中文或英文字符串。文字编码发生在 sample.py 或数据准备程序中。',
    'B 为批次数，T 为序列长度，C 为隐藏维度，V 为词表大小。注意：输出分支由 targets 是否为 None 决定，而不是由 model.train() / model.eval() 决定。'], nodes=[
    node('F01','接收 token 与可选标签','idx [B,T]；targets 可为 None',0,'idx: LongTensor[B,T]；可选 targets[B,T]','读取 b,t 和 idx.device',
         ['idx.size() 必须能解包为两个维度。token ID 需在词表范围内；targets 的结构应与 idx 对齐。','此函数不负责分词，也不会自动把 idx 右移生成标签；调用者必须准备好 targets。'],
         '教学词表 V=8：idx=[[1,2,3]]，targets=[[2,3,4]]。这些 ID 不是 GPT-2 分词器的实际编码。','fwd'),
    node('F02','序列长度是否允许？','assert T <= block_size',1,'t 与 config.block_size','通过断言，或抛出 AssertionError',
         ['直接 forward 只检查长度并断言，不会自动裁剪输入。','生成循环会在调用 forward 前裁剪上下文；这是两个不同位置的行为。'],
         'block_size=4 时，T=3 通过；直接输入 T=5 会失败。','fwd',kind='decision'),
    node('FX','超长输入：断言失败','本次 forward 终止',1,'T > block_size','AssertionError；没有 logits/loss',
         ['断言包含实际长度和允许的最大长度。','不会进入 Embedding、Attention 或损失计算。其他错误如越界 ID 可能在后续 PyTorch 操作中抛出。'],
         'idx=[[0,1,2,3,4]] 且 block_size=4：在进入 Embedding 前终止。','fwd',col=2,kind='terminal'),
    node('F03','Token + Position Embedding','查表、相加、Dropout',2,'idx[B,T]；pos=arange(T)[T]','x[B,T,C]',
         ['wte 将每个 token ID 查成 C 维向量，得到 [B,T,C]。','wpe 将位置 0..T-1 查成 [T,C]，沿 batch 维广播相加。','transformer.drop 对相加结果做 Dropout；概率由配置决定，是否激活由 training 状态决定。'],
         'B=1,T=3,C=4：token 向量 [1,3,4] 加位置向量 [3,4]，输出仍为 [1,3,4]；具体元素依赖模型权重。','fwd'),
    node('F04','依次通过全部 Block','for block in transformer.h',3,'x[B,T,C]','x[B,T,C]；经过 n_layer 层',
         ['按 ModuleList 顺序执行，每个 Block 的输出成为下一个的输入。','每层有独立参数，不是重复调用同一组权重。','Block 内部包含两次 Pre-LayerNorm、注意力、MLP 和两条残差，见子流程。'],
         'n_layer=2 时，x[1,3,4] → Block 0 → x[1,3,4] → Block 1 → x[1,3,4]。','fwd',subflow='block'),
    node('F05','最终 LayerNorm','transformer.ln_f',4,'最后一层 Block 输出[B,T,C]','归一化后的 x[B,T,C]',
         ['在每个 token 的最后一维 C 上归一化。','使用可学习缩放 weight、可选 bias，以及 epsilon=1e-5；形状不变。'],
         '输入是 3 个 4 维 token 向量，归一化分别在各自的 4 个通道上进行，不混合不同 token。','ln'),
    node('F06','是否传入 targets？','targets is not None',5,'x[B,T,C] 与 targets','选择全位置损失分支或末位置分支',
         ['这是输出和损失的分界点。','eval() 但传 targets 仍算 loss；train() 但 targets=None 仍走末位置 logits 分支。Dropout 则由 training 状态控制。'],
         'targets=[[2,3,4]] 时走 F07；targets=None 时走 F08。','fwd',kind='decision'),
    node('F07','全位置投影 + 交叉熵','[B,T,C] → [B,T,V]',6,'x[B,T,C]；targets[B,T]','logits[B,T,V]；标量 loss',
         ['lm_head 把每个位置的隐藏向量投影到词表，输出未归一化的 logits。','把 logits 展平成 [B*T,V]、targets 展成 [B*T]，调用 cross_entropy；它内部完成 log-softmax / NLL 计算。','ignore_index=-1 的目标位置不参与损失。返回的是默认 reduction 的标量损失。'],
         'V=8、B=1、T=3：logits[1,3,8] → [3,8]，targets[1,3] → [3]；3 个位置共同贡献平均 loss。','fwd',col=0),
    node('F08','仅投影最后一个位置','[B,1,C] → [B,1,V]',6,'x[B,T,C]；targets=None','logits[B,1,V]；loss=None',
         ['x[:,[-1],:] 保留长度为 1 的时间维，而非降成二维。','只减少最后的 lm_head 投影开销；前面的 Embedding 和所有 Block 仍处理整个输入上下文。'],
         'x[1,3,4] 取成 [1,1,4]，投影得到 [1,1,8]，用于预测紧接输入之后的一个 token。','fwd',col=2),
    node('F09','返回 logits 与 loss','两个返回值，形状由分支决定',7,'F07 或 F08 的结果','(logits, loss)',
         ['标签分支可继续反向传播；无标签分支可交给生成函数做采样。','forward 自身不更新参数，也不把 logits 解码成文本。'],
         '有标签返回 ([1,3,8],标量)；无标签返回 ([1,1,8],None)。','fwd',kind='terminal'),
], edges=[edge('F01','F02','读取 T'),edge('F02','FX','否：超长',from_side='right',to_side='left'),edge('F02','F03','是：T 合法'),edge('F03','F04','隐藏序列 [B,T,C]'),edge('F04','F05','完成 n_layer 层'),edge('F05','F06','归一化隐藏序列'),edge('F06','F07','是：有 targets'),edge('F06','F08','否：targets=None'),edge('F07','F09','全位置 logits + loss'),edge('F08','F09','末位置 logits + None')])

block_nodes = [
    ('B01','第一层归一化','LayerNorm(x)','x[B,T,C]','z1[B,T,C]',['保存原 x 供残差使用，只对当前 token 的通道维做归一化。','F.layer_norm 使用学习的 weight、可选 bias 和 epsilon=1e-5。'],'x[1,3,4] → z1[1,3,4]，token 数量不变。','ln'),
    ('B02','因果自注意力','attn(z1)','z1[B,T,C]','a[B,T,C]',['把归一化输入交给 CausalSelfAttention，内部多头计算见注意力图。','输出是可与原 x 相加的 C 维向量。'],'a[1,3,4] 的第 2 个位置只依赖第 1、2 个位置。','att'),
    ('B03','第一条残差','x = x + a','原 x[B,T,C] 与 a[B,T,C]','x1[B,T,C]',['逐元素相加，保留原状态并加入注意力变换。','输入和注意力输出形状必须相同，不拼接、不增加通道。'],'两个 [1,3,4] 相加仍为 [1,3,4]。','block'),
    ('B04','第二层归一化','ln_2(x1)','x1[B,T,C]','z2[B,T,C]',['使用另一组 LayerNorm 参数。','这是 Pre-LN：MLP 之前归一化；残差主干仍保留未归一化的 x1。'],'z2[1,3,4] 与 z1 形状相同，但参数和输入不同。','block'),
    ('B05','MLP 升维与 GELU','C → 4C → GELU','z2[B,T,C]','中间激活[B,T,4C]',['c_fc 是 Linear(C,4C)，把每个 token 的特征升维。','GELU 逐元素施加非线性。MLP 不直接混合不同 token 的位置。'],'C=4：每个 token 从 4 维变 16 维，整体 [1,3,4] → [1,3,16]。','mlp'),
    ('B06','MLP 降维与 Dropout','4C → C','中间激活[B,T,4C]','m[B,T,C]',['c_proj 将 4C 投影回 C。','再做配置概率的 Dropout，确保输出能参与残差相加。'],'[1,3,16] → [1,3,4]；输出是特征变换，不是词表概率。','mlp'),
    ('B07','第二条残差并返回','return x1 + m','x1[B,T,C] 与 m[B,T,C]','下一层 Block 的输入[B,T,C]',['逐元素相加并返回。','源码两行残差计算中，第一条使用原 x，第二条使用已经更新的 x1。'],'n_layer=2 时，本结果进入第二个 Block；最后一个 Block 的结果进入 ln_f。','block'),
]
block = dict(id='block',short_title='Block 内部',title='02 · 一个 Transformer Block 内部做什么',intro=['这张图展开 F04 中的一层。整个 Block 不改变 [B,T,C] 的外部形状；注意力混合可见位置的信息，MLP 则变换各位置的特征。'],nodes=[],edges=[])
for i, args in enumerate(block_nodes):
    nid,title,sub,inp,out,detail,ex,src = args
    n=node(nid,title,sub,i,inp,out,detail,ex,src,col=0,kind='terminal' if i==len(block_nodes)-1 else 'step',subflow='attention' if nid=='B02' else None)
    block['nodes'].append(n)
    if i: block['edges'].append(edge(block_nodes[i-1][0],nid,inp.split('；')[0]))

attention=dict(id='attention',short_title='Attention 两条分支',title='03 · 多头因果注意力：拆头、遮罩，再合并',intro=[
    '核心计算是 softmax(QKᵀ / √d + causal mask)V。训练状态影响 Dropout；因果遮罩确保当前位置不能看未来 token。',
    '源码把 hasattr(F,"scaled_dot_product_attention") 命名为 self.flash。它仅检查接口是否存在；具体使用哪种后端由 PyTorch 运行环境决定，不等于保证调用 FlashAttention CUDA 内核。'],nodes=[
    node('A01','接收归一化隐藏状态','x [B,T,C]',0,'x[B,T,C]，且 C % H == 0','读取 B,T,C；每头维度 d=C/H',
         ['n_embd % n_head == 0 在构造注意力模块时断言。','此处输入来自 B01 的 LayerNorm 输出，不是 token ID。'],'C=4,H=2，因此每头 d=2。','att'),
    node('A02','一次投影，拆成 Q/K/V','[B,T,3C] → 3×[B,H,T,d]',1,'x[B,T,C]','q,k,v 各为[B,H,T,d]',
         ['c_attn 是 Linear(C,3C)，然后按最后一维 split(C)。','每个 Q/K/V 先 view(B,T,H,d)，再 transpose(1,2) 把 head 放到第二维。'],
         '[1,3,4] → [1,3,12] → 三个 [1,3,4] → 三个 [1,2,3,2]。','att'),
    node('A03','SDPA 接口可用吗？','if self.flash',2,'初始化时缓存的布尔标志','选择接口分支或显式矩阵分支',
         ['构造时通过 hasattr 检查 F.scaled_dot_product_attention。','不是每个 token 动态选择，且并非按是否训练来选择；两条路径都遵守因果约束。'],
         '可用：走 A04；不可用：走 A05、A06。实际后端未在本环境运行核实。','att',kind='decision'),
    node('A04','调用 PyTorch SDPA','is_causal=True',3,'q,k,v[B,H,T,d]','y[B,H,T,d]',
         ['传入 attn_mask=None、is_causal=True。','dropout_p 在 training 状态取配置 dropout，否则强制为 0。','注意力分数的具体中间张量不由这段代码显式创建；不能据此断言实际显存占用或速度。'],
         'q/k/v[1,2,3,2] → y[1,2,3,2]。dropout=0 时，无论 train/eval，此处概率都是 0。','att',col=0),
    node('A05','显式算分数与因果遮罩','QKᵀ / √d；未来置 −∞',3,'q,k[B,H,T,d]','遮罩后的 scores[B,H,T,T]',
         ['q @ k.transpose(-2,-1) 计算每个 query 与各 key 的匹配分数，并乘 1/sqrt(d)。','取注册下三角遮罩的 [:,:,:T,:T] 部分；未来位置的分数填为 -inf。','遮罩仅在没有 SDPA 接口时注册为 buffer，它不是训练参数。'],
         'T=3 时，可见关系为 [[1,0,0],[1,1,0],[1,1,1]]；第一个 token 只能看自己。','att',col=2),
    node('A06','归一化权重并汇总 V','softmax → Dropout → @V',4,'scores[B,H,T,T]；v[B,H,T,d]','y[B,H,T,d]',
         ['softmax 沿最后一维，对可见 key 位置归一化。','attn_dropout 按模块 training 状态工作；然后权重乘 v 得到每个 query 的加权结果。'],
         '第一个位置的权重为 [1,0,0]，所以其输出就是对应的第一个 V。完整教学数值见贯穿实例之后。','att',col=2),
    node('A07','合并多头与输出投影','transpose → contiguous → view',5,'y[B,H,T,d]','投影与 Dropout 后[B,T,C]',
         ['transpose(1,2) 还原成 [B,T,H,d]；contiguous 后 view 成 [B,T,C]。','c_proj 是 Linear(C,C)，再应用 resid_dropout。','合并是拼回各头通道，不是求平均。'],
         '[1,2,3,2] → [1,3,2,2] → [1,3,4] → 线性投影后 [1,3,4]。','att'),
    node('A08','返回注意力增量','交回第一条残差',6,'最终注意力输出[B,T,C]','a[B,T,C]；由 B03 与主干相加',
         ['注意力模块本身没有做外层残差加法，残差发生在 Block.forward。','输出不是 token 概率；词表投影发生在 GPT.lm_head。'],
         '返回 a[1,3,4]，再计算 x+a。','att',kind='terminal'),
],edges=[edge('A01','A02','隐藏序列'),edge('A02','A03','Q / K / V'),edge('A03','A04','是：接口存在'),edge('A03','A05','否：显式计算'),edge('A05','A06','masked scores'),edge('A04','A07','y [B,H,T,d]',via=[[170,755],[500,755]]),edge('A06','A07','y [B,H,T,d]'),edge('A07','A08','attention output')])

generation=dict(id='generation',short_title='生成循环与采样',title='04 · 如何一个 token、一个 token 地生成',intro=[
    'generate 接收已经编码的 idx，循环 max_new_tokens 次；每次只采一个新 token，再把它拼回完整序列。@torch.no_grad() 关闭梯度记录，但不会自动调用 eval()。',
    '这份实现没有 KV cache，也没有遇到 EOS 提前停止的逻辑。每轮重新计算裁剪后的完整上下文；完整 idx 继续增长，裁剪只作用于送入 forward 的 idx_cond。'],nodes=[
    node('G00','接收生成参数','idx；max_new_tokens；温度；top_k',0,'idx[B,L]；max_new_tokens=N；temperature；top_k','进入无梯度生成循环',
         ['调用者应先设置 model.eval() 以关闭 Dropout；sample.py 会这样做。','temperature 应为正，top_k 若设置应为正整数；函数本身未显式验证这些参数。','生成接口本身不分词、不解码，也不加载权重。'],
         '教学输入 idx=[[0,1,2,3,4]]，block_size=4，N=1，temperature=1，top_k=2。','gen'),
    node('G01','是否还需生成？','for _ in range(N)',1,'循环计数与 max_new_tokens','继续一次迭代，或返回 idx',
         ['源码为固定次数的 for 循环，此节点把隐含循环控制画出来。','N=0 时不调用 forward，直接返回原序列；没有 EOS 检查。'],
         'N=1 时第一次进入；拼接 1 个 token 后循环结束。','gen',kind='decision'),
    node('G02','当前序列是否过长？','idx.size(1) <= block_size',2,'完整 idx[B,L]','选择保留全部或保留末尾窗口',
         ['这里判断的是完整生成序列当前的长度 L。','等于 block_size 时不裁剪；只有大于时才切片。'],
         'L=5、block_size=4：过长，走 G04；L=3 时走 G03。','gen',kind='decision'),
    node('G03','保留整个上下文','idx_cond = idx',3,'idx[B,L]，L<=block_size','idx_cond[B,L]',
         ['直接引用当前 idx 作为模型上下文。','原始序列和送入模型的上下文相同。'],
         '[[1,2,3]] 保持为 [[1,2,3]]。','gen',col=0),
    node('G04','截取末尾上下文','idx[:, -block_size:] ',3,'idx[B,L]，L>block_size','idx_cond[B,block_size]',
         ['只把最后 block_size 个 token 送入模型。','原始 idx 不丢 token；随后拼接仍基于完整 idx。','forward 会重新从位置 0 编到窗口长度-1，使用学习的位置 Embedding。'],
         '原序列 [[0,1,2,3,4]] 保留；idx_cond=[[1,2,3,4]]。','gen',col=2),
    node('G05','调用完整前向','self(idx_cond)；不传 targets',4,'idx_cond[B,T]，T<=block_size','logits[B,1,V]；loss=None',
         ['经过 F01..F09，固定走 targets=None 的末位置输出分支。','没有 KV cache，所以每次重新执行上下文的所有 Block。'],
         '本例上下文 [1,4] → logits[1,1,8]，这里不声称任何真实 logits 数值。','gen',subflow='forward'),
    node('G06','末位置分数除以温度','logits[:, -1, :] / temperature',5,'logits[B,1,V]；temperature>0','scaled logits[B,V]',
         ['去掉时间维，并把每个词表分数除以温度。','在分数非退化时，较小正温度使分布更集中，较大温度使分布更平；不是简单取最大值。'],
         '教学 logits=[1,2,0,-1,0.5,-0.5,1.5,-2]；温度 1 时不变，温度 0.5 时每项乘 2。','gen'),
    node('G07','是否启用 top-k？','top_k is not None',6,'scaled logits[B,V]；top_k','选择阈值过滤或保留全部分数',
         ['None 才表示不做 top-k；不是 0。','k 被限制为 min(top_k,V)，但源码没有防止非正 k。'],
         'top_k=2 走 G08；top_k=None 直接走 G09。','gen',kind='decision'),
    node('G08','按第 k 大分数过滤','低于阈值的分数置 −∞',7,'scaled logits[B,V]','过滤后的 logits[B,V]',
         ['torch.topk 得到第 k 大分数阈值。低于阈值才屏蔽，等于阈值的并列分数会留下。','因此存在并列时，保留数量可能超过 k；不能把这段代码解释成严格只剩 k 个 token。'],
         '教学 V=8,k=2：保留 ID 1 的 2.0 和 ID 6 的 1.5，其余置 -inf。','gen',col=0),
    node('G09','softmax 得到概率','沿词表维归一化',8,'可能已过滤的 logits[B,V]','probs[B,V]，每行之和为 1',
         ['softmax 沿最后一维计算；被置 -inf 的 token 概率为 0。','这一步输出概率；前面所有 logits 都只是未归一化分数。'],
         '上述 top-2 例子的非零概率：P(ID1)=0.622459，P(ID6)=0.377541。这是教学计算，不是模型实测。','gen'),
    node('G10','按概率采一个 token','torch.multinomial(...,1)',9,'probs[B,V]','idx_next[B,1]',
         ['每个 batch 样本独立采一个 token。','采样不等于 argmax；可采到较低但非零概率的 token。具体输出依赖随机状态。'],
         '为了走通教学路径，假设本次抽到 ID1；这不是实际调用 nanoGPT 得到的输出。','gen'),
    node('G11','拼回完整序列','torch.cat((idx,idx_next),dim=1)',10,'完整 idx[B,L]；idx_next[B,1]','完整 idx[B,L+1]；返回循环控制',
         ['拼接对象是完整 idx，不是裁剪后的 idx_cond。','继续迭代时再根据新长度决定上下文窗口。'],
         '原 [[0,1,2,3,4]] + 教学抽样 [[1]] → [[0,1,2,3,4,1]]；完整输出长度为 6。','gen'),
    node('G12','返回完整生成序列','return idx',1,'原始 token + 所有新 token','LongTensor[B,L_initial+N]（N>=0）',
         ['返回值包含原始提示词，不仅包含新生成部分。','sample.py 再把 token ID 转回文本并打印。词表必须与所用权重匹配。'],
         '本例返回 6 个 token；N=0 时原样返回 5 个 token。','gen',col=2,kind='terminal'),
],edges=[edge('G00','G01','初始完整序列'),edge('G01','G12','否：循环结束',from_side='right',to_side='left'),edge('G01','G02','是：执行一轮'),edge('G02','G03','未超长：保留'),edge('G02','G04','超长：裁剪'),edge('G03','G05','idx_cond[B,T]'),edge('G04','G05','idx_cond[B,block_size]'),edge('G05','G06','最后位置 logits'),edge('G06','G07','温度调整后分数'),edge('G07','G08','是：top_k 非 None'),edge('G07','G09','否：保留所有分数',from_side='right',to_side='right',via=[[670,1007],[670,1319]]),edge('G08','G09','过滤后的 logits'),edge('G09','G10','概率 [B,V]'),edge('G10','G11','新 token [B,1]'),edge('G11','G01','下一次迭代',from_side='left',to_side='left',via=[[12,1631],[12,227]],label_at=[95,920])])

flows=[forward,block,attention,generation]


def traces(ids, overrides=None, replace_t=3):
    lookup={n['id']:n for f in flows for n in f['nodes']}
    shapes={'[B,T,3C]':f'[1,{replace_t},12]','[B,T,4C]':f'[1,{replace_t},16]',
            '[B,H,T,T]':f'[1,2,{replace_t},{replace_t}]','[B,H,T,d]':f'[1,2,{replace_t},2]',
            '[B,T,H,d]':f'[1,{replace_t},2,2]','[B,T,C]':f'[1,{replace_t},4]',
            '[B,T,V]':f'[1,{replace_t},8]','[B,T]':f'[1,{replace_t}]',
            '[B,1,C]':'[1,1,4]','[B,1,V]':'[1,1,8]','[B,V]':'[1,8]','[B,1]':'[1,1]'}
    def concrete(text):
        for symbolic,actual in shapes.items(): text=text.replace(symbolic,actual)
        return text
    result=[]
    for i,nid in enumerate(ids):
        n=lookup[nid]
        event=dict(node=nid,input=concrete(n['input']),output=concrete(n['output']))
        if nid=='G01': event.update(input='固定生成次数的循环控制',output='否：退出循环' if ids[i+1:i+2]==['G12'] else '是：执行下一轮')
        if nid=='G07': event.update(input='top_k=2' if 'G08' in ids else 'top_k=None',output='进入 top-k 过滤' if 'G08' in ids else '跳过 top-k')
        if nid=='F06': event.update(input='targets=[[2,3,4]]' if 'F07' in ids else 'targets=None',output='全位置 logits + CE' if 'F07' in ids else '仅末位置投影')
        if nid=='A03': event.update(input='self.flash=True' if 'A04' in ids else 'self.flash=False',output='SDPA 接口路径' if 'A04' in ids else '显式矩阵路径')
        if overrides and nid in overrides:
            event.update(overrides[nid])
        result.append(event)
    return result


block_path=[n['id'] for n in block['nodes']]
manual=['A01','A02','A03','A05','A06','A07','A08']
sdpa=['A01','A02','A03','A04','A07','A08']
train_path=['F01','F02','F03','F04','F05','F06','F07','F09']
infer_path=['F01','F02','F03','F04','F05','F06','F08','F09']
long_path=['G00','G01','G02','G04','G05','G06','G07','G08','G09','G10','G11','G01','G12']
short_path=['G00','G01','G02','G03','G05','G06','G07','G09','G10','G11','G01','G12']
examples=[
    dict(id='train',title='案例 A · 有标签的训练前向',provenance='源码推演 · 未运行 PyTorch',description=['用三个 token 预测右移一位的三个标签。假定该环境没有 SDPA 接口，展开显式注意力路径；这里只追踪一个代表性 Block，实际 n_layer=2 会重复两次。'],input=dict(idx=[[1,2,3]],targets=[[2,3,4]],training=True,dropout=0.0),output='logits[1,3,8] 与标量 loss；具体浮点数依赖实际权重，未测量。',paths={
        'forward':traces(train_path,{'F01':dict(input={'idx':[[1,2,3]],'targets':[[2,3,4]]},output='b=1,t=3'), 'F03':dict(input='idx[1,3]；pos=[0,1,2]',output='tok_emb[1,3,4] + pos_emb[3,4] → x[1,3,4]'),'F07':dict(input='x[1,3,4]；targets=[[2,3,4]]',output='logits[1,3,8]；CE([3,8],[3]) → 标量 loss'),'F09':dict(input='全位置分支结果',output='(logits[1,3,8], scalar loss)')}),
        'block':traces(block_path),'attention':traces(manual,{'A02':dict(input='x[1,3,4]',output='Q,K,V 各 [1,2,3,2]'),'A05':dict(input='Q/K[1,2,3,2]',output='scores[1,2,3,3]；上三角为 -inf'),'A06':dict(input='weights[1,2,3,3]；V[1,2,3,2]',output='y[1,2,3,2]')})}),
    dict(id='long',title='案例 B · 长上下文 + top-k 生成',provenance='源码推演 · 抽样结果为教学假设',description=['原始长度 5 超过窗口 4，只把末尾 4 个 token 送入模型。使用 SDPA 接口分支，top_k=2，生成 1 个 token。教学 logits 和假设抽中的 ID 不来自真实模型权重。'],input=dict(idx=[[0,1,2,3,4]],max_new_tokens=1,block_size=4,temperature=1,top_k=2),output='形状确定为 [1,6]；教学假设抽中 ID1 时，返回 [[0,1,2,3,4,1]]。',paths={
        'generation':traces(long_path,{'G00':dict(input='idx=[[0,1,2,3,4]], N=1, block_size=4, top_k=2',output='进入生成循环'),'G02':dict(input='L=5，窗口=4',output='5>4，走裁剪'),'G04':dict(input=[[0,1,2,3,4]],output=[[1,2,3,4]]),'G05':dict(input=[[1,2,3,4]],output='logits[1,1,8],None'),'G06':dict(input='教学 logits=[1,2,0,-1,0.5,-0.5,1.5,-2]；temperature=1',output='scaled logits 保持不变'),'G08':dict(input='top_k=2；第2大阈值=1.5',output='ID1:2.0；ID6:1.5；其余 -inf'),'G09':dict(input='非屏蔽分数=[2.0,1.5]',output='P(ID1)=0.622459；P(ID6)=0.377541'),'G10':dict(input='教学概率分布',output='教学假设 idx_next=[[1]]，非实测'),'G11':dict(input='完整 idx 长度5 + idx_next长度1',output=[[0,1,2,3,4,1]]),'G12':dict(input='已生成1个token，结束循环',output=[[0,1,2,3,4,1]])},replace_t=4),
        'forward':traces(infer_path,{'F01':dict(input={'idx':[[1,2,3,4]],'targets':None},output='B=1,T=4'),'F08':dict(input='x[1,4,4]',output='x_last[1,1,4] → logits[1,1,8]'),'F09':dict(input='无 targets',output='(logits[1,1,8],None)')},replace_t=4),
        'block':traces(block_path,replace_t=4),'attention':traces(sdpa,{'A02':dict(input='x[1,4,4]',output='Q/K/V 各 [1,2,4,2]'),'A04':dict(input='q,k,v[1,2,4,2]；is_causal=True；dropout_p=0',output='y[1,2,4,2]')},replace_t=4)}),
    dict(id='short',title='案例 C · 短上下文，不做 top-k',provenance='源码推演 · 未运行 PyTorch',description=['输入没有超过窗口，保留整个上下文。top_k=None，因此没有阈值过滤；所有有限分数 token 都参与 softmax。'],input=dict(idx=[[1,2,3]],max_new_tokens=1,temperature=1,top_k=None),output='返回 [1,4] token 张量；具体抽中的 ID 未知。',paths={'generation':traces(short_path,{'G03':dict(input=[[1,2,3]],output=[[1,2,3]]),'G07':dict(input='top_k=None',output='跳过 G08，不做过滤'),'G10':dict(input='完整词表 softmax 概率',output='随机 token[B,1]；没有实测 ID'),'G11':dict(input='idx[1,3] + token[1,1]',output='idx[1,4]'),'G12':dict(input='完成 1 次迭代',output='完整序列[1,4]')}),'forward':traces(infer_path),'block':traces(block_path),'attention':traces(manual)}),
    dict(id='invalid',title='案例 D · 直接 forward 输入过长',provenance='源码推演 · 异常出口',description=['绕过 generate，直接调用 forward。这里没有窗口裁剪，会在长度断言失败后结束。'],input=dict(idx=[[0,1,2,3,4]],block_size=4),output='AssertionError；不返回 logits 或 loss。',paths={'forward':traces(['F01','F02','FX'],{'F01':dict(input=[[0,1,2,3,4]],output='B=1,T=5'),'F02':dict(input='5 <= 4',output=False),'FX':dict(input='T=5, block_size=4',output='AssertionError: Cannot forward sequence of length 5, block size is only 4')},replace_t=5)}),
    dict(id='zero',title='案例 E · 生成 0 个新 token',provenance='源码推演 · 零次循环',description=['max_new_tokens=0，循环体一次也不执行。即使原序列超过窗口，也不会触发 forward 的长度断言，因为根本没有调用它。'],input=dict(idx=[[0,1,2,3,4]],max_new_tokens=0,block_size=4),output=[[0,1,2,3,4]],paths={'generation':traces(['G00','G01','G12'],{'G00':dict(input='idx长度5；N=0',output='进入 for range(0)'),'G01':dict(input='range(0)',output='循环体不执行'),'G12':dict(input=[[0,1,2,3,4]],output=[[0,1,2,3,4]])})}),
]


def softmax(values):
    maximum=max(values)
    exps=[math.exp(v-maximum) for v in values]
    return [v/sum(exps) for v in exps]


def attention_math():
    q=k=[[1,0],[0,1],[1,1]]
    v=[[10,0],[0,20],[30,30]]
    scores=[]; weights=[]; result=[]
    for i, query in enumerate(q):
        row=[sum(a*b for a,b in zip(query,key))/math.sqrt(2) if j<=i else -math.inf for j,key in enumerate(k)]
        probs=softmax(row)
        scores.append([round(x,6) if math.isfinite(x) else '-inf' for x in row])
        weights.append([round(x,6) for x in probs])
        result.append([round(sum(probs[j]*v[j][d] for j in range(3)),6) for d in range(2)])
    assert result[0]==[10.0,0.0]
    assert all(abs(sum(row)-1)<1e-5 for row in weights)
    return dict(Q=q,K=k,V=v,masked_scores=scores,weights=weights,weighted_V=result)


sources=[]
for id,file,a,b,label in [
    ('ln','model.py',18,27,'model.py:18–27 · LayerNorm'),('att','model.py',29,76,'model.py:29–76 · CausalSelfAttention'),
    ('mlp','model.py',78,92,'model.py:78–92 · MLP'),('block','model.py',94,106,'model.py:94–106 · Block'),
    ('init','model.py',108,168,'model.py:108–168 · 配置、初始化与统计'),('fwd','model.py',170,193,'model.py:170–193 · GPT.forward'),
    ('crop','model.py',195,204,'model.py:195–204 · crop_block_size'),('pretrained','model.py',206,261,'model.py:206–261 · from_pretrained'),
    ('opt','model.py',263,287,'model.py:263–287 · configure_optimizers'),('mfu','model.py',289,303,'model.py:289–303 · estimate_mfu'),
    ('gen','model.py',305,330,'model.py:305–330 · generate'),('batch','train.py',114,131,'train.py:114–131 · get_batch'),
    ('update','train.py',290,314,'train.py:290–314 · 梯度累积与更新'),('sampling','sample.py',30,89,'sample.py:30–89 · 编码与生成调用')]:
    lines=(HERE/'source'/file).read_text(encoding='utf-8-sig').splitlines()
    assert len(lines)>=b
    sources.append(dict(id=id,label=label,url=f'https://github.com/karpathy/nanoGPT/blob/{SHA}/{file}#L{a}-L{b}',code='\n'.join(f'{i:>3}  {lines[i-1]}' for i in range(a,b+1)),note='上游 nanoGPT / Andrej Karpathy / MIT；原始版本与许可证一并保存在 source/。'))


def module(name,location,inp,out,details,example,status='源码已读 · 静态解释'):
    return dict(name=name,location=location,input=inp,output=out,details=details,example=example,status=status)


modules=[
    module('GPTConfig 与 GPT.__init__','model.py:108–148','词表、上下文长度、层数、头数、隐藏维度、Dropout、bias 配置','GPT 模块；初始化后的可学习参数',[
        '创建 wte[V,C]、wpe[block_size,C]、Dropout、n_layer 个 Block、最终 LayerNorm 和无 bias 的 lm_head。',
        'wte.weight 与 lm_head.weight 共享同一个参数，不是复制数值后各自更新。普通 Linear 和 Embedding 按 N(0,0.02²) 初始化，Linear 的可选 bias 置零。',
        '以 c_proj.weight 结尾的注意力和 MLP 投影再按标准差 0.02/sqrt(2*n_layer) 特殊初始化。LayerNorm weight 初始为 1，可选 bias 初始为 0。',
        '构造时要求 vocab_size/block_size 非 None；各注意力模块还检查 C 可被 H 整除。bias=False 时 Linear/LayerNorm 不创建偏置。'],
        '教学配置 V=8,C=4,L=2,H=2,block_size=4；共享权重形状为 [8,4]，不要理解成两份独立矩阵。'),
    module('LayerNorm','model.py:18–27','任意末维为 C 的隐藏张量；学习缩放与可选偏置','相同形状的张量','沿最后一维使用 F.layer_norm，epsilon=1e-5。逐 token 归一化，不跨时间位置混合。','[1,3,4] → [1,3,4]。'),
    module('CausalSelfAttention','model.py:29–76','x[B,T,C]，C%H=0','注意力增量[B,T,C]','完整 SDPA / 显式计算分支见 A01–A08；残差加法在 Block 中。','[1,3,4] → Q/K/V[1,2,3,2] → [1,3,4]。'),
    module('MLP','model.py:78–92','隐藏序列[B,T,C]','特征变换[B,T,C]','依次 Linear(C,4C)、GELU、Linear(4C,C)、Dropout；不同 token 独立使用相同参数。','[1,3,4] → [1,3,16] → [1,3,4]。'),
    module('Block','model.py:94–106','隐藏序列[B,T,C]','隐藏序列[B,T,C]','完整 Pre-LN 与两条残差见 B01–B07。','形状不变，但向量内容经过注意力和 MLP 更新。'),
    module('GPT.forward','model.py:170–193','idx[B,T]，可选 targets[B,T]','有标签：[B,T,V]+标量 loss；无标签：[B,1,V]+None','完整分支、长度失败出口、位置查表和层循环见 F01–F09。','idx=[[1,2,3]],targets=[[2,3,4]] → [1,3,8]+loss。'),
    module('crop_block_size','model.py:195–204','新的 block_size，必须不大于当前值','无显式返回值；修改 config、wpe 参数及可能存在的遮罩 buffer',[
        '先断言只允许缩短；截取位置 Embedding 权重并包装成新的 Parameter。',
        '逐层检查是否有手工注意力 bias buffer；有则同步裁剪，没有则跳过。这是修改模型结构，区别于 generate 中临时裁剪 token 输入。'],
        '模型窗口 1024 → 256：wpe 只保留前 256 行；尝试扩大到 2048 会断言失败。'),
    module('from_pretrained','model.py:206–261','gpt2/gpt2-medium/gpt2-large/gpt2-xl；可选 dropout 覆盖','装载预训练权重后的 GPT 对象',[
        '仅允许已列出的四种模型名，override_args 只允许 dropout。按模型名设 L/H/C，固定 V=50257、block_size=1024、bias=True。',
        '初始化本地 GPT，再通过 transformers 加载 GPT2LMHeadModel。过滤注意力遮罩 buffer 后比较状态字典键数量。',
        '遇到 attn.c_attn / attn.c_proj / mlp.c_fc / mlp.c_proj 的 weight，先断言转置形状匹配，再转置拷贝；其余参数要求形状相同后直接拷贝。此处未执行网络下载和加载。'],
        'GPT-2 的 Conv1D 权重若为 [C,3C]，本地 Linear 需要 [3C,C]，因此转置。这个例子不是说所有参数都要转置。'),
    module('configure_optimizers','model.py:263–287','模型参数、weight_decay、learning_rate、betas、device_type','AdamW 优化器对象',[
        '先排除 requires_grad=False 参数。维度 >=2 的参数进入 weight_decay 组，维度 <2 的进入 decay=0 组；所以 Embedding 会衰减。',
        '仅当 AdamW 签名存在 fused 且 device_type==cuda 时传 fused=True；否则使用普通构造。这里只构造优化器，不更新权重。'],
        'wte.weight[8,4] 属衰减组；LayerNorm.weight[4] 属不衰减组；CPU 不启用 fused。'),
    module('_init_weights / get_num_params','model.py:150–168','初始化模块，或 non_embedding 布尔值','初始化函数无返回；统计函数返回参数元素总数',[
        '_init_weights 区分 Linear 与 Embedding；其它类型不由此函数处理。',
        'get_num_params 默认从总参数数减去 wpe 的位置参数；共享 token/lm_head 参数仍保留，因为也承担输出投影。'],
        '教学 wpe[4,4] 有 16 个元素；默认统计会减去这 16 个，不减共享 token 权重。'),
    module('estimate_mfu','model.py:289–303','每次迭代的前后向次数 fwdbwd_per_iter 与耗时 dt>0','相对 A100 BF16 峰值的估算比率',[
        '用参数数、L/H/Q/T 估算 FLOPs，再除实际传入时间和源码硬编码的 312e12 FLOPS。',
        '它不是 GPU 硬件利用率传感器，换设备不会自动换峰值。报告不能把这个估算当成测得的性能。'],
        '即使在其它 GPU 上调用，分母仍是 A100 的常数；本演示没有性能数值。'),
    module('GPT.generate','model.py:305–330','token idx、生成次数、温度、可选 top_k','包含原提示词和新增 token 的完整序列','上下文裁剪、forward、温度、top-k、采样与回边见 G00–G12；只有固定次数停止。','长度 5 + 生成 1 → 返回长度 6，即使模型只看到末尾 4 个。'),
    module('训练端与采样端的连接','train.py:114–131,290–314；sample.py:30–89','已准备的二进制 token 数据，或提示文字与 tokenizer','训练批次/参数更新，或生成文本',[
        'get_batch 的 x 取连续窗口，y 从同一窗口向右偏移 1。训练循环用 model(X,Y)，按累积次数缩放 loss 后 backward，必要时裁剪梯度，再 step/update/zero_grad。',
        'sample.py 从检查点或 GPT-2 初始化，执行 eval；优先读取数据集 meta.pkl 的字符映射，否则假定 GPT-2 编码。支持从 FILE: 提示文件读文本；编码后加 batch 维，调用 generate，再 decode 打印。',
        '本报告完整展开的执行范围是 model.py 核心。train.py 的 DDP、学习率、检查点循环和 sample.py 的全部初始化分支只作连接说明，不声称全仓库均已画完。'],
        '连续 token [1,2,3,4] 构造 x=[1,2,3]、y=[2,3,4]；sample 输入 token[3] 加 batch 维成 [1,3]。',status='连接点已读 · 非全文件覆盖'),
]

data=dict(
    title='读懂 nanoGPT：从输入到生成',
    subtitle='固定版本源码导读 · 完整前向、Block、Attention 与生成循环 · 每一步都有输入输出与可追踪的例子',
    version=f'karpathy/nanoGPT · {SHA} · 分析日期 2026-10-14 · 源码快照未修改',
    summary=[
        'nanoGPT 的模型核心是一套 decoder-only GPT：读入一段 token ID，用因果注意力逐层变换隐藏向量，再预测后续 token。训练时把每个位置的预测与右移标签比较；生成时取最后位置分数，抽样一个 token，再循环。',
        '这份报告先给整体前向图，再展开 Block 和 Attention，最后把它们放回生成循环。流程节点、步骤卡片、案例和固定版本源码互相链接。你可以按图读，也可以选择案例查看它真正经过的分支。'],
    scope=[
        '完整分析 model.py 的全部类与方法；对 forward、Block、Attention、generate 四个核心运行流程逐节点展开，辅助初始化、权重加载、优化器、统计和裁剪方法在模块清单中解释输入输出、步骤和分支。',
        'train.py 与 sample.py 仅用于解释调用连接。未分析所有数据准备器、配置文件、DDP、检查点和编译后端的完整运行流程。不是整个 nanoGPT 仓库的全覆盖报告，也不是当前商业 GPT 产品的内部实现说明。',
        '环境未安装 PyTorch，未运行原始 nanoGPT、训练或下载模型权重。张量形状来自代码推演；数值教学例子由独立的 Python 标准库计算，不冒充模型输出。'],
    config=dict(B=1,T=3,C=4,H=2,head_dim=2,n_layer=2,vocab_size=8,block_size=4,dropout=0.0,bias=True,note='教学微型配置，非上游默认；token ID 为教学构造，不对应真实文本编码。'),
    technologies=[
        dict(name='Token / Position Embedding',problem='离散 ID 怎样成为可计算向量？',principle='Embedding 本质是按 ID 查参数矩阵的行；位置向量提供顺序信息。',usage='wte(idx) + wpe(arange(T))，随后 Dropout。位置 Embedding 是学习参数。',limit='此实现使用学习的绝对位置向量，不是 RoPE。源码：model.py:126–138,174–179。'),
        dict(name='Causal Multi-head Attention',problem='怎样利用上下文，又不偷看答案？',principle='Q/K 决定对可见位置的权重，V 提供被汇总的内容；下三角遮罩禁止看未来。',usage='一次 Linear(C,3C) 产生 Q/K/V，拆 H 个头，再用 SDPA 或显式矩阵路径计算。',limit='SDPA 接口可用不保证实际 FlashAttention 内核；本报告未测速度或显存。源码：model.py:29–76。'),
        dict(name='Pre-LN / Residual / MLP',problem='每层怎样更新特征并保留已有状态？',principle='先归一化再变换，残差相加；MLP 做逐位置非线性特征变换。',usage='x+attn(ln_1(x))；然后 x+mlp(ln_2(x))。MLP 中间维度 4C。',limit='形状不变不代表内容不变；MLP 不直接跨 token 混合。源码：model.py:18–27,78–106。'),
        dict(name='Next-token Loss / Sampling',problem='怎样学习预测，又怎样生成？',principle='训练用右移标签上的交叉熵；生成把 logits 转概率，再采一个 token。',usage='有 targets 算全位置 loss；无 targets 只投影最后位置。generate 加温度与可选 top-k。',limit='这份实现没有 KV cache 或 EOS 早停；训练输出分支由 targets 决定。源码：model.py:170–193,305–330。'),
    ],flows=flows,examples=examples,modules=modules,sources=sources,
    math_notes=[
        dict(title='手算一头注意力：每一步都能复核',explanation=['以下 Q/K/V 是人为构造的小矩阵，不是实际模型激活。仅模拟 d=2、T=3、dropout=0 的一头显式计算，暂不包含 QKV 投影、其它头和输出投影。','第一个 query 只能看自己，所以其输出恰好为第一个 V。第二个 query 只能汇总前两个 V，第三个才能汇总全部三个。'],values=attention_math()),
        dict(title='手算 top-k、概率和一个位置的损失',explanation=['教学 logits 长度为 V=8，与教学配置一致。top_k=2 后只保留分数 2.0 与 1.5（本例没有并列）。','交叉熵例子用相同未过滤 logits，目标 ID1；只是一个位置的 CE，不是整批训练 loss，也不是生成时额外计算 loss。'],values=dict(logits=[1,2,0,-1,0.5,-0.5,1.5,-2],top_k=2,retained_ids=[1,6],retained_probabilities=[round(p,6) for p in softmax([2,1.5])],single_position_CE_target_ID1=round(-math.log(softmax([1,2,0,-1,0.5,-0.5,1.5,-2])[1]),6))),
    ],verification=[
        dict(status='源码确认',title='固定版本、逐行依据',detail=f'源码固定到 {SHA}；model.py 330 行已阅读。源码摘录按原始行号生成，上游许可证保留。'),
        dict(status='独立教学计算',title='注意力与概率数值',detail='标准库计算 softmax、因果注意力与一个位置的 CE，并检查概率和与因果首位置输出。这验证教学算术，不验证 nanoGPT 的 PyTorch 实现。'),
        dict(status='未执行',title='原始模型、训练与推理',detail='当前环境没有 PyTorch；未加载检查点、未执行 nanoGPT、未测精度、速度、显存或 SDPA 后端。所有案例均明确标为源码推演。'),
        dict(status='浏览器与结构检查',title='HTML 与数据格式',detail='发行前由构建脚本检查步骤/边/案例引用，另在浏览器检查桌面、窄屏、路径选择、缩放、源码跳转和打印。具体检查记录见仓库 VALIDATION.md。'),
    ],limitations=[
        '形状推演假设合法 token ID、有效 targets、正温度、合法 top_k 和可用依赖。源码并未对每个非法参数提供友好校验；例如 top_k=0 或 temperature=0 不应当作正常配置。',
        '案例覆盖核心输出分支、注意力实现选择、上下文保留/裁剪、top-k 有/无、长度断言和零次生成。Dropout 非零训练、权重下载、检查点兼容与硬件后端未实测。',
        '生成示例假设抽中某个 ID 只为说明拼接与回路，不能用于评估语言能力。该源码采用 GPT-2 风格结构，与现代大型模型可能存在位置编码、归一化、MLP、缓存等差异。',
    ],footer=f'Built with code-to-report · nanoGPT © 2022 Andrej Karpathy / MIT · Source {SHA[:12]} · No remote runtime resources.')


def main():
    errors=builder.validate(data)
    if errors:
        raise SystemExit('\n'.join(errors))
    (HERE/'report-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'report.html').write_text(builder.render(data),encoding='utf-8')
    print(f'Rebuilt {HERE / "report.html"}; {sum(len(f["nodes"]) for f in flows)} steps, {len(examples)} examples.')


if __name__=='__main__':
    main()
