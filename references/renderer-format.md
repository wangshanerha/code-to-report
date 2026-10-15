# 结构化 HTML 渲染

渲染器仅负责结构校验和排版，不会自动读懂项目。先按源码建立报告数据，再渲染。无需安装第三方 Python 库，Python 3.9+ 即可。

```sh
python scripts/build_report.py report-data.json --validate-only
python scripts/build_report.py report-data.json -o reports/technical-report.html
```

完整实例：`examples/nanogpt/report-data.json`。重新生成演示：

```sh
python examples/nanogpt/build_demo.py
```

## 顶层字段

| 字段 | 类型与用途 |
|---|---|
| title, subtitle, version | 标题、简介、日期/提交/工作区状态等版本文字 |
| summary, scope | 字符串列表：项目目标和明确的分析边界 |
| config | 可选 JSON 对象：贯穿案例的配置 |
| technologies | 对象列表，包含 name/problem/principle/usage/limit |
| flows | 一个或多个完整运行流程；格式见下方 |
| examples | 贯穿案例和逐步输入输出 |
| modules | 模块覆盖清单：name/location/status/input/output/details/example |
| sources | 源码证据：id/label/code/url/note；url 可省略，代码按文本转义 |
| verification | 验证项：status/title/detail |
| limitations | 限制与未核实内容的字符串列表 |
| math_notes | 可选教学计算：title/explanation/values；必须说明教学来源 |
| paper_mapping | 可选字符串列表；复杂论文表格可扩展渲染器或直接编写 HTML |
| footer | 可选页脚，保留第三方署名 |

文字内容用纯文本，details / principle / explanation 等可接受字符串或字符串列表（节点 details 用列表）。原生渲染器偏向中文；其它语言或复杂报告可改模板或直接输出等价 HTML，不必把内容强塞到有限字段中。

## Flow

每个流程有 `id`、`short_title`、`title`、`intro`、`nodes`、`edges`。ID 用 ASCII 字母开头，仅字母/数字/下划线/连字符，所有流程和节点 ID 全局唯一。

每个节点：

```json
{
  "id": "S01",
  "title": "读取输入",
  "subtitle": "idx [B,T]",
  "kind": "step",
  "col": 1,
  "row": 0,
  "input": "LongTensor[B,T]",
  "output": "读取后的尺寸和设备",
  "details": ["解释真实处理步骤、算法、分支与边界"],
  "example": "给出具体值、形状或已明确标注的未知项",
  "sources": ["src-forward"]
}
```

`kind` 为 `step`、`decision` 或 `terminal`；决策至少有两条带条件的出边。可选 `subflow` 指向子流程 ID。辅助控制节点也要说明其对应的真实源码。

布局按网格指定：每列间距 330、每行间距 156，节点左上角为 `(40+col*330,24+row*156)`，节点宽 260、高 94。主路径建议 col=1，分支放 0/2；单列细节图可 col=0。标题控制在约 13 个汉字，副标题避免过长。渲染后必须目视调整，不能只通过数据检查。

每条边：

```json
{"from":"S01","to":"S02","label":"输入合法，继续处理"}
```

可选 `from_side` / `to_side` 为 top/bottom/left/right，默认 bottom/top。跨行、回路或复杂分支可用 `via: [[x,y],...]` 指定 SVG 中间拐点，用 `label_at:[x,y]` 调整标签。确保拐点处于画布内、不穿过其它节点。画布由最大网格列/行计算；需更大边距时扩展 renderer，不用不可读的挤压布局。

## Example

```json
{
  "id":"valid-input",
  "title":"合法输入的完整路径",
  "provenance":"源码推演 / 实际运行 / 教学构造，准确选用",
  "description":["为何选择此例，触发哪些分支"],
  "input":{"value":1},
  "output":{"result":2},
  "paths":{
    "main":[
      {"node":"S01","input":{"value":1},"output":"读取 value=1"},
      {"node":"S02","input":1,"output":2}
    ]
  }
}
```

这里的节点需要与对应 flow 一致。路径列表可重复节点表示循环，连续节点必须存在真实的图边。每次访问分别给出输入输出。未经过的流程不要放空路径；直接不列该 flow，界面会显示未调用。

## 校验能力与边界

检查必填字段、ID、布局单元冲突、边引用、判断节点出口数量、节点证据引用、案例路径连续性。不会证明：源码解释准确、分支穷尽、例子正确、图形连线不重叠、模型实测通过。报告必须再做语义检查和浏览器视觉检查。
