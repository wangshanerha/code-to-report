# 验证记录

日期：2026-10-14。环境：Windows、Python 3.12.14、Node.js + Playwright、本机 Microsoft Edge 无窗口模式。

## 已完成

- 重建 nanoGPT 演示，固定上游提交 `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`。
- 9 项 unittest：有效报告与离线导航、核心图全部节点/边有案例、错误边、决策缺出口、错误路径跳跃、重复 ID/缺证据、脚本注入转义、源码行号逐行对应、精简安装包完整性与许可。
- 浏览器：1512×1060 桌面、390×844 窄屏；未请求远程运行资源，无页面脚本错误，无窄屏页面溢出。
- 验证案例选择同步、正确/错误分支高亮、缩放、节点跳转、引用展开、打印到 PDF，以及关闭 JavaScript 后仍存在全部 38 个步骤卡片。
- 人工目视检查首页、完整前向图、生成图、步骤卡片和窄屏布局。
- 教学计算：检查因果注意力首位置只取第一个 V、各位置概率和接近 1；数值与公式说明一致。
- 技能元数据、内部文件引用和压缩包结构做独立静态检查。

## 限制

- 未安装或执行 PyTorch、nanoGPT、模型权重；没有训练、语言能力、速度或显存测试。
- 例子中的形状为源码推演；采样 ID 的具体结果为教学假设。独立标准库算术不等于执行原始模型。
- 未在独立 WorkBuddy 会话完成导入，也未在第二个 Codex 会话验证自动匹配和端到端报告生成。
- 结构与案例覆盖检查不能证明源码解释完全正确；当前分支覆盖指标只针对报告中的 4 个核心流程，不表示全仓库或所有运行参数覆盖。
- 系统通用 skill 校验器依赖未提供的 PyYAML，因此未使用其成功结果作为依据；本项目运行时与自带测试只需要 Python 标准库。
- PDF 只作为浏览器打印可用性检查；HTML 是主要阅读载体，长流程图可缩放滚动。

## 复查命令

```sh
python examples/nanogpt/build_demo.py
python scripts/build_report.py examples/nanogpt/report-data.json --validate-only
python -m unittest discover -s tests -v
python scripts/package_skill.py
```

浏览器检查使用开发环境中的 Playwright 和本机 Edge；它们不属于 skill 的必需依赖，也不打入安装包。更换宿主后仍应检查文件访问、浏览器能力和安装目录。
