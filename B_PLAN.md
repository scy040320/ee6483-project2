# Team 69 — B 模块审查与实验计划

审查日期：2026-10-18。来源：ee6483-project2-main (3).zip、两份课程 PDF、sampleSubmission.csv。
GitHub API 返回 404，无法核实远端；以下状态仅针对上传快照。

## 1. 仓库审查

- 文件布局：README.md；requirements.txt / requirements-lock.txt；configs/augmentation.json；data/manifests/cats_dogs_seed42.json；src 下 11 个 Python 文件。
- A：BaselineCNN 代码、数据工具、增强预览、训练/评估/绘图接口已经存在。README 声称小样本流程通过，ZIP 无 runs、checkpoint 或日志，无法独立确认该运行，更不能认定正式实验已完成。
- B：原快照没有预训练模型、独立训练入口或 submission 导出。
- C：原快照没有 CIFAR-10、不平衡选择或两种处理方法；README 只有接入示例。其他成员电脑或远端情况未知。
- 无 .git 历史，不把 ZIP 当成远端最新版。开发使用独立 b_workspace；未推送或修改 main。
- 所有原始文件与 ZIP 逐字节校验一致；sources 及用户下载文件未修改。

## 2. 数据与公共 API

- 清单：train=20000（每类10000）、val=5000（每类2500），路径唯一，cat=0、dog=1。只做清单结构核对；无图片，未做解码或内容重复检查。
- 保留课程划分；测试500张无标签，不用于选模型/学习率/阈值。
- src.data.build_loaders 返回 loaders 字典及 manifest，batch=(images, labels, paths)。支持固定 manifest；不可同时提供 manifest 与按类数量限制。
- train_one_epoch / evaluate 使用 N×2 原始 logits 与 CrossEntropyLoss，指标按样本数加权。evaluate 可返回路径、标签、置信度和类别概率。
- src.train / src.evaluate 只支持 BaselineCNN；B 独立加载 ResNet18，不修改模型工厂。
- 共享预处理：RGB，正方形双线性 resize，ImageNet mean/std；训练增强为翻转0.5、旋转±15°，验证/测试无随机增强。
- ResNet18 官方权重默认推理为短边256后中心裁剪224；本项目使用共享正方形 resize，是有意的协议偏离，不能声称完全沿用官方预处理。128是公平对照；224 resize 另做性能实验。

## 3. 模型与训练

ImageNet IMAGENET1K_V1 ResNet18 → 全局平均池化512维 → Linear(512,2)。
输入 N×3×128×128（可单独做224），输出 N×2；交叉熵；AdamW；固定轮次，无调度器。
骨干冻结时同时冻结相应 BatchNorm running statistics，避免“仅训练分类头”实际改变特征。
head 模式仅更新 fc；layer4 模式更新 layer4+fc；all 仅保留为扩展，不纳入初始搜索。
最佳 checkpoint 按 val accuracy 最大、同准确率 val loss 最小选择；测试前独立 reload 复核。

## 4. 实施前制定的实验矩阵（现已执行，实际结果见 B_RESULTS.md）

全部使用共享完整清单、batch32、seed42、weight decay1e-4、10 epochs、AdamW，无调度。

|ID|模型|训练增强|解冻|尺寸|头学习率|骨干学习率|
|---|---|---|---|---|---|---|
|A0|现有BaselineCNN|无|全模型|128|1e-3|不适用|
|A1|现有BaselineCNN|共享增强|全模型|128|1e-3|不适用|
|B0|预训练ResNet18|无|layer4+fc|128|1e-3|1e-4|
|B1|预训练ResNet18|共享增强|layer4+fc|128|1e-3|1e-4|
|B2|预训练ResNet18|共享增强|仅fc|128|1e-3|不适用|
|B3|预训练ResNet18|共享增强|layer4+fc|128|3e-4|3e-5|
|B4|B1/B3中验证更好配置|共享增强|layer4+fc|224|按胜者|按胜者|

A0/A1是独立重新运行现有A模型的受控对照，结果记为B组织的复跑，不能冒充A已有结果。
B0/B1唯一改变增强；B1/B2改变解冻范围；B1/B3改变学习率；B4改变尺寸。
若胜出的是head模式，应固定head模式再做lr搜索，不直接把layer4结果当head调参。
同轮次不意味着同耗时/同算力，记录参数量和逐轮耗时。ImageNet额外监督信息是方法差异，报告必须披露。
资源允许时最终配置与主要对照复跑seed42/43/44，报告样本标准差及各次指标；不要仅重跑B后宣称A/B统计显著。
验证集反复用于有限搜索，存在选择偏差；先固定矩阵，不根据测试图片改变模型。

## 5. 文件边界与交付

仅新增 b_module/{__init__,model,train,evaluate,predict,submission,compare,preflight}.py 与 B_PLAN.md、B_GUIDE.md、B_VALIDATION.md。
每次实验输出新目录，拒绝覆盖。输出配置、manifest、history.csv、metrics.json、best_model.pt、last_model.pt、learning_curves.png。
评估输出accuracy、loss、逐类precision/recall/F1、macro-F1、混淆矩阵、正确/错误验证案例、预测明细。
预测严格按sampleSubmission的ID顺序找图片，检查缺失/重复/多余ID，导出整数label及独立概率明细、checkpoint哈希。
本节为实施前计划；最终checkpoint与submission.csv位置见B_RESULTS.md。测试无标签，不能自动识别测试“正确/错误”；人工检查必须明确属于人工判断。

## 6. 评分覆盖与限制

PDF：文献20%、数据5%、模型与复现15%、调参5%、验证/提交15%、案例10%、模型与数据处理对比10%、CIFAR-10 10%、不平衡10%。
B支撑迁移模型、调参、验证/预测、案例及对比；文献综述需小组另外完成，CIFAR与不平衡由C负责。B完成不代表全组交付完成。
截止时间按课程PDF为2026-11-13 23:59。

参考：
- https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

## 数据下载后的补充核查

原始字节SHA256发现5组重复，其中1组跨train/val：train/cat/cat.2339.jpg 与 val/cat/cat.12408.jpg。主实验保留官方划分；评估补充排除后者的4999张敏感性准确率。仅检查字节完全相同，未排除感知近似图。详细见report_assets/b_team69/b_data_duplicates.json。

## 实际运行约定

GPU为RTX2060 6GB，正式实验num_workers=4、CPU线程4，所有对照一致。B3在B1/B2中验证更好的冻结模式内搜索学习率。B4按胜出增强配置做224尺寸实验。最终seed42模型由验证指标选定；seed43/44仅用于稳定性评估，不按复跑结果重新挑提交种子。增强baseline同时复跑43/44；若最终模型为224而baseline为128，多种子结果属于完整方案比较，不代表单因素模型差异。主要模型公平结论仍取128矩阵。


交接包说明：本包的可提交素材位于report_assets/b_team69/；各实验轻量记录位于其experiments/子目录。模型和所选运行的本地结果位于02_LOCAL_RESULTS/runs/，复制到仓库根目录后才能按原runs路径加载。不要用完整旧仓库覆盖A/C的新版本。
