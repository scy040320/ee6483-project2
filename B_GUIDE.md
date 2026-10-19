# B 部分循序渐进操作指南

数据下载、独立环境安装、11次正式训练及评估均已完成。以下保留复现步骤；你无需重复下载或安装。实际成绩见 B_RESULTS.md，最终检查见 B_VALIDATION.md。

## 第1步：获取课程图片

代码ZIP不含图片。课程PDF第1页的数据链接：
https://drive.google.com/file/d/1q0r6yeHQMS17R3wz-s2FIbMR5DAGZK5v/view?usp=sharing

此链接从PDF超链接提取，当前工具无法验证是否能下载。用浏览器登录学校账户打开；如需权限，通过课程NTULearn/老师给的数据入口获取。
不要用Kaggle数据替换：课程明确说明数据与Kaggle版本不完全相同。
下载并解压后，找直接包含 train、val、test 的文件夹，将它的路径提供给我。应有：

```
dataset/
  train/cat/  train/dog/
  val/cat/    val/dog/
  test/1.jpg ... 500.jpg
```

不要挪动现有A/C文件，也不必现在手动安装软件。PyTorch就是运行神经网络的软件库；GPU是加速训练的显卡。先确认电脑和已有环境，再选择安装方式。

## 第2步：环境检查

在独立 b_workspace 文件夹打开终端。以下仅在已经选好Python环境后执行；python应指向该环境。Windows路径用引号包住。

```powershell
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('GPU available:', torch.cuda.is_available())"
```

若报ModuleNotFoundError，说明这个Python里没有相关库；将报错发给我，按实际显卡配置安装，不盲目复制A的CUDA锁文件。
官方安装入口：https://pytorch.org/get-started/locally/
除了torch/torchvision，依赖沿用requirements.txt。代码使用已有较稳定API，不要求强制匹配A的CUDA13.0版本。

## 第3步：核对图片和运行小样本

下文 DATA_PATH 是示例占位符，换成你的真实数据路径；SAMPLE_PATH 换成sampleSubmission.csv完整路径。

```powershell
python -m b_module.preflight --data-dir "DATA_PATH" --sample-submission "SAMPLE_PATH" --verify-images
python -m b_module.train --data-dir "DATA_PATH" --output-dir runs/b_smoke --smoke-per-class 8 --epochs 1 --batch-size 4 --mode head
python -m b_module.evaluate --data-dir "DATA_PATH" --checkpoint runs/b_smoke/best_model.pt --output-dir runs/b_smoke/evaluation
```

首次ResNet18需要联网下载ImageNet权重；失败会停止，不会偷偷改用随机权重。小样本只验证流程，不能写成正式准确率。
smoke subset固定来自公共manifest；不更改完整清单。默认num_workers=0适合Windows。

## 第4步：正式公平对照

先看B_PLAN.md矩阵，估计一轮耗时后再决定是否调整全部实验的统一轮次。以下命令默认10轮、128尺寸、batch32、seed42。

```powershell
python -m b_module.train --data-dir "DATA_PATH" --model baseline_cnn --output-dir runs/b_a0
python -m b_module.train --data-dir "DATA_PATH" --model baseline_cnn --augment --output-dir runs/b_a1
python -m b_module.train --data-dir "DATA_PATH" --mode layer4 --output-dir runs/b_b0
python -m b_module.train --data-dir "DATA_PATH" --mode layer4 --augment --output-dir runs/b_b1
python -m b_module.train --data-dir "DATA_PATH" --mode head --augment --output-dir runs/b_b2
python -m b_module.train --data-dir "DATA_PATH" --mode layer4 --augment --lr 0.0003 --backbone-lr 0.00003 --output-dir runs/b_b3
```

不要一次盲跑所有命令。先完成B1并看日志、耗时、显存和曲线，再按矩阵继续。输出目录存在时会拒绝覆盖，重跑换新名字。
224实验保留其他配置，添加 --image-size 224；如显存不足，所有同组比较统一调整batch，并标注不属于原128公平组。
最终复跑添加 --seed 43 / --seed 44；确认全部完成后用每次最佳验证准确率计算mean和样本std。

## 第5步：验证与比较

对正式候选执行独立评估：

```powershell
python -m b_module.evaluate --data-dir "DATA_PATH" --checkpoint runs/b_b1/best_model.pt --output-dir runs/b_b1/evaluation
python -m b_module.compare runs/b_a0 runs/b_a1 runs/b_b0 runs/b_b1 runs/b_b2 runs/b_b3 --output report_assets/b_team69/b_comparison.csv
```

检查曲线、混淆矩阵、分类指标、正确和错误案例。解释背景干扰、姿态、遮挡等需对实际图片作判断，不能预写实验结论。
对比工具拒绝smoke和未完成运行，显示划分/尺寸/种子/预算差异警告。表中不同模式的backbone_lr配置即使保留也未必使用，应结合mode解释。

## 第6步：最终测试预测

完成验证集选型后，把checkpoint路径换成真实胜出实验：

```powershell
python -m b_module.predict --checkpoint runs/b_b1/best_model.pt --test-dir "DATA_PATH/test" --sample-submission "SAMPLE_PATH" --output-dir runs/b_final_submission
```

实际生成：submission.csv（500行id,label）、test_predictions.csv（概率明细）、provenance.json（模型溯源）。
ID按样例排列，label只能为0或1。若实际测试图片并非1.jpg形式，会停止并报告，需先核对命名再显式调整映射。
不能报告测试准确率；只有真实标签可支持该指标。不要把概率当成课程要求的label。

## 第7步：交付小组

提交B新增代码、操作说明、实验配置/日志、最佳模型、验证结果和图、最终submission、报告中的B贡献说明。
通过小组授权的私有方式传模型；权重/图片/runs不提交Git。远端访问恢复后在b/pretrained分支加入B新增文件，以PR集成。
本工作副本没有远端Git历史，不要把整个副本直接强推到仓库。A/C源文件均无需替换。

## 本次已准备的位置（无需手动重新下载）

- 课程原始压缩包：downloads/datasets.zip，校验与来源记录：downloads/data_source.json。
- 本次解压的高速训练数据：/tmp/ee6483_data/datasets。临时目录清理后可从原始ZIP重新解压。
- 本次隔离Python：/tmp/ee6483_venv/bin/python；这是WSL/Linux环境，不是Windows的python.exe。临时环境清理后按report_assets/b_team69/b_environment_lock.txt重建同平台依赖。
- 正式GPU训练需工具允许本机GPU访问；默认受限环境可能误报CUDA不可用。
- 结果和所有正式日志保存在本工作副本runs下，不在临时目录。
- 完整实验命令记录：report_assets/b_team69/b_experiment_runner_source.txt；实时状态：report_assets/b_team69/b_experiment_runner.log。
- 跨集合完全重复图片的敏感性分析：评估命令添加 --sensitivity-exclude val/cat/cat.12408.jpg。主结果仍使用全部5000张。

用于本次提交的原始样例已复制到 report_assets/b_team69/sampleSubmission.csv，字节与用户上传文件一致；复现时 SAMPLE_PATH 可填写该路径。

比较入口会同时生成 b_comparison.statistics.json，按相同模型、尺寸、增强、学习率和训练预算分组，输出均值、样本标准差及种子清单；JSON中的accuracy单位为0–1，换成百分比需乘100。只有一次运行的组标准差为null。

## 本次最终交付位置

- 实验与案例报告：B_RESULTS.md。
- 最佳模型：runs/b_b4_seed42/best_model.pt（224输入、layer4微调、seed42、第6轮）。
- 最终提交：runs/b_final_submission/submission.csv。
- 训练和评估指标、曲线、混淆矩阵及逐图片预测：各runs目录。
- 完整工作副本保留全部模型；交付ZIP仅包含最终模型权重及所有正式运行的日志/图，避免重复打包大量中间权重。
- ZIP不含1GB原始数据与临时Python环境；课程数据下载地址、SHA256和精确依赖清单已包含。


交接包说明：本包的可提交素材位于report_assets/b_team69/；各实验轻量记录位于其experiments/子目录。模型和所选运行的本地结果位于02_LOCAL_RESULTS/runs/，复制到仓库根目录后才能按原runs路径加载。不要用完整旧仓库覆盖A/C的新版本。
