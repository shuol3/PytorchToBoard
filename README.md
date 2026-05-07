# PytorchToBoard

本项目用于把固定输入目录中的 PyTorch 音频分类模型恢复、导出、校验，并生成可集成到 nRF5340 固件中的推理代码。

## 项目主流程

1. 在 `NNFramework-main\in\` 中放置模型定义、权重、配置文件和数据目录。
2. 运行 `NNFramework-main\run_model_to_board.py`，完成模型恢复、候选导出、能力分析、候选选择和板端代码生成。
3. 生成结果会写入 `NNFramework-main\artifacts\latest_run\`，板端源码会写入 `pencilv_stage_tflite_end2end\src\generated\`。
4. 在 `pencilv_stage_tflite_end2end\` 中执行 `west build`，即可把生成后的模型代码编译进 nRF5340 固件。

## 目录说明

- `NNFramework-main/`
  当前主部署流水线，负责输入检查、模型恢复、TFLite 导出、候选分析、校验与板端文件生成。
- `pencilv_stage_tflite_end2end/`
  nRF5340 Zephyr 固件工程，负责音频采集、调用生成的推理运行时代码并输出识别结果。
- `pipeline_spec.html`
  项目流程与约束的补充说明页面。

## 固定输入

`NNFramework-main\in\` 当前固定读取以下内容：

- `model.py`
- `train_1s.yaml`
- 唯一的一个 `.pt` 权重文件
- `data/` 原始数据目录
- `calibration_data/` 校准数据目录

如果需要替换模型或数据，直接更新该目录中的对应文件即可。

## 常用命令

执行模型到板端代码的主流程：

```powershell
python .\NNFramework-main\run_model_to_board.py
```

重建校准数据：

```powershell
python .\NNFramework-main\in\prepare_calibration_data.py
```

编译 nRF5340 固件：

```powershell
cd .\pencilv_stage_tflite_end2end
west build -p always -b pencilv/nrf5340/cpuapp -s . -o=-j1 -- -DBOARD_ROOT=.
```

烧录固件：

```powershell
west flash
```

## 输出位置

- 流水线运行产物：`NNFramework-main\artifacts\latest_run\`
- 板端生成代码：`pencilv_stage_tflite_end2end\src\generated\`

## 当前定位

当前仓库的核心用途是“模型部署到板端”和“nRF5340 固件集成验证”。仓库中的说明文件和源码文本建议统一按 UTF-8 维护，以避免中文内容出现乱码。
