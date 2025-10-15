# ComfyUI_RH_mammothmoda

ComfyUI 自定义节点，用于 MammothModa2 文本生成图像。

## 功能特点

- **文本生成图像**：使用 MammothModa2 进行高质量图像合成
- **性能优化**：支持 INT8 量化实现高效推理
- **Flash Attention**：使用 flash_attention_2 加速

## 安装方法

### 通过 ComfyUI Manager 安装（推荐）
在 ComfyUI Manager 中搜索 "RunningHub Mammothmoda" 并安装。

### 手动安装
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/HM-RunningHub/ComfyUI_RH_mammothmoda.git
cd ComfyUI_RH_mammothmoda
pip install -r requirements.txt
```

## 模型下载

下载 MammothModa2-Preview 模型并放置到：
```
ComfyUI/models/MammothModa2-Preview/
```

模型来源：请查看 MammothModa2 官方仓库获取模型权重。

## 使用说明

1. **加载模型**：使用 "RunningHub Mammothmoda Loader" 节点
2. **生成图像**：连接到 "RunningHub Mammothmoda T2I Sampler" 节点
3. **配置参数**：设置提示词、尺寸（宽度/高度）、步数和引导比例
4. **运行**：执行工作流生成图像

## 节点说明

- **RunningHub Mammothmoda Loader**：加载 MammothModa2 模型
- **RunningHub Mammothmoda T2I Sampler**：从文本提示词生成图像

## 系统要求

- 支持 CUDA 的 GPU（推荐 24GB 显存）
- PyTorch（支持 CUDA）
- flash-attn 包

## 许可证

详见 LICENSE 文件。

