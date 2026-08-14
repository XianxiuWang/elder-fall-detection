```
graph TD
    A["🖥️ Web 前端 / Web 看板<br/>Vue.js + ECharts<br/>健康仪表盘 + 告警中心"]
    
    B["⚙️ 后端服务层<br/>FastAPI"]
    B1["用户服务"]
    B2["数据服务"]
    B3["AI 服务"]
    B4["通知服务"]
    
    C["🗄️ 数据存储层"]
    C1["MySQL<br/>业务数据"]
    C2["InfluxDB<br/>时序体征"]
    C3["MinIO<br/>图片/音频"]
    C4["Redis<br/>缓存+队列"]
    
    D["🧠 AI 算法层<br/>核心大脑"]
    D1["CV 视觉模块<br/>姿态/步态/跌倒<br/>人脸/表情<br/>环境风险"]
    D2["语音/NLP 模块<br/>语音识别<br/>诈骗检测<br/>情感分析"]
    D3["LLM 交互模块<br/>认知评估<br/>回忆疗法<br/>声音克隆"]
    D4["主动预测引擎<br/>步态趋势分析<br/>情绪曲线<br/>社交画像"]
    
    E["📦 边缘计算层<br/>Orange Pi 5 Plus / RK3588"]
    E1["ONNX → RKNN 转换<br/>6 TOPS NPU 推理加速"]
    E2["萤石摄像头<br/>视频流接入"]
    E3["🔒 本地推理<br/>隐私数据不出门"]

    A --> B
    B --> B1
    B --> B2
    B --> B3
    B --> B4
    
    B --> C
    C --> C1
    C --> C2
    C --> C3
    C --> C4
    
    B3 --> D
    D --> D1
    D --> D2
    D --> D3
    D --> D4
    
    D1 --> E
    D2 --> E
    D3 --> E
    E --> E1
    E --> E2
    E --> E3
```
