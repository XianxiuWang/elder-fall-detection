**AI智能居家监护系统 API规范.md**

项目：基于多模态 AI 监测的老年人跌倒风险研究  
赛队编号：XH-202617  
文档版本：V1.0  
编制人：钱永佳（项目负责人 / 系统架构）  
编制日期：2026-07-18  
配套材料：全员短期行动指南、项目计划实施书 V2.2

**目录**

1.  通用通信规范

<!-- -->

2.  全局统一返回结构体

<!-- -->

3.  系统整体数据流转架构

<!-- -->

4.  全模块接口详细定义（按开发分工划分）

<!-- -->

5.  接口测试统一规范（谢萌萌专项执行标准）

<!-- -->

6.  团队协作接口开发约束（全员强制执行）

<!-- -->

7.  Mock 并行开发规范（适配全员短期行动指南）

<!-- -->

8.  各成员完整职责补充说明（新增：王纤秀硬件专项权责）

**1. 通用通信规范**

**1.1 传输协议**

统一使用 HTTP/1.1 + JSON 格式交互，所有请求、响应均为标准 JSON；  
图片、视频二进制文件统一存入 MinIO 对象存储，接口仅传递资源 URL
字符串，不直接传输二进制流。

**1.2 接口路由命名规范**

统一固定前缀：/api/v1/业务模块/操作  
遵循 RESTful 请求方式约定：

-   查询列表 / 详情：GET

<!-- -->

-   新增业务数据：POST

<!-- -->

-   修改已有数据：PUT

<!-- -->

-   删除记录：DELETE

**1.3 跨模块通信基础规则**

CV 视觉、NLP 语音、前端页面、边缘硬件、测试模块**仅依据本文档 API
完成交互**，禁止跨模块直接调用代码、直写数据库；无硬件 /
真实算法数据时，全部使用 Mock JSON 先行开发。

**2. 全局统一返回结构体**

**2.1 成功响应（code=200）**

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"code": 200,<br />
"message": "success",<br />
"data": {}<br />
}</td>
</tr>
</tbody>
</table>

说明：data 字段可承载单个对象、数组、空对象，所有业务返回数据统一存放至
data 内。

**2.2 失败响应（客户端 / 服务端异常）**

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"code": 400,<br />
"message": "错误描述信息"<br />
}</td>
</tr>
</tbody>
</table>

状态码定义：

-   400：参数缺失、参数格式错误、无权限、数据不存在

<!-- -->

-   500：后端服务、AI 算法模型、数据库内部异常

**3. 系统整体数据流转架构**

**3.1 视觉跌倒监测链路（硬件链路由王纤秀全权负责）**

萤石摄像头视频流 → Orange Pi 边缘 RKNN 推理 (MediaPipe) →
步态原始数据接口 → FastAPI 后端入库 (InfluxDB 时序库) →
跌倒风险预测接口计算风险分 → 生成告警记录 → Vue 前端 Dashboard
可视化展示

**3.2 NLP 语音心理监护链路**

语音采集 → Whisper ASR 转写文本 → 诈骗识别接口风险判定 / AI
陪伴对话接口情感分析 → LLM 认知评估 / 回忆疗法生成回复 → 对话数据入库 →
前端对话面板 + 情绪曲线展示

**3.3 人脸情绪监测链路**

门口摄像头图像 → 人脸分析接口识别人脸 + 表情 → 情绪时序数据存入 InfluxDB
→ 后端聚合情绪曲线 → 前端心理看板展示

**3.4 健康体征链路**

智能硬件采集心率 / 血压 / 血氧 / 体温 → 健康数据上传接口入库 →
前端健康仪表盘汇总展示

**4. 全模块接口详细定义**

**4.1 老人档案模块（负责人：钱永佳、田靖宇）**

**创建老人档案**

请求方式：POST  
接口地址：/api/v1/elderly/  
请求体：

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"name": "张三",<br />
"age": 75,<br />
"gender": "男"<br />
}</td>
</tr>
</tbody>
</table>

成功返回 data 示例：

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"elderly_id": 1,<br />
"name": "张三",<br />
"age": 75,<br />
"gender": "男",<br />
"create_time": "2026-07-18 16:30:00"<br />
}</td>
</tr>
</tbody>
</table>

**4.2 健康数据模块（负责人：田靖宇）**

**上传老人健康指标**

请求方式：POST  
接口地址：/api/v1/health-record/  
用途：接收智能硬件采集心率、血压、血氧、体温时序数据，写入 InfluxDB
时序数据库  
请求体：

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"elderly_id": 1,<br />
"heart_rate": 80,<br />
"blood_pressure": "120/80",<br />
"blood_oxygen": 98,<br />
"temperature": 36.5<br />
}</td>
</tr>
</tbody>
</table>

**4.3 步态 CV 视觉模块（负责人：王纤秀，含全部硬件对接工作）**

**4.3.1 姿态步态模拟 / 分析结果接收**

请求方式：POST  
接口地址：/api/v1/pose-analysis/simulate  
请求体：

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"elderly_id": 1,<br />
"action_type": "high_risk"<br />
}</td>
</tr>
</tbody>
</table>

action\_type 枚举说明：

-   normal：正常步态

<!-- -->

-   unstable：异常步态

<!-- -->

-   high\_risk：高风险步态  
    返回 data：

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"walking_speed": 0.6,<br />
"step_length": 0.35,<br />
"body_sway": 10,<br />
"balance_score": 55,<br />
"gait_score": 45<br />
}</td>
</tr>
</tbody>
</table>

**4.3.2 原始步态数据上传**

请求方式：POST  
接口地址：/api/v1/gait-data/  
用途：边缘 Orange Pi
推理完成后，上传原始步态参数，后端存储时序数据；硬件视频流接入、模型推理输出由王纤秀负责  
请求体：

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"elderly_id": 1,<br />
"walking_speed": 1.2,<br />
"step_length": 0.65,<br />
"body_sway": 5,<br />
"balance_score": 95<br />
}</td>
</tr>
</tbody>
</table>

**4.3.3 跌倒风险预测计算接口**

请求方式：POST  
接口地址：/api/v1/fall-prediction/run  
用途：后端读取 7
天步态时序数据，计算跌倒风险评分与风险成因；底层步态数据输入源由王纤秀硬件模块提供  
请求体：

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"elderly_id": 1<br />
}</td>
</tr>
</tbody>
</table>

返回 data：

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"risk_score": 85,<br />
"risk_level": "高风险",<br />
"reason": [<br />
"步速持续下降",<br />
"身体晃动幅度增加"<br />
]<br />
}</td>
</tr>
</tbody>
</table>

**4.4 人脸 & 情绪识别模块（负责人：王权瑶，预留接口）**

**人脸图像分析接口**

请求方式：POST  
接口地址：/api/v1/face-analysis/  
请求体：

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"image": "http://minio.local/face/20260718/1.jpg",<br />
"elderly_id": 1<br />
}</td>
</tr>
</tbody>
</table>

返回 data：

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"person": "unknown",<br />
"confidence": 0.9,<br />
"emotion": "sad",<br />
"emotion_confidence": 0.82<br />
}</td>
</tr>
</tbody>
</table>

**4.5 NLP 语音大模型模块（负责人：刘博）**

**4.5.1 电信诈骗文本识别接口**

请求方式：POST  
接口地址：/api/v1/scam-detection/analyze  
请求体：

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"elderly_id": 1,<br />
"text": "公安要求转账到安全账户"<br />
}</td>
</tr>
</tbody>
</table>

返回 data：

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"risk_score": 80,<br />
"risk_level": "高风险"<br />
}</td>
</tr>
</tbody>
</table>

**4.5.2 AI 陪伴对话接口（认知评估 + 回忆疗法）**

请求方式：POST  
接口地址：/api/v1/conversation/chat  
请求体：

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"elderly_id": 1,<br />
"message": "最近儿女不在身边，感觉很孤单"<br />
}</td>
</tr>
</tbody>
</table>

返回 data：

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"emotion": "孤独",<br />
"score": 80,<br />
"reply": "我陪您聊聊天，您年轻的时候有没有难忘的往事呀？"<br />
}</td>
</tr>
</tbody>
</table>

**4.6 告警记录模块（负责人：田靖宇）**

**获取全量告警记录**

请求方式：GET  
接口地址：/api/v1/alarm-record/  
返回 data（数组结构）：

<table>
<tbody>
<tr class="odd">
<td>json<br />
[<br />
{<br />
"type": "跌倒风险",<br />
"level": "紧急",<br />
"status": "未处理",<br />
"elderly_id": 1,<br />
"create_time": "2026-07-18 17:00:00"<br />
}<br />
]</td>
</tr>
</tbody>
</table>

**4.7 前端 Dashboard 大盘模块（负责人：郭轩言）**

**首页综合统计数据接口**

请求方式：GET  
接口地址：/api/v1/ai-analysis/dashboard-summary  
用途：前端可视化总览页面数据源，支撑 ECharts 各类趋势图表  
返回 data：

<table>
<tbody>
<tr class="odd">
<td>json<br />
{<br />
"elderly_count": 10,<br />
"health_score": 85,<br />
"fall_risk_count": 2,<br />
"scam_risk_count": 1,<br />
"alarm_count": 3<br />
}</td>
</tr>
</tbody>
</table>

**5. 接口测试统一规范（谢萌萌专项执行标准）**

所有业务模块开发完成后，必须完整提交以下全套测试资料，缺一不可：

1.  完整接口请求地址

<!-- -->

2.  标准请求 JSON（正常用例、边界用例、异常用例三类）

<!-- -->

3.  标准成功 / 失败返回 JSON 样例

<!-- -->

4.  Postman/ApiPost 接口测试截图

<!-- -->

5.  全部入参、出参字段详细说明文档

<!-- -->

6.  全链路并发压测报告（第 6-7 周系统集成阶段统一执行）

测试覆盖场景：正常参数、空值、超范围数值、并发请求、网络超时；最终输出
Excel 格式测试矩阵。  
硬件设备全链路稳定性测试由谢萌萌配合王纤秀共同完成。

**6. 团队协作接口开发约束（全员强制执行）**

1.  模块隔离：各业务模块仅通过本文档 API
    通信，禁止直接修改其他成员代码、直连数据库；硬件设备输出数据严格遵循本接口规范输出
    JSON。

<!-- -->

2.  接口变更约束：路由、请求 / 返回 JSON
    字段修改必须提前同步钱永佳，更新本规范文档后方可开发；硬件输出格式改动需王纤秀、钱永佳共同确认。

<!-- -->

3.  数据格式统一：全系统仅使用 JSON
    交互，禁止自定义私有传输格式；萤石摄像头、Orange Pi
    硬件推理输出统一适配本文档接口结构。

<!-- -->

4.  代码目录隔离：项目采用 monorepo 架构，cv/、nlp/、backend/、frontend
    独立目录，代码互不侵入；硬件相关代码统一存放于 cv/edge
    目录，由王纤秀维护。

<!-- -->

5.  版本同步：所有成员以本 V1.0
    规范为唯一开发标准，接口实现必须与文档完全一致。

**7. Mock 并行开发规范（适配全员短期行动指南）**

1.  无硬件设备、无真实 AI 推理结果时，全部使用 Mock JSON
    完成页面、接口、算法逻辑开发；王纤秀同步编写硬件模拟输出脚本，提供摄像头视频流
    Mock 数据。

<!-- -->

2.  后端田靖宇统一提供全套 Mock 接口骨架，前端郭轩言、算法王纤秀 /
    王权瑶 / 刘博直接对接 Mock 数据。

<!-- -->

3.  萤石摄像头、Orange Pi
    硬件调试完成后，仅替换底层数据源，接口入参、出参结构无需改动，由王纤秀完成硬件与后端接口联调。

<!-- -->

4.  所有 CV、NLP 算法模块必须先按照本文档定义输出标准
    JSON，再对接边缘端视频 /
    语音流；摄像头硬件输出数据由王纤秀做格式转换适配接口标准。

**8. 各成员完整职责补充说明**

**8.1 王纤秀 完整权责（硬件专项）**

1.  硬件统筹：全权负责萤石摄像头调试、SDK 对接、视频流采集、Orange Pi 5
    Plus（RK3588）边缘硬件环境搭建、刷机、RKNN 工具链部署；

<!-- -->

2.  CV 视觉算法：MediaPipe
    人体姿态提取、步态特征计算、跌倒风险底层推理、YOLO
    目标检测、ONNX→RKNN 模型量化与边缘部署；

<!-- -->

3.  硬件输出适配：摄像头原始视频流处理，标准化输出
    gait-data、pose-analysis 接口规定 JSON 数据；

<!-- -->

4.  配套硬件功能：TTS 语音播报、硬件端声音克隆推理、设备稳定性调试；

<!-- -->

5.  按项目实施书全部 CV +
    边缘硬件相关任务落地，硬件异常、摄像头对接问题统一由王纤秀排查解决；

<!-- -->

6.  硬件联调：后期与田靖宇后端、谢萌萌测试完成硬件 - 后端全链路打通。

**8.2 其余成员原有职责不变**

-   钱永佳：项目总负责人、架构、API 文档统筹、进度管控

<!-- -->

-   田靖宇：后端 FastAPI、数据库、健康 / 告警接口开发

<!-- -->

-   王权瑶：人脸表情 CV 算法、人脸接口、项目书撰写

<!-- -->

-   刘博：ASR 语音、诈骗识别、AI 陪伴 LLM 对话模块

<!-- -->

-   郭轩言：Vue 前端、仪表盘、实时告警可视化

<!-- -->

-   谢萌萌：数据集标注、全模块接口测试、系统压测、演示环境搭建

**文件使用说明**

1.  本文件为项目唯一接口合约，所有开发人员以此为准；

<!-- -->

2.  任何接口改动需由负责人钱永佳统一更新文档并同步全队；

<!-- -->

3.  硬件相关输出格式调整，必须由王纤秀与钱永佳确认后更新本文档；

<!-- -->

4.  测试人员谢萌萌依据第 5 章节规范完成全模块接口、硬件链路验收。
