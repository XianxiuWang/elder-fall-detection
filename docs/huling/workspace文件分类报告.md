# Workspace 文件分类报告

> 路径: D:\Users\wangxianxiu\.openclaw\workspace\
> 统计时间: 2026-07-11
> 文件总数: 约 160 个

---

## 📂 分类总览

| 类别 | 数量 | 说明 |
|------|------|------|
| 🏠 系统/配置 | 7 | AGENTS, SOUL, USER, TOOLS 等 |
| 🧠 记忆/日志 | 7 | memory/ 目录下的日记和模板 |
| 🔮 紫微斗数分析 | 21 | 命盘分析报告和排盘数据 |
| 🤖 机器学习(护龄模型) | ~50 | 跌倒检测 ML 项目 |
| 🏠 智能家居 STM32 | ~20 | 嵌入式 IoT 项目 |
| 📊 数据处理脚本 | 25+ | 数据提取、文档生成、格式转换 |
| 📋 数据文件 | 15+ | CSV、Excel、JSON、TXT 数据 |
| 📄 文档成品 | 10+ | DOCX / PDF 最终输出 |
| 🔧 临时/过渡文件 | 20+ | tmp、verify、fixup 类文件 |
| 📦 压缩包 | 2 | ZIP 文件 |
| 🔌 嵌入式 C 代码 | ~15 | tmp_src/ STM32 驱动代码 |

---

## 一、🏠 系统/配置文件（7个）

| 文件 | 大小 | 用途 |
|------|------|------|
| AGENTS.md | 8.7KB | Agent 行为规范 |
| BOOTSTRAP.md | 1.5KB | 初始化引导 |
| SOUL.md | 1.7KB | 人格定义 |
| IDENTITY.md | 632B | 身份标识 |
| USER.md | 481B | 用户信息 |
| TOOLS.md | 858B | 工具配置 |
| HEARTBEAT.md | 167B | 心跳任务 |

---

## 二、🧠 记忆/日志文件（7个）

| 文件 | 说明 |
|------|------|
| memory/2026-05-09.md | 日记 |
| memory/2026-05-22.md | 日记 |
| memory/2026-05-23.md | 日记 |
| memory/2026-06-28.md | 日记 |
| memory/hu-ling/2026-05-26-项目对话总结.md | 护龄项目记录 |
| memory/task-templates/sub-session-progress.md | 子会话模板 |
| MEMORY.md | 长期记忆 |

---

## 三、🔮 紫微斗数 / 玄学分析（21个，位于"玄学"目录）

### 3.1 批次1 — 紫微斗数命盘分析
| 文件 | 类型 | 大小 |
|------|------|------|
| 1/紫微斗数命盘分析_2026-06-28.md | MD | 24KB |
| 1/紫微斗数命局分析_2026-06-28.md | MD | 34KB |
| 1/紫微斗数命局分析_女命_2026-06-28.md | MD | 31KB |
| 1/紫微斗数全息命理分析.docx | DOCX | 54KB |
| 1/紫微斗数全息命理分析_女命.docx | DOCX | 54KB |
| 1/双盘合断_两人关系分析_2026-06-28.md | MD | 9KB |
| 1/星系格局综合分析总结.md | MD | 15KB |
| 1/命盘解码分析_完整版.docx | DOCX | 63KB |
| 1/两人命盘综合对比记录_2026-06-28.md | MD | 58KB |
| 1/summary_output.txt | TXT | 18KB |

### 3.2 批次2 — 合婚/配对/时间窗口分析
| 文件 | 类型 | 大小 |
|------|------|------|
| 2/紫微斗数命盘排盘_时间校准.md | MD | 14KB |
| 2/紫微斗数命盘解读.md | MD | 24KB |
| 2/紫微斗数命盘解读_详细.md | MD | 63KB |
| 2/紫微斗数命盘解读_女命.md | MD | 65KB |
| 2/紫微斗数双盘合婚解读.md | MD | 24KB |
| 2/八字合婚综合分析.md | MD | 18KB |
| 2/双盘合婚综合分析_完整版.md | MD | 36KB |
| 2/命盘综合比对分析.md | MD | 21KB |
| 2/2027H2时间窗口分析报告.md | MD | 12KB |
| 2/紫微排盘备注整理_合婚方向.md | MD | 13KB |

### 3.3 批次3 — 本日最新分析
| 文件 | 类型 | 大小 |
|------|------|------|
| 3/紫微斗数命盘分析_乙酉男.md | MD | (今日 1号) |
| 3/紫微斗数命盘分析_乙酉女.md | MD | (今日 2号) |

### 3.4 根目录临时文件
| 文件 | 类型 | 大小 |
|------|------|------|
| 紫微斗数命盘分析_乙酉男.md | MD | 14.8KB |
| 紫微斗数命盘分析_乙酉女.md | MD | 18.2KB |

---

## 四、🤖 机器学习项目 — 护龄模型（~50个文件，位于 huling_model/）

### 4.1 核心 Python 脚本
| 文件 | 用途 |
|------|------|
| train_classifier.py (27KB) | 分类器训练 |
| train_model.py (14KB) | 模型训练 |
| train_two_phase.py (28KB) | 两阶段训练 |
| train_quick.py (5KB) | 快速训练 |
| inference.py (44KB) | 推理引擎 |
| inference_demo.py (17KB) | 推理演示 |
| feature_extractor.py (31KB) | 特征提取 |
| dataset_loader.py (25KB) | 数据加载 |
| config.py (5KB) | 配置管理 |
| export_model.py (12KB) | 模型导出 |
| export_to_c.py (17KB) | C语言导出 |
| compare_models.py (11KB) | 模型对比 |
| generate_report.py (28KB) | 报告生成 |

### 4.2 数据采集/处理
| 文件 | 用途 |
|------|------|
| collect_data.py | 数据采集 |
| data_capture.py | 数据捕获 |
| prepare_data.py | 数据准备 |
| generate_test_data.py | 测试数据生成 |
| process_urfd_images.py | URFD图像处理 |
| extract_main_data.py | 主数据提取 |
| download_datasets.py | 数据集下载 |

### 4.3 测试/验证
| 文件 | 用途 |
|------|------|
| camera_test.py / camera_test2.py | 摄像头测试 |
| check_data.py | 数据检查 |
| quick_test.py | 快速测试 |
| test_inference.py | 推理测试 |
| merge_and_train.py | 合并训练 |
| optimize_mlp.py | MLP优化 |

### 4.4 部署文件 (deploy/)
| 文件 | 用途 |
|------|------|
| deploy/huling_deploy.h | C头文件 |
| deploy/huling_features.c | 特征提取C代码 |
| deploy/random_forest.c (5.5MB!) | 随机森林模型 |
| deploy/random_forest.h | RF头文件 |
| deploy/random_forest_wrapper.c | RF包装器 |
| deploy/scaler_params.h | 标准化参数 |
| deploy/test_data.h | 测试数据 |
| deploy/test_deploy.c | 部署测试 |
| deploy/test_deploy.exe | 编译后测试程序 |

### 4.5 DUOS 板端部署 (deploy/duos/)
| 文件 | 用途 |
|------|------|
| CMakeLists.txt | 构建配置 |
| DEPLOY.md / DEPLOY_DETAILED.md | 部署文档 |
| main.c | DUOS主程序 |
| test_deploy_duos.c | DUOS测试 |
| udp_server.c | UDP服务器 |
| keypoint_bridge.py | 关键点桥接 |
| toolchain-arm.cmake / toolchain-riscv.cmake | 交叉编译工具链 |

### 4.6 Web 应用 (webapp/)
| 文件 | 用途 |
|------|------|
| server.py | Flask服务器 |
| static/js/dashboard.js | 前端仪表盘 |
| templates/index.html | 首页模板 |

### 4.7 模型文件 (models/)
| 文件 | 类型 | 大小 |
|------|------|------|
| pose_classifier.joblib | 模型 | 261KB |
| pose_classifier_backup_20260509.joblib | 备份 | 1.5MB |
| confusion_matrix.png | 混淆矩阵图 | 27KB |

### 4.8 项目文档
| 文件 | 类型 |
|------|------|
| README.md | MD |
| requirements.txt | TXT |
| 项目文档_文件说明.docx | DOCX |
| 项目答辩演示文稿.docx | DOCX |
| 答辩PPT演示稿.md | MD |
| 详细项目计划书.docx / .md | DOCX+MD |
| 修改标注的-RF和MLP.docx | DOCX |

### 4.9 项目文档子目录 (项目文档/)
| 文件 | 类型 |
|------|------|
| 教程-MPU6050教程.md | MD |
| 教程-V2-多模态.md | MD |
| 教程-开发环境文档-V2.md | MD |
| 教程-技术路线图.md | MD |
| 教程-01-环境搭建+MediaPipe.md | MD |
| 详细项目计划书.docx / .md | DOCX+MD |

### 4.10 数据文件 (data/)
| 文件 | 大小 |
|------|------|
| main_data_features_20260509_162752.csv | 2.0MB |
| urfd_features_20260507_204029.csv | 1.9KB |
| urfd_features_20260507_205529.csv | 3.0MB |

### 4.11 Python缓存 (__pycache__/)
| 文件 | 说明 |
|------|------|
| *.cpython-38.pyc × 6 | Python字节码缓存 |

---

## 五、🏠 智能家居 STM32 项目（~20个文件，位于 smart-home-stm32/）

| 文件 | 用途 |
|------|------|
| src.zip (20KB) | 源码打包 |
| src/main.c / main.h | 主程序 |
| src/app/iot_cloud.c / .h | IoT云平台对接 |
| src/hardware/dht11.c / .h | 温湿度传感器 |
| src/hardware/esp8266.c / .h | WiFi模块 |
| src/hardware/oled.c / .h | OLED显示 |
| src/hardware/paj7620.c / .h | 手势传感器 |
| src/hardware/relay.c / .h | 继电器控制 |
| src/hardware/su03t.c / .h | 语音识别模块 |
| src/system/delay.c / .h | 延时函数 |
| src/system/usart.h | 串口 |

---

## 六、📊 数据处理/文档生成脚本（根目录 25+ Python 文件）

### 6.1 主要生成脚本
| 文件 | 大小 | 用途 |
|------|------|------|
| build_complete.py | 67KB | 完整构建 |
| build_docx.py | 62KB | DOCX构建 |
| build_explanations.py | 28KB | 解析构建 |
| generate_doc.py | 34KB | 文档生成 |
| generate_docx.py | 61KB | DOCX生成 |
| generate_duos_doc.py | 88KB | DUOS文档生成 |
| generate_duos_docx.py | 24KB | DUOS DOCX |
| generate_judge_qa.py | 58KB | 评测问答 |
| gen_doc.py / gen_docx.py | 14KB/54KB | 文档生成变体 |
| gen_milkv_doc.py | 38KB | MilkV文档 |
| gen_plan2_docx.py | 48KB | 方案2 DOCX |
| gen_report.py | 25KB | 报告生成 |
| md2docx_defense.py | 27KB | MD转DOCX(答辩) |
| modify_docx_v2.py | 15KB | DOCX修改 |
| fall_detection_analysis.py | 16KB | 跌倒检测分析 |

### 6.2 数据处理/提取脚本
| 文件 | 用途 |
|------|------|
| parse_xlsx.py | Excel解析 |
| explore_data.py / explore_data_peek.py / explore_csv.py | 数据探索 |
| extract_docx.py / extract_docx2.py | DOCX提取 |
| extract_pdf.py / extract_pdfs.py | PDF提取 |
| extract_questions.py | 题目提取 |
| dump_questions.py | 题目导出 |
| group_questions.py | 题目分组 |
| analyze_data.py | 数据分析 |
| insurance_prediction.py | 保险预测 |
| pin_compare.py | 引脚对比 |
| read_pdf.py / read_pdfs.py / read_ican_pdf.py | PDF读取 |
| read_pins.py | 引脚读取 |
| read_files.py | 文件批量读取 |
| get_headers.py | 表头获取 |

### 6.3 修复/验证脚本
| 文件 | 用途 |
|------|------|
| fix_docx.py / fixup_docx.py / fixup2_docx.py / fixup3_docx.py | DOCX修复 |
| fix_pdfs.py | PDF修复 |
| fix_q52_q53.py / fix_quotes.py | 内容修复 |
| verify_doc.py / verify_docx.py / verify2.py / verify3.py / verify_final.py | 验证脚本 |
| check_docx.py / check_missing.py | 检查脚本 |

### 6.4 工具/辅助脚本
| 文件 | 用途 |
|------|------|
| copy_files.py / copy_to_target.py / copy_docx.py / copy_docx2.py | 文件复制 |
| clean_docx.py | DOCX清理 |
| _extract_docx.ps1 | PowerShell提取 |
| find_dir.py | 目录查找 |
| test_read.py | 读测试 |

---

## 七、📋 数据文件（15+）

### 7.1 结构化数据
| 文件 | 类型 | 大小 |
|------|------|------|
| data.xlsx | Excel | 1.8MB |
| eval.xlsx | Excel | 1.3MB |
| data_parsed.csv | CSV | 1.0MB |
| eval_parsed.csv | CSV | 697KB |
| questions_extracted.json | JSON | 108KB |
| filelist.json | JSON | 915B |
| tmp_doc.json | JSON | 7.5KB |
| tmp_tables.json | JSON | 18KB |

### 7.2 文本数据
| 文件 | 类型 | 大小 |
|------|------|------|
| all_md_content.txt | TXT | 94KB |
| all_questions_dump.txt | TXT | 72KB |
| extracted_text.txt | TXT | 131KB |
| pdf_text.txt | TXT | 18KB |
| hal_sources.txt | TXT | 91KB |
| docx_full_dump.txt | TXT | 43KB |
| docx_verify.txt | TXT | 43KB |
| doc_output.txt | TXT | 17KB |
| main_c_content.txt | TXT | 14KB |
| ican_pdf_text.txt | TXT | 6.6KB |
| tmp_doc.txt | TXT | 53KB |
| competition_notice.txt | TXT | 6KB |
| final_check.txt | TXT | 2.8KB |
| verify2.txt / verify3.txt | TXT | 3KB/4.6KB |
| sample_questions.txt / sample_questions2.txt | TXT | 7.5KB/17KB |

### 7.3 分片文本
| 文件 | 大小 | 说明 |
|------|------|------|
| file1.txt | 12KB | 分片数据1 |
| file2.txt | 12KB | 分片数据2 |
| file3.txt | 88KB | 分片数据3 |

---

## 八、📄 文档成品（根目录 DOCX/PDF）

| 文件 | 类型 | 大小 |
|------|------|------|
| competition_notice.pdf | PDF | 927KB |
| PPT演示_手势识别智能家居_完整版.docx | DOCX | 53KB |
| temp_output.docx | DOCX | 52KB |
| temp_ppt_outline.md | MD | 32KB |

---

## 九、📦 压缩包

| 文件 | 大小 |
|------|------|
| huling_model.zip | 5.5MB |
| huling_model (2).zip | 5.6MB |

---

## 十、🔌 嵌入式 C 源码（tmp_src/）

| 文件 | 用途 |
|------|------|
| Src_main.c (15KB) | STM32主程序 |
| AliESP8266_AliESP8266.c/.h | 阿里云 IoT 连接 |
| dht11_dht11.c/.h | DHT11传感器驱动 |
| KEY_key.c/.h | 按键驱动 |
| Src_gpio.c | GPIO配置 |
| Src_tim.c | 定时器配置 |
| Src_usart.c | 串口配置 |
| Inc_main.h / gpio.h / tim.h / usart.h | 头文件 |

---

## 十一、🗑️ 建议清理的文件

### 临时/过渡文件（可删除）
| 文件 | 原因 |
|------|------|
| tmp_doc.json / tmp_doc.txt / tmp_tables.json | 临时数据 |
| file1.txt / file2.txt / file3.txt | 分片临时数据 |
| fix_*.py / fixup_*.py / clean_docx.py | 一次性修复脚本 |
| copy_*.py / copy_docx*.py | 一次性复制脚本 |
| verify*.py / verify*.txt / check_*.py | 验证脚本及输出 |
| docx_full_dump.txt / docx_verify.txt | 临时dump |
| final_check.txt | 临时检查 |
| temp_output.docx / temp_ppt_outline.md | 临时输出 |
| huling_model (2).zip | 重复备份 |
| huling_model/__pycache__/ | Python缓存 |
| huling_model/项目文档/_docx_text.txt / _all_content.json | 中间产物 |
| *_parsed.csv | 可从Excel重新生成 |
| all_md_content.txt / all_questions_dump.txt | 批量dump |

### 需要归档的文件
| 类别 | 建议 |
|------|------|
| 紫微斗数分析 | 统一移入 玄学/ 目录，按日期/对象分子目录 |
| 护龄项目源码 | 保持 huling_model/ 结构，清理缓存和临时文件 |
| 智能家居项目 | 保持 smart-home-stm32/ 结构 |
| 嵌入式C源码 | tmp_src/ 确认是否需要，考虑归入对应项目 |
| 数据处理脚本 | 按用途分目录：generators/ extractors/ utils/ |

---

## 📊 总体统计

| 类别 | 文件数 | 占用空间(约) |
|------|--------|------------|
| Python 脚本 | 60+ | ~800KB |
| Markdown 文档 | 40+ | ~800KB |
| C/C++ 源码 | 35+ | ~6MB (含5.5MB模型C文件) |
| DOCX | 12+ | ~600KB |
| TXT 数据 | 20+ | ~700KB |
| CSV/Excel | 5 | ~6.5MB |
| JSON | 4 | ~150KB |
| PDF | 1 | ~1MB |
| ZIP | 2 | ~11MB |
| 图片 | 1 | 27KB |
| 系统/配置 | 7 | ~15KB |
| 编译产物 | 2 | ~2.9MB |
| **合计** | **~160个** | **~30MB** |

---

## 🎯 建议整理方案

```
workspace/
├── 01_系统配置/          ← AGENTS, SOUL, USER, TOOLS, HEARTBEAT, MEMORY
├── 02_记忆日志/          ← memory/ 目录
├── 10_玄学分析/          ← 紫微斗数所有报告
│   ├── 2026-06-28/       ← 按日期归档
│   ├── 2026-07-06/
│   └── 2026-07-11/
├── 20_护龄项目/          ← huling_model/ 清理后
├── 30_智能家居/          ← smart-home-stm32/
├── 40_数据处理/          ← Python脚本按子目录
│   ├── generators/       ← generate_*/build_*/gen_*
│   ├── extractors/       ← extract_*/parse_*/read_*
│   └── utils/            ← copy_*/verify_*/fix_*
├── 50_数据文件/          ← Excel/CSV/JSON数据
├── 60_输出成品/          ← 最终DOCX/PDF
├── 90_归档/              ← 不再活跃的旧项目
└── 99_待清理/            ← 确认后可删除的文件
```

---

> 以上分类仅供参考。是否需要对某些文件执行整理操作（如移动、归档、删除临时文件）？请告知。
