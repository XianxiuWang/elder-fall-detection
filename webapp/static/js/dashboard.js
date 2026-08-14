/**
 * 护龄 v3 — 仪表盘前端逻辑
 * =============================
 * Socket.IO 实时数据推送 + Chart.js 趋势图 + 告警管理
 */

// ─── 状态图标映射 ───
const STATE_ICONS = {
    0: '🚶', 1: '🪑', 2: '🛏️', 3: '⏰', 4: '⚠️', 5: '🚨', 6: '📴',
};
const STATE_COLORS = {
    0: 'green', 1: 'green', 2: 'green', 3: 'yellow', 4: 'orange', 5: 'red', 6: 'red',
};

// ─── Socket.IO 连接 ───
const socket = io();

socket.on('connect', () => {
    console.log('✅ WebSocket 已连接');
    document.getElementById('status-dot').className = 'status-dot online';
    document.getElementById('status-text').textContent = '已连接';
});

socket.on('disconnect', () => {
    console.log('❌ WebSocket 断开');
    document.getElementById('status-dot').className = 'status-dot offline';
    document.getElementById('status-text').textContent = '已断开';
});

// 接收状态更新
socket.on('state_update', (data) => {
    updateDashboard(data);
});

// 接收告警（单独推送，优先处理）
socket.on('alert', (alert) => {
    showAlertPopup(alert);
});

// ─── 定时获取初始状态（兜底） ───
async function fetchInitialState() {
    try {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        updateDashboard(data);
    } catch (e) {
        console.log('获取初始状态失败，等待 WebSocket...');
    }
}

// ─── 主更新函数 ───
function updateDashboard(data) {
    if (!data) return;

    const stateId = data.state_id || 6;
    const isOnline = data.is_online;
    const icon = STATE_ICONS[stateId] || '❓';
    const color = STATE_COLORS[stateId] || 'gray';

    // 设备信息
    document.getElementById('device-id').textContent = data.device_id || '—';
    document.getElementById('status-dot').className =
        `status-dot ${isOnline ? 'online' : 'offline'}`;
    document.getElementById('status-text').textContent = isOnline ? '在线' : '离线';

    // 健康指标
    updateVital('hr', data.heart_rate, 'bpm', { low: 50, high: 100 });
    updateVital('spo2', data.spo2, '%', { low: 95, high: 100 });
    updateVital('temp', data.temperature, '°C', { low: 36.0, high: 37.3 });
    updateVital('battery', data.battery, '%', { low: 20, high: 100 });

    // 当前状态
    document.getElementById('current-state-name').textContent =
        `${icon} ${data.state_name || '未知'}`;
    document.getElementById('current-confidence').textContent =
        data.confidence ? ` (${(data.confidence * 100).toFixed(0)}%)` : '';

    document.getElementById('big-state-icon').textContent = icon;
    document.getElementById('big-state-name').textContent = data.state_name || '未知';
    document.getElementById('big-state-conf').textContent =
        data.confidence ? `置信度: ${(data.confidence * 100).toFixed(0)}%` : '';
    document.getElementById('alert-level-text').textContent =
        data.alert_level || '—';

    // 更新时间
    if (data.last_update) {
        const d = new Date(data.last_update * 1000);
        document.getElementById('last-update-time').textContent =
            d.toLocaleTimeString('zh-CN');
    }

    // 告警横幅 (state_id >= 4 显示红色横幅)
    updateAlertBanner(stateId, data.state_name, data.alert_level);

    // 时间线
    if (data.state_timeline) {
        updateTimeline(data.state_timeline);
    }

    // 告警记录
    if (data.alert_history) {
        updateAlertList(data.alert_history);
    }

    // 趋势图
    if (data.vital_history) {
        updateChart(data.vital_history);
    }
    // ── 无人活动计时显示 ──
    updateInactiveTimer(data.inactive_info);
}

// ─── 无人活动计时显示 ───
function updateInactiveTimer(info) {
    const box = document.getElementById('inactive-timer-box');
    if (!info) {
        box.style.display = 'none';
        return;
    }

    if (info.is_inactive && info.daytime_seconds > 0) {
        box.style.display = 'block';
        document.getElementById('inactive-timer').textContent = info.daytime_str || '计算中...';
        document.getElementById('inactive-daytime-range').textContent = info.daytime_range;
        document.getElementById('inactive-threshold').textContent =
            `${info.alert_threshold_hours} 小时`;
    } else if (info.is_inactive) {
        // 刚进入无人状态，还没有累计时间
        box.style.display = 'block';
        document.getElementById('inactive-timer').textContent = '等待计时...';
        document.getElementById('inactive-daytime-range').textContent = info.daytime_range;
        document.getElementById('inactive-threshold').textContent =
            `${info.alert_threshold_hours} 小时`;
    } else {
        box.style.display = 'none';
    }
}

function updateVital(type, value, unit, range) {
    const elValue = document.getElementById(`${type}-value`);
    const elSub = document.getElementById(`${type}-sub`);

    if (value === null || value === undefined) {
        elValue.textContent = '—';
        elSub.textContent = '';
        elSub.style.color = 'var(--text-secondary)';
        return;
    }

    elValue.textContent = value;

    // 异常判断
    if (range.low !== undefined && value < range.low) {
        elSub.textContent = `⬇ 偏低`;
        elSub.style.color = 'var(--warning)';
    } else if (range.high !== undefined && value > range.high) {
        elSub.textContent = `⬆ 偏高`;
        elSub.style.color = 'var(--danger)';
    } else {
        elSub.textContent = '正常';
        elSub.style.color = 'var(--success)';
    }
}

// ─── 告警横幅 ───
let lastAlertBannerState = -1;
function updateAlertBanner(stateId, stateName, alertLevel) {
    if (stateId === lastAlertBannerState) return;  // 避免重复闪烁
    lastAlertBannerState = stateId;

    const banner = document.getElementById('alert-banner');
    const bannerIcon = document.getElementById('alert-banner-icon');
    const bannerText = document.getElementById('alert-banner-text');

    banner.style.display = 'flex';

    if (stateId === 5) {
        banner.className = 'alert-banner emergency';
        bannerIcon.textContent = '🚨';
        bannerText.textContent = '紧急告警! 检测到跌倒事件，请立即确认老人安全！';
    } else if (stateId === 6) {
        banner.className = 'alert-banner warning';
        bannerIcon.textContent = '⚠️';
        bannerText.textContent = '失联告警: 长时间未检测到人体活动';
    } else if (stateId === 4) {
        banner.className = 'alert-banner warning';
        bannerIcon.textContent = '🟠';
        bannerText.textContent = '异常姿态: 老人姿势异常，请关注';
    } else if (stateId === 3) {
        banner.className = 'alert-banner warning';
        bannerIcon.textContent = '🟡';
        bannerText.textContent = '久坐提醒: 老人已长时间保持坐姿未动';
    } else {
        banner.className = 'alert-banner normal';
        bannerIcon.textContent = '🟢';
        bannerText.textContent = `系统运行正常 — 当前状态: ${stateName}`;
    }
}

// ─── 弹窗告警（紧急事件） ───
function showAlertPopup(alert) {
    // 浏览器通知
    if (Notification.permission === 'granted') {
        new Notification('护龄 告警', {
            body: alert.message,
            icon: '🛡️',
        });
    }

    // 控制台日志
    console.warn(`[告警] ${alert.level}: ${alert.message}`);
}

// 请求通知权限
if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission();
}

// ─── 状态时间线 ───
function updateTimeline(timelineData) {
    const container = document.getElementById('timeline-container');

    if (!timelineData || timelineData.length === 0) {
        container.innerHTML = '<div style="color: var(--text-secondary); text-align: center; padding: 20px;">暂无数据</div>';
        return;
    }

    // 取最近20条，倒序显示
    const recent = timelineData.slice(-20).reverse();
    let html = '';

    for (const item of recent) {
        const icon = STATE_ICONS[item.state_id] || '❓';
        const color = STATE_COLORS[item.state_id] || 'gray';
        html += `
            <div class="timeline-item">
                <span class="timeline-dot ${color}"></span>
                <span class="timeline-time">${item.time}</span>
                <span>${icon} ${item.state_name}</span>
            </div>`;
    }

    container.innerHTML = html;
}

// ─── 告警列表 ───
function updateAlertList(alertHistory) {
    const container = document.getElementById('alert-list');
    const countEl = document.getElementById('alert-count');

    if (!alertHistory || alertHistory.length === 0) {
        container.innerHTML = '<div style="color: var(--text-secondary); text-align: center; padding: 20px;">暂无告警</div>';
        countEl.textContent = '0 条';
        return;
    }

    countEl.textContent = `${alertHistory.length} 条`;

    // 倒序显示最近告警
    const recent = alertHistory.slice(-50).reverse();
    let html = '';

    for (const alert of recent) {
        const levelEmoji = alert.level.includes('🔴') ? '🔴' :
                          alert.level.includes('🟠') ? '🟠' : '🟡';
        html += `
            <div class="alert-list-item">
                <span class="level-icon">${levelEmoji}</span>
                <div>
                    <div>${alert.message}</div>
                    <div class="alert-time">${alert.time}</div>
                </div>
            </div>`;
    }

    container.innerHTML = html;
}

// ─── Chart.js 趋势图 ───
let vitalChart = null;

function initChart() {
    const ctx = document.getElementById('vitalChart').getContext('2d');

    vitalChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: '心率 (bpm)',
                    data: [],
                    borderColor: '#f85149',
                    backgroundColor: 'rgba(248, 81, 73, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    fill: true,
                    yAxisID: 'y',
                },
                {
                    label: 'SpO₂ (%)',
                    data: [],
                    borderColor: '#58a6ff',
                    backgroundColor: 'rgba(88, 166, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    fill: true,
                    yAxisID: 'y',
                },
                {
                    label: '体温 (°C)',
                    data: [],
                    borderColor: '#ff7b72',
                    backgroundColor: 'rgba(255, 123, 114, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    fill: true,
                    yAxisID: 'y1',
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#8b949e',
                        usePointStyle: true,
                        pointStyleWidth: 8,
                    },
                },
            },
            scales: {
                x: {
                    ticks: { color: '#8b949e', maxTicksLimit: 12 },
                    grid: { color: 'rgba(48,54,61,0.5)' },
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    min: 30,
                    max: 120,
                    ticks: { color: '#8b949e' },
                    grid: { color: 'rgba(48,54,61,0.3)' },
                    title: {
                        display: true,
                        text: 'bpm / %',
                        color: '#8b949e',
                    },
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    min: 35,
                    max: 38,
                    ticks: { color: '#ff7b72' },
                    grid: { drawOnChartArea: false },
                    title: {
                        display: true,
                        text: '°C',
                        color: '#ff7b72',
                    },
                },
            },
        },
    });
}

function updateChart(vitalHistory) {
    if (!vitalChart) {
        initChart();
    }

    if (!vitalHistory || !vitalHistory.timestamps) return;

    const labels = vitalHistory.timestamps.slice(-60);  // 最近60个数据点
    const hr = vitalHistory.heart_rate ? vitalHistory.heart_rate.slice(-60) : [];
    const spo2 = vitalHistory.spo2 ? vitalHistory.spo2.slice(-60) : [];
    const temp = vitalHistory.temperature ? vitalHistory.temperature.slice(-60) : [];

    vitalChart.data.labels = labels;
    vitalChart.data.datasets[0].data = hr;
    vitalChart.data.datasets[1].data = spo2;
    vitalChart.data.datasets[2].data = temp.map(t => t !== null ? t : null);

    vitalChart.update('none');
}

// ─── 时钟更新 ───
function updateClock() {
    const now = new Date();
    document.getElementById('current-time').textContent =
        now.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
        });
}

// ─── 启动 ───
fetchInitialState();
updateClock();
setInterval(updateClock, 1000);

console.log('🛡️ 护龄仪表盘已就绪');
console.log('  · Socket.IO 实时连接');
console.log('  · Chart.js 趋势图');
console.log('  · 7 类状态分类: 行走|坐着|躺卧|久坐|异常|跌倒|无人');
