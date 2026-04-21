/**
 * NeuralEdge - AI Bitcoin Prediction Dashboard
 * Brain_Model_57 Frontend Application
 */

// Configuration
const API_BASE_URL = 'http://localhost:8002';
const REFRESH_INTERVAL = 30000; // 30 seconds

// State
let priceChart = null;
let lastData = null;

// Utility Functions
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value);
}

function formatNumber(value, decimals = 0) {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(value);
}

function formatTime(date) {
    return date.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

function getTrendClass(value) {
    return value >= 0 ? 'trend up' : 'trend down';
}

function getTrendIcon(value) {
    return value >= 0 ? '↑' : '↓';
}

function getSignalColor(signal) {
    switch (signal?.toLowerCase()) {
        case 'accumulate': return '#10b981';
        case 'distribute': return '#ef4444';
        default: return '#6366f1';
    }
}

function getSignalBg(signal) {
    switch (signal?.toLowerCase()) {
        case 'accumulate': return 'rgba(16, 185, 129, 0.15)';
        case 'distribute': return 'rgba(239, 68, 68, 0.15)';
        default: return 'rgba(99, 102, 241, 0.15)';
    }
}

// Chart Functions
function initChart() {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    // Create gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(99, 102, 241, 0.3)');
    gradient.addColorStop(1, 'rgba(99, 102, 241, 0)');

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'BTC/USD',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: gradient,
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointHoverBackgroundColor: '#6366f1',
                pointHoverBorderColor: '#ffffff',
                pointHoverBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(22, 22, 31, 0.95)',
                    titleColor: '#a1a1aa',
                    bodyColor: '#ffffff',
                    borderColor: '#27273a',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return formatCurrency(context.parsed.y);
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(39, 39, 58, 0.5)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#71717a',
                        maxTicksLimit: 8
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(39, 39, 58, 0.5)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#71717a',
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            }
        }
    });
}

function updateChart(data) {
    if (!priceChart) return;

    priceChart.data.labels = data.map(d => {
        const date = new Date(d.date);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    priceChart.data.datasets[0].data = data.map(d => d.price);
    priceChart.update('default');
}

// API Functions
async function fetchAPI(endpoint) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.warn(`API fetch failed for ${endpoint}:`, error);
        return null;
    }
}

async function loadDashboard() {
    try {
        const [btc, signal, fearGreed, engine, history] = await Promise.all([
            fetchAPI('/api/btc'),
            fetchAPI('/api/signal'),
            fetchAPI('/api/fear-greed'),
            fetchAPI('/api/engine'),
            fetchAPI('/api/btc/history/full')
        ]);

        updateUI({ btc, signal, fearGreed, engine, history });
        lastData = { btc, signal, fearGreed, engine, history };
    } catch (error) {
        console.error('Failed to load dashboard:', error);
    }
}

// UI Update Functions
function updateUI(data) {
    const { btc, signal, fearGreed, engine, history } = data;

    // Update BTC Price
    if (btc) {
        document.getElementById('btc-price').textContent = formatCurrency(btc.price);
        document.getElementById('btc-change').textContent = `${btc.change_24h >= 0 ? '+' : ''}${btc.change_24h}%`;
        
        const trendEl = document.getElementById('btc-trend');
        trendEl.className = getTrendClass(btc.change_24h);
        trendEl.querySelector('span:first-child').textContent = getTrendIcon(btc.change_24h);
    }

    // Update Fear & Greed
    if (fearGreed) {
        document.getElementById('fg-value').textContent = fearGreed.value;
        document.getElementById('fg-classification').textContent = fearGreed.classification;
    }

    // Update Signal
    if (signal) {
        document.getElementById('signal-value').textContent = signal.signal;
        document.getElementById('confidence-value').textContent = `${signal.confidence}%`;
        
        const signalDisplay = document.getElementById('signal-display');
        signalDisplay.style.background = getSignalBg(signal.signal);
        signalDisplay.style.borderColor = getSignalColor(signal.signal);
        signalDisplay.querySelector('.signal-value').style.color = getSignalColor(signal.signal);

        const confidenceBar = document.getElementById('confidence-bar');
        confidenceBar.style.width = `${signal.confidence}%`;
        confidenceBar.style.background = `linear-gradient(90deg, ${getSignalColor(signal.signal)} 0%, ${getSignalColor(signal.signal).replace('1', '0.8')})`;
    }

    // Update Engine Metrics
    if (engine) {
        // On-Chain
        document.getElementById('active-addresses').textContent = engine.head1_onchain?.active_wallets || '--';
        document.getElementById('tx-count').textContent = formatNumber(engine.head1_onchain?.tx_count || 0, 0);
        document.getElementById('hash-rate').textContent = engine.head1_onchain?.hash_rate || '--';
        document.getElementById('mvrv-ratio').textContent = engine.head1_onchain?.mvrv_ratio ? engine.head1_onchain.mvrv_ratio.toFixed(2) : '--';
        document.getElementById('exchange-flow').textContent = engine.head1_onchain?.whale_flow || '--';

        const onchainScore = engine.head1_onchain?.score_value || 72;
        document.getElementById('onchain-score').textContent = `${onchainScore}/100`;
        document.getElementById('onchain-bar').style.width = `${onchainScore}%`;

        // Sentiment
        document.getElementById('google-trends').textContent = `${engine.head2_sentiment?.google_trends || '--'}/100`;
        document.getElementById('social-volume').textContent = engine.head2_sentiment?.social_volume || '--';
        document.getElementById('sentiment-score').textContent = engine.head2_sentiment?.sentiment_score || '--';
        document.getElementById('news-sentiment').textContent = engine.head2_sentiment?.news_sentiment || '--';
        document.getElementById('social-momentum').textContent = engine.head2_sentiment?.social_momentum || '--';

        const sentimentScore = engine.head2_sentiment?.score_value || fearGreed?.value || 50;
        document.getElementById('sentiment-bar-score').textContent = `${sentimentScore}/100`;
        document.getElementById('sentiment-bar').style.width = `${sentimentScore}%`;
    }

    // Update Chart
    if (history && history.length > 0) {
        updateChart(history);
    }

    // Update Last Update Time
    document.getElementById('last-update').textContent = formatTime(new Date());
}

// Initialize Application
function init() {
    // Initialize chart
    initChart();

    // Load initial data
    loadDashboard();

    // Set up auto-refresh
    setInterval(loadDashboard, REFRESH_INTERVAL);

    // Update time every second
    setInterval(() => {
        if (lastData) {
            document.getElementById('last-update').textContent = formatTime(new Date());
        }
    }, 1000);

    console.log('NeuralEdge Dashboard initialized');
    console.log('Model: Brain_Model_57');
    console.log('Accuracy: 57.46%');
    console.log('Refresh Interval: 30s');
}

// Start application when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
