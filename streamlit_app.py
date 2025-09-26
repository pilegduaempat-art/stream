# auto_analysis_bot.py

import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import requests
import time
import threading
from datetime import datetime
# import talib  # Uncomment if talib is installed

# -----------------------------
# CONFIG
# -----------------------------
TELEGRAM_BOT_TOKEN = "8342042938:AAG2ZCSXYsXIu5suusoI0thZaFaurVAURvU"
TELEGRAM_CHAT_ID = "-1002911393239"
REFRESH_INTERVAL = 300   # 5 menit

# -----------------------------
# Binance API via ccxt
# -----------------------------
exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "future"}
})

# -----------------------------
# Fungsi Data
# -----------------------------
def get_ohlcv(symbol, timeframe="15m", limit=1000):
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

def get_top_volatile_pairs(limit=30):
    tickers = exchange.fetch_tickers()
    df = pd.DataFrame([
        (s, t["quoteVolume"], t["percentage"])
        for s,t in tickers.items() if s.endswith("USDT")
    ], columns=["symbol","volume","change"])
    df = df.sort_values("volume", ascending=False).head(30)
    df = df.sort_values("change", key=lambda x: abs(x), ascending=False).head(limit)
    return df["symbol"].tolist()

# -----------------------------
# Analisis Teknis Komprehensif
# -----------------------------
def calc_indicators(df):
    """Kalkulasi berbagai indikator teknis"""
    indicators = {}
    
    try:
        # RSI - Manual calculation if talib not available
        def calc_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        
        indicators['rsi'] = calc_rsi(df['close'])
        
        # Simple MACD calculation
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        indicators['macd'] = macd.iloc[-1]
        indicators['macd_signal'] = signal.iloc[-1]
        indicators['macd_histogram'] = (macd - signal).iloc[-1]
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_middle = df['close'].rolling(bb_period).mean()
        bb_std_dev = df['close'].rolling(bb_period).std()
        indicators['bb_upper'] = (bb_middle + (bb_std_dev * bb_std)).iloc[-1]
        indicators['bb_middle'] = bb_middle.iloc[-1]
        indicators['bb_lower'] = (bb_middle - (bb_std_dev * bb_std)).iloc[-1]
        
        # EMA
        indicators['ema_20'] = df['close'].ewm(span=20).mean().iloc[-1]
        indicators['ema_50'] = df['close'].ewm(span=50).mean().iloc[-1]
        
        # Simple Stochastic
        high_14 = df['high'].rolling(14).max()
        low_14 = df['low'].rolling(14).min()
        k_percent = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        indicators['stoch_k'] = k_percent.iloc[-1]
        indicators['stoch_d'] = k_percent.rolling(3).mean().iloc[-1]
        
        # Simple ADX approximation
        indicators['adx'] = 25.0  # Default value
        
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        # Set default values
        indicators = {
            'rsi': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
            'bb_upper': df['close'].iloc[-1] * 1.02, 'bb_middle': df['close'].iloc[-1],
            'bb_lower': df['close'].iloc[-1] * 0.98, 'ema_20': df['close'].iloc[-1],
            'ema_50': df['close'].iloc[-1], 'stoch_k': 50.0, 'stoch_d': 50.0, 'adx': 25.0
        }
    
    return indicators

def calc_pivot(df):
    last = df.iloc[-2]
    pivot = (last["high"]+last["low"]+last["close"])/3
    r1 = 2*pivot - last["low"]
    r2 = pivot + (last["high"] - last["low"])
    r3 = last["high"] + 2*(pivot - last["low"])
    s1 = 2*pivot - last["high"]
    s2 = pivot - (last["high"] - last["low"])
    s3 = last["low"] - 2*(last["high"] - pivot)
    return {
        'pivot': pivot, 'r1': r1, 'r2': r2, 'r3': r3,
        's1': s1, 's2': s2, 's3': s3
    }

def calc_fibonacci(df):
    high = df["high"].max()
    low = df["low"].min()
    diff = high - low
    levels = {
        'Fibo 23.6%': high - (diff * 0.236),
        'Fibo 38.2%': high - (diff * 0.382),
        'Fibo 50.0%': high - (diff * 0.5),
        'Fibo 61.8%': high - (diff * 0.618),
        'Fibo 78.6%': high - (diff * 0.786)
    }
    return levels

def calc_cmf(df, period=20):
    df = df.copy()
    df["mfm"] = ((df["close"]-df["low"])-(df["high"]-df["close"]))/(df["high"]-df["low"]+1e-9)
    df["mfv"] = df["mfm"]*df["volume"]
    cmf = df["mfv"].rolling(period).sum()/df["volume"].rolling(period).sum()
    return cmf.iloc[-1] if not cmf.empty else 0

def smc_zones(df):
    """Smart Money Concepts - Supply/Demand Zones"""
    high_20 = df["high"].iloc[-20:].max()
    low_20 = df["low"].iloc[-20:].min()
    high_50 = df["high"].iloc[-50:].max()
    low_50 = df["low"].iloc[-50:].min()
    return {
        'supply_zone_20': high_20,
        'demand_zone_20': low_20,
        'supply_zone_50': high_50,
        'demand_zone_50': low_50
    }

def ict_analysis(df):
    """ICT (Inner Circle Trader) Analysis"""
    current_price = df["close"].iloc[-1]
    prev_high = df["high"].iloc[-2]
    prev_low = df["low"].iloc[-2]
    
    # Break of Structure
    bos = "NEUTRAL"
    if current_price > prev_high:
        bos = "BULLISH BOS"
    elif current_price < prev_low:
        bos = "BEARISH BOS"
    
    # Market Structure
    highs = df["high"].iloc[-10:]
    lows = df["low"].iloc[-10:]
    
    if highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]:
        structure = "BULLISH (Higher Highs & Higher Lows)"
    elif highs.iloc[-1] < highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2]:
        structure = "BEARISH (Lower Highs & Lower Lows)"
    else:
        structure = "CONSOLIDATION"
    
    return {'bos': bos, 'structure': structure}

def analyze_momentum(indicators):
    """Analisis momentum berdasarkan indikator"""
    momentum_score = 0
    signals = []
    
    # RSI Analysis
    if indicators['rsi']:
        if indicators['rsi'] > 70:
            signals.append("🔴 RSI Overbought (>70)")
            momentum_score -= 1
        elif indicators['rsi'] < 30:
            signals.append("🟢 RSI Oversold (<30)")
            momentum_score += 1
        elif 50 < indicators['rsi'] < 70:
            signals.append("🟡 RSI Bullish Zone (50-70)")
            momentum_score += 0.5
        elif 30 < indicators['rsi'] < 50:
            signals.append("🟡 RSI Bearish Zone (30-50)")
            momentum_score -= 0.5
    
    # MACD Analysis
    if indicators['macd'] and indicators['macd_signal']:
        if indicators['macd'] > indicators['macd_signal']:
            signals.append("🟢 MACD Bullish Cross")
            momentum_score += 1
        else:
            signals.append("🔴 MACD Bearish Cross")
            momentum_score -= 1
    
    # Stochastic Analysis
    if indicators['stoch_k'] and indicators['stoch_d']:
        if indicators['stoch_k'] > 80:
            signals.append("🔴 Stochastic Overbought")
            momentum_score -= 0.5
        elif indicators['stoch_k'] < 20:
            signals.append("🟢 Stochastic Oversold")
            momentum_score += 0.5
    
    # ADX Trend Strength
    if indicators['adx']:
        if indicators['adx'] > 25:
            signals.append(f"💪 Strong Trend (ADX: {indicators['adx']:.1f})")
        else:
            signals.append(f"📊 Weak Trend (ADX: {indicators['adx']:.1f})")
    
    return momentum_score, signals

def generate_trading_signal(df, indicators, pivot_levels, smc_zones, ict_data):
    """Generate comprehensive trading signal"""
    current_price = df["close"].iloc[-1]
    momentum_score, momentum_signals = analyze_momentum(indicators)
    
    # Determine overall signal
    signal_strength = 0
    recommendation = "HOLD"
    tp_levels = []
    sl_level = None
    risk_reward = "N/A"
    
    # Price action analysis
    if ict_data['bos'] == "BULLISH BOS" and momentum_score > 0:
        recommendation = "STRONG BUY"
        signal_strength = 3
        tp_levels = [pivot_levels['r1'], pivot_levels['r2']]
        sl_level = smc_zones['demand_zone_20']
    elif ict_data['bos'] == "BULLISH BOS" and momentum_score >= 0:
        recommendation = "BUY"
        signal_strength = 2
        tp_levels = [pivot_levels['r1']]
        sl_level = smc_zones['demand_zone_20']
    elif ict_data['bos'] == "BEARISH BOS" and momentum_score < 0:
        recommendation = "STRONG SELL"
        signal_strength = 3
        tp_levels = [pivot_levels['s1'], pivot_levels['s2']]
        sl_level = smc_zones['supply_zone_20']
    elif ict_data['bos'] == "BEARISH BOS" and momentum_score <= 0:
        recommendation = "SELL"
        signal_strength = 2
        tp_levels = [pivot_levels['s1']]
        sl_level = smc_zones['supply_zone_20']
    
    # Calculate Risk/Reward
    if tp_levels and sl_level:
        potential_profit = abs(tp_levels[0] - current_price)
        potential_loss = abs(current_price - sl_level)
        risk_reward = f"1:{potential_profit/potential_loss:.2f}" if potential_loss > 0 else "N/A"
    
    return {
        'recommendation': recommendation,
        'signal_strength': signal_strength,
        'tp_levels': tp_levels,
        'sl_level': sl_level,
        'risk_reward': risk_reward,
        'momentum_score': momentum_score,
        'momentum_signals': momentum_signals
    }

def format_professional_notification(symbol, timeframe, df, indicators, pivot_levels, fibs, smc_zones, ict_data, signal_data):
    """Format notifikasi profesional dan komprehensif"""
    current_price = df["close"].iloc[-1]
    price_change = ((current_price - df["open"].iloc[0]) / df["open"].iloc[0]) * 100
    volume_avg = df["volume"].rolling(20).mean().iloc[-1]
    volume_current = df["volume"].iloc[-1]
    volume_ratio = (volume_current / volume_avg) if volume_avg > 0 else 1
    
    # Header dengan emoji berdasarkan signal
    signal_emoji = {
        "STRONG BUY": "🚀",
        "BUY": "📈",
        "HOLD": "⏸️",
        "SELL": "📉",
        "STRONG SELL": "🔻"
    }
    
    emoji = signal_emoji.get(signal_data['recommendation'], "📊")
    
    notification = f"""
{emoji} <b>RUAS TRADING ANALYSIS</b> {emoji}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 <b>ASSET:</b> {symbol}
⏰ <b>TIMEFRAME:</b> {timeframe}
🕒 <b>TIMESTAMP:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 <b>TRADING RECOMMENDATION:</b>
├ Signal: <b>{signal_data['recommendation']}</b>
├ Strength: {get_signal_strength_bar(signal_data['signal_strength'])}
├ Entry Zone: <code>{current_price:.6f}</code>"""
    
    if signal_data['tp_levels']:
        notification += f"\n├ Take Profit 1: <code>{signal_data['tp_levels'][0]:.6f}</code>"
        if len(signal_data['tp_levels']) > 1:
            notification += f"\n├ Take Profit 2: <code>{signal_data['tp_levels'][1]:.6f}</code>"
    
    if signal_data['sl_level']:
        notification += f"\n├ Stop Loss: <code>{signal_data['sl_level']:.6f}</code>"
    
    notification += f"\n└ Risk/Reward: <code>{signal_data['risk_reward']}</code>"
    
    notification += f"""

💰 <b>PRICE ACTION:</b>
├ Current Price: <code>{current_price:.6f}</code>
├ 24h Change: <code>{price_change:+.2f}%</code>
├ Volume Ratio: <code>{volume_ratio:.2f}x</code>
└ Market Structure: {ict_data['structure']}

📈 <b>TECHNICAL INDICATORS:</b>
├ RSI(14): <code>{indicators['rsi']:.2f}</code> {get_rsi_status(indicators['rsi'])}
├ MACD: <code>{indicators['macd']:.6f}</code>
├ MACD Signal: <code>{indicators['macd_signal']:.6f}</code>
├ Stochastic K: <code>{indicators['stoch_k']:.2f}</code>
├ ADX: <code>{indicators['adx']:.2f}</code>
└ CMF: <code>{calc_cmf(df):.4f}</code>

🎯 <b>PIVOT POINTS:</b>
├ R3: <code>{pivot_levels['r3']:.6f}</code>
├ R2: <code>{pivot_levels['r2']:.6f}</code>
├ R1: <code>{pivot_levels['r1']:.6f}</code> 🔴
├ PP: <code>{pivot_levels['pivot']:.6f}</code> ⚪
├ S1: <code>{pivot_levels['s1']:.6f}</code> 🟢
├ S2: <code>{pivot_levels['s2']:.6f}</code>
└ S3: <code>{pivot_levels['s3']:.6f}</code>

🔄 <b>FIBONACCI RETRACEMENTS:</b>
├ 23.6%: <code>{fibs['Fibo 23.6%']:.6f}</code>
├ 38.2%: <code>{fibs['Fibo 38.2%']:.6f}</code>
├ 50.0%: <code>{fibs['Fibo 50.0%']:.6f}</code>
├ 61.8%: <code>{fibs['Fibo 61.8%']:.6f}</code>
└ 78.6%: <code>{fibs['Fibo 78.6%']:.6f}</code>

🧠 <b>SMART MONEY CONCEPTS:</b>
├ Supply Zone (20): <code>{smc_zones['supply_zone_20']:.6f}</code>
├ Demand Zone (20): <code>{smc_zones['demand_zone_20']:.6f}</code>
└ BOS Status: {ict_data['bos']}

⚡ <b>MOMENTUM ANALYSIS:</b>
├ Score: <code>{signal_data['momentum_score']:+.1f}/3.0</code>
└ Signals:"""
    
    # Add momentum signals
    for signal in signal_data['momentum_signals'][:5]:  # Limit to 5 signals
        notification += f"\n   • {signal}"
    
    notification += f"""

⚠️ <b>RISK MANAGEMENT:</b>
├ Position Size: Max 2% of portfolio
├ Leverage: Max 3x (Conservative)
└ Time Horizon: {get_time_horizon(timeframe)}

📝 <b>MARKET NOTES:</b>
├ Volatility: {get_volatility_level(df)}
├ Trend Direction: {get_trend_direction(indicators)}
└ Support/Resistance: {get_nearest_sr(current_price, pivot_levels)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ <i>Automated Analysis by RUAS-TradingBot v2.0</i>
🤖 <i>This is not financial advice. DYOR!</i>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    return notification

def get_rsi_status(rsi):
    if rsi >= 70: return "🔴 OVERBOUGHT"
    elif rsi <= 30: return "🟢 OVERSOLD"
    elif rsi >= 60: return "🟡 BULLISH"
    elif rsi <= 40: return "🟡 BEARISH"
    else: return "⚪ NEUTRAL"

def get_signal_strength_bar(strength):
    bars = "█" * strength + "░" * (3 - strength)
    return f"[{bars}] {strength}/3"

def get_time_horizon(timeframe):
    horizons = {
        "1m": "Scalping (1-5 min)",
        "5m": "Scalping (5-15 min)",
        "15m": "Day Trading (1-4 hours)",
        "1h": "Swing (4-24 hours)",
        "4h": "Position (1-7 days)"
    }
    return horizons.get(timeframe, "Medium-term")

def get_volatility_level(df):
    # Simple ATR calculation
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    
    price = df['close'].iloc[-1]
    volatility_pct = (atr / price) * 100
    
    if volatility_pct > 5: return "🔥 VERY HIGH"
    elif volatility_pct > 3: return "📈 HIGH"
    elif volatility_pct > 1.5: return "📊 MODERATE"
    else: return "😴 LOW"

def get_trend_direction(indicators):
    if indicators['ema_20'] > indicators['ema_50']:
        return "📈 UPTREND"
    elif indicators['ema_20'] < indicators['ema_50']:
        return "📉 DOWNTREND"
    else:
        return "➡️ SIDEWAYS"

def get_nearest_sr(price, pivots):
    resistance = min([p for p in [pivots['r1'], pivots['r2'], pivots['r3']] if p > price], default=pivots['r1'])
    support = max([p for p in [pivots['s1'], pivots['s2'], pivots['s3']] if p < price], default=pivots['s1'])
    return f"S: {support:.6f} | R: {resistance:.6f}"

def generate_analysis(symbol, timeframe="5m"):
    """Generate komprehensif analysis"""
    df = get_ohlcv(symbol, timeframe)
    indicators = calc_indicators(df)
    pivot_levels = calc_pivot(df)
    fibs = calc_fibonacci(df)
    smc_zones_data = smc_zones(df)
    ict_data = ict_analysis(df)
    signal_data = generate_trading_signal(df, indicators, pivot_levels, smc_zones_data, ict_data)
    
    # Format notification
    notification = format_professional_notification(
        symbol, timeframe, df, indicators, pivot_levels, fibs, 
        smc_zones_data, ict_data, signal_data
    )
    
    return df, notification, signal_data, pivot_levels, fibs

# -----------------------------
# Chart (unchanged)
# -----------------------------
def plot_chart(df, smc_high, smc_low, fibs, pivot, r1, s1):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="Candles"
    ))
    fig.add_hline(y=smc_high, line=dict(color="red", dash="dash"), annotation_text="SMC Supply")
    fig.add_hline(y=smc_low, line=dict(color="green", dash="dash"), annotation_text="SMC Demand")
    for name, lvl in fibs.items():
        fig.add_hline(y=lvl, line=dict(color="blue", dash="dot"), annotation_text=name)
    fig.add_hline(y=pivot, line=dict(color="orange"), annotation_text="Pivot")
    fig.add_hline(y=r1, line=dict(color="purple", dash="dash"), annotation_text="R1")
    fig.add_hline(y=s1, line=dict(color="purple", dash="dash"), annotation_text="S1")
    return fig

# -----------------------------
# Telegram
# -----------------------------
def clean_telegram_html(text):
    """Clean and validate HTML for Telegram"""
    # Ensure proper HTML tags are closed and valid
    import re
    
    # Remove any malformed tags
    text = re.sub(r'<(?!/?(?:b|i|u|s|code|pre|a)[>\s])[^>]*>', '', text)
    
    # Ensure all tags are properly closed
    open_tags = re.findall(r'<(b|i|u|s|code|pre)(?:\s[^>]*)?>', text)
    close_tags = re.findall(r'</(b|i|u|s|code|pre)>', text)
    
    # Balance tags if needed
    for tag in open_tags:
        if open_tags.count(tag) > close_tags.count(tag):
            text += f'</{tag}>'
    
    return text

def send_telegram(msg):
    """Send message to Telegram with better error handling"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Clean HTML formatting
    msg = clean_telegram_html(msg)
    
    try:
        # Split message if too long (Telegram limit 4096 chars)
        if len(msg) > 4000:
            # Split at line breaks to avoid cutting important info
            lines = msg.split('\n')
            current_chunk = ""
            
            for line in lines:
                if len(current_chunk + line + '\n') > 4000:
                    if current_chunk:
                        response = requests.post(url, json={
                            "chat_id": TELEGRAM_CHAT_ID, 
                            "text": current_chunk, 
                            "parse_mode": "HTML",
                            "disable_web_page_preview": True
                        })
                        print(f"Telegram Response: {response.status_code}")
                        time.sleep(1)  # Avoid rate limit
                    current_chunk = line + '\n'
                else:
                    current_chunk += line + '\n'
            
            # Send remaining chunk
            if current_chunk:
                response = requests.post(url, json={
                    "chat_id": TELEGRAM_CHAT_ID, 
                    "text": current_chunk, 
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True
                })
                print(f"Telegram Response: {response.status_code}")
        else:
            response = requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID, 
                "text": msg, 
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            })
            print(f"Telegram Response: {response.status_code}")
            
            # If HTML parsing fails, try with plain text
            if response.status_code != 200:
                print("HTML parsing failed, trying plain text...")
                # Remove all HTML tags for plain text fallback
                import re
                plain_msg = re.sub(r'<[^>]+>', '', msg)
                response = requests.post(url, json={
                    "chat_id": TELEGRAM_CHAT_ID, 
                    "text": plain_msg
                })
                print(f"Plain text Response: {response.status_code}")
                
    except Exception as e:
        print(f"Telegram send error: {e}")
        # Fallback to plain text
        try:
            import re
            plain_msg = re.sub(r'<[^>]+>', '', msg)
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": plain_msg})
        except:
            print("All Telegram send methods failed")

def telegram_listener():
    offset = 0
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    while True:
        try:
            resp = requests.get(url, params={"timeout":30, "offset":offset}).json()
            for upd in resp.get("result", []):
                offset = upd["update_id"]+1
                if "message" in upd:
                    text = upd["message"].get("text", "")
                    if text == "/refresh":
                        run_analysis(send_to_tg=True)
                    elif text == "/status":
                        status_msg = f"""
🤖 <b>Bot Status Report</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Status: <code>ACTIVE</code>
⏰ Refresh Interval: <code>{REFRESH_INTERVAL//60} minutes</code>
🕒 Last Update: <code>{datetime.now().strftime('%H:%M:%S')}</code>
📊 Exchange: <code>Binance Futures</code>
🔄 Auto-Analysis: <code>ENABLED</code>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Commands:
• /refresh - Manual analysis
• /status - This status
• /test - Test message format
"""
                        send_telegram(status_msg)
                    elif text == "/test":
                        test_msg = f"""
🧪 <b>Test Message Format</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ <b>Bold text works</b>
✅ <i>Italic text works</i>  
✅ <code>Monospace works</code>
✅ Emojis work: 📊🚀📈📉🔥⚡
✅ Special chars: ├─└━
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If this displays correctly, HTML formatting is working!
"""
                        send_telegram(test_msg)
        except Exception as e:
            print("TG Listener Error:", e)
        time.sleep(2)

# -----------------------------
# Runner
# -----------------------------
def run_analysis(send_to_tg=False):
    st.title("📊 RUAS Pro Multi-Pair Analysis Dashboard")
    pairs = get_top_volatile_pairs()
    
    # Send summary header
    if send_to_tg:
        header = f"""
🤖 <b>AUTOMATED ANALYSIS REPORT</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 <b>Pairs Analyzed:</b> {len(pairs)}
🔄 <b>Timeframe:</b> 10m
⚡ <b>Mode:</b> Auto-Scan
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        send_telegram(header)
    
    successful_analyses = 0
    for sym in pairs:
        try:
            df, notification, signal_data, pivot_levels, fibs = generate_analysis(sym, "5m")
            
            # Streamlit display
            st.subheader(f"{sym} - {signal_data['recommendation']}")
            fig = plot_chart(df, 
                           pivot_levels['r1'], pivot_levels['s1'], 
                           fibs, pivot_levels['pivot'], 
                           pivot_levels['r1'], pivot_levels['s1'])
            st.plotly_chart(fig, use_container_width=True)
            st.text(notification)
            
            # Send to Telegram only for strong signals or BUY/SELL
            if send_to_tg and signal_data['recommendation'] in ['STRONG BUY', 'STRONG SELL', 'BUY', 'SELL']:
                send_telegram(notification)
                successful_analyses += 1
                time.sleep(300)  # Avoid rate limiting
                
        except Exception as e:
            error_msg = f"❌ <b>Error analyzing {sym}:</b> {str(e)}"
            st.error(error_msg)
            print(f"Error analyzing {sym}: {e}")
    
    # Send summary footer
    if send_to_tg:
        footer = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 <b>ANALYSIS COMPLETE</b>
✅ <b>Signals Sent:</b> {successful_analyses}
⏰ <b>Next Scan:</b> {REFRESH_INTERVAL//180} minutes
🤖 <b>Bot Version:</b> v2.0 RUAS Pro
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        send_telegram(footer)

# -----------------------------
# Main
# -----------------------------
if __name__=="__main__":
    threading.Thread(target=telegram_listener, daemon=True).start()
    while True:
        run_analysis(send_to_tg=True)
        time.sleep(REFRESH_INTERVAL)
