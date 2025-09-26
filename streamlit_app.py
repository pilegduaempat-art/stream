#!/usr/bin/env python3
"""
Complete Standalone Streamlit Crypto Trading Dashboard
Real-time Binance Futures Scanner with AI Trading Recommendations
Now with Telegram Notifications
"""

import streamlit as st
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import json
import logging
import warnings
from datetime import datetime, timedelta
import time
import math
import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums and Data Classes
class MarketSentiment(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH" 
    NEUTRAL = "NEUTRAL"
    VOLATILE = "VOLATILE"

@dataclass
class TradingRecommendation:
    symbol: str
    signal: str  # LONG, SHORT, HOLD
    confidence: float
    entry_price: float
    tp1: float
    tp2: float
    stop_loss: float
    risk_reward_ratio: float
    reasoning: str
    timestamp: datetime

# Telegram Notification System
class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if not self.enabled:
            logger.warning("Telegram notifications disabled - BOT_TOKEN or CHAT_ID not set")
    
    async def send_message(self, message: str) -> bool:
        if not self.enabled:
            return False
            
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Telegram notification sent successfully")
                        return True
                    else:
                        logger.error(f"Telegram notification failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def format_recommendation_message(self, rec: TradingRecommendation) -> str:
        signal_emoji = "🟢" if rec.signal == "LONG" else "🔴" if rec.signal == "SHORT" else "🟡"
        
        message = f"""
🚀 *AI Trading Signal*

{signal_emoji} *{rec.symbol}* - *{rec.signal}*
📊 Confidence: *{rec.confidence:.1%}*
💰 Entry: *${rec.entry_price:.4f}*
🎯 TP1: *${rec.tp1:.4f}*
🎯 TP2: *${rec.tp2:.4f}*
🛑 Stop Loss: *${rec.stop_loss:.4f}*
📈 R:R Ratio: *{rec.risk_reward_ratio:.2f}:1*

🤖 *Analysis:*
_{rec.reasoning}_

⏰ {rec.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        """
        return message.strip()
    
    def format_market_summary(self, recommendations: List[TradingRecommendation], market_data: List[Dict]) -> str:
        high_conf = [r for r in recommendations if r.confidence > 0.7]
        long_signals = len([r for r in recommendations if r.signal == "LONG"])
        short_signals = len([r for r in recommendations if r.signal == "SHORT"])
        avg_volatility = np.mean([pair['volatility_score'] for pair in market_data])
        
        message = f"""
📊 *Market Summary*

🔍 Pairs Analyzed: *{len(market_data)}*
🔥 Avg Volatility: *{avg_volatility:.2f}%*
🟢 LONG Signals: *{long_signals}*
🔴 SHORT Signals: *{short_signals}*
🎯 High Confidence: *{len(high_conf)}*

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        return message.strip()

# Technical Analysis Functions
class TechnicalAnalyzer:
    @staticmethod
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(data).rolling(window=period).mean().values
    
    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(data).ewm(span=period).mean().values
    
    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        delta = pd.Series(data).diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    @staticmethod
    def macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ema_fast = TechnicalAnalyzer.ema(data, fast)
        ema_slow = TechnicalAnalyzer.ema(data, slow)
        macd = ema_fast - ema_slow
        macd_signal = TechnicalAnalyzer.ema(macd, signal)
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sma = TechnicalAnalyzer.sma(data, period)
        std = pd.Series(data).rolling(window=period).std().values
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return pd.Series(tr).rolling(window=period).mean().values
    
    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        plus_dm = pd.Series(high).diff()
        minus_dm = pd.Series(low).diff() * -1
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = TechnicalAnalyzer.atr(high, low, close, 1)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / pd.Series(tr).rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / pd.Series(tr).rolling(window=period).mean())
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx.values
    
    @staticmethod
    def cmf(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 20) -> np.ndarray:
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_multiplier = np.nan_to_num(mf_multiplier)
        mf_volume = mf_multiplier * volume
        
        cmf = pd.Series(mf_volume).rolling(window=period).sum() / pd.Series(volume).rolling(window=period).sum()
        return cmf.values

# Binance API Client
class BinanceAPI:
    BASE_URL = "https://fapi.binance.com"
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 500) -> List:
        url = f"{self.BASE_URL}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def get_24hr_ticker(self, symbol: str = None) -> Union[Dict, List]:
        url = f"{self.BASE_URL}/fapi/v1/ticker/24hr"
        params = {"symbol": symbol} if symbol else {}
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def get_exchange_info(self) -> Dict:
        url = f"{self.BASE_URL}/fapi/v1/exchangeInfo"
        async with self.session.get(url) as response:
            return await response.json()

# Volatility Scanner
class VolatilityScanner:
    def __init__(self):
        self.api = None
    
    async def initialize(self):
        self.api = BinanceAPI()
        await self.api.__aenter__()
    
    async def cleanup(self):
        if self.api:
            await self.api.__aexit__(None, None, None)
    
    async def get_top_volatile_pairs(self, limit: int = 10) -> List[Dict]:
        try:
            # Get exchange info
            exchange_info = await self.api.get_exchange_info()
            active_symbols = []
            
            for symbol_info in exchange_info['symbols']:
                if (symbol_info['status'] == 'TRADING' and 
                    symbol_info['symbol'].endswith('USDT') and
                    symbol_info['contractType'] == 'PERPETUAL'):
                    active_symbols.append(symbol_info['symbol'])
            
            # Get 24hr ticker data
            tickers = await self.api.get_24hr_ticker()
            volatility_data = []
            
            for ticker in tickers:
                symbol = ticker['symbol']
                if symbol in active_symbols:
                    try:
                        price_change = float(ticker['priceChangePercent'])
                        high_price = float(ticker['highPrice'])
                        low_price = float(ticker['lowPrice'])
                        close_price = float(ticker['lastPrice'])
                        volume = float(ticker['volume'])
                        
                        if low_price > 0:
                            true_range = ((high_price - low_price) / low_price) * 100
                        else:
                            true_range = 0
                        
                        volatility_score = (abs(price_change) * 0.6 + true_range * 0.4)
                        
                        volatility_data.append({
                            'symbol': symbol,
                            'price': close_price,
                            'change_24h': price_change,
                            'high_24h': high_price,
                            'low_24h': low_price,
                            'volume': volume,
                            'volatility_score': volatility_score
                        })
                    except:
                        continue
            
            volatility_data.sort(key=lambda x: x['volatility_score'], reverse=True)
            return volatility_data[:limit]
            
        except Exception as e:
            logger.error(f"Error getting volatile pairs: {e}")
            return []

# AI Trading Engine
class AITradingEngine:
    def __init__(self):
        self.api = None
        self.analyzer = TechnicalAnalyzer()
        self.telegram = TelegramNotifier()
    
    async def initialize(self):
        self.api = BinanceAPI()
        await self.api.__aenter__()
    
    async def cleanup(self):
        if self.api:
            await self.api.__aexit__(None, None, None)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            df['sma_20'] = self.analyzer.sma(close, 20)
            df['sma_50'] = self.analyzer.sma(close, 50)
            df['ema_12'] = self.analyzer.ema(close, 12)
            df['ema_26'] = self.analyzer.ema(close, 26)
            
            df['macd'], df['macd_signal'], df['macd_hist'] = self.analyzer.macd(close)
            df['rsi'] = self.analyzer.rsi(close, 14)
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.analyzer.bollinger_bands(close)
            df['atr'] = self.analyzer.atr(high, low, close)
            df['adx'] = self.analyzer.adx(high, low, close)
            df['cmf'] = self.analyzer.cmf(high, low, close, volume)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def analyze_sentiment(self, df: pd.DataFrame) -> Tuple[MarketSentiment, float]:
        try:
            latest = df.iloc[-1]
            score = 0
            
            # RSI analysis
            if latest['rsi'] < 30:
                score += 2
            elif latest['rsi'] > 70:
                score -= 2
            elif 45 < latest['rsi'] < 55:
                score += 1
            
            # MACD analysis
            if latest['macd'] > latest['macd_signal']:
                score += 1
            else:
                score -= 1
            
            # Moving average analysis
            if latest['close'] > latest['sma_20'] > latest['sma_50']:
                score += 2
            elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                score -= 2
            
            # CMF analysis
            if latest['cmf'] > 0.2:
                score += 1
            elif latest['cmf'] < -0.2:
                score -= 1
            
            # Volatility check
            volatility = latest['atr'] / latest['close'] * 100
            if volatility > 5:
                sentiment = MarketSentiment.VOLATILE
            elif score >= 3:
                sentiment = MarketSentiment.BULLISH
            elif score <= -3:
                sentiment = MarketSentiment.BEARISH
            else:
                sentiment = MarketSentiment.NEUTRAL
            
            confidence = min(0.95, 0.5 + abs(score) * 0.1)
            return sentiment, confidence
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return MarketSentiment.NEUTRAL, 0.5
    
    def calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        try:
            recent_data = df.tail(50)
            highs = recent_data['high'].rolling(window=5, center=True).max()
            lows = recent_data['low'].rolling(window=5, center=True).min()
            
            pivot_highs = []
            pivot_lows = []
            
            for i in range(len(recent_data)):
                if recent_data['high'].iloc[i] == highs.iloc[i] and not pd.isna(highs.iloc[i]):
                    pivot_highs.append(recent_data['high'].iloc[i])
                if recent_data['low'].iloc[i] == lows.iloc[i] and not pd.isna(lows.iloc[i]):
                    pivot_lows.append(recent_data['low'].iloc[i])
            
            return {
                'resistance': sorted(pivot_highs, reverse=True)[:3],
                'support': sorted(pivot_lows)[:3]
            }
        except:
            return {'resistance': [], 'support': []}
    
    async def generate_recommendation(self, symbol: str, volatility_data: Dict) -> Optional[TradingRecommendation]:
        try:
            # Get market data
            klines = await self.api.get_klines(symbol, "1h", 100)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Analyze sentiment
            sentiment, confidence = self.analyze_sentiment(df)
            
            # Get current data
            latest = df.iloc[-1]
            current_price = latest['close']
            
            # Calculate S/R levels
            sr_levels = self.calculate_support_resistance(df)
            
            # Decision logic
            signal_type = "HOLD"
            bullish_factors = 0
            bearish_factors = 0
            reasoning_parts = []
            
            # RSI
            rsi = latest['rsi']
            if rsi < 30:
                bullish_factors += 2
                reasoning_parts.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                bearish_factors += 2
                reasoning_parts.append(f"RSI overbought ({rsi:.1f})")
            
            # CMF
            cmf = latest['cmf']
            if cmf > 0.2:
                bullish_factors += 2
                reasoning_parts.append(f"Strong money flow (+{cmf:.3f})")
            elif cmf < -0.2:
                bearish_factors += 2
                reasoning_parts.append(f"Weak money flow ({cmf:.3f})")
            
            # MACD
            if latest['macd'] > latest['macd_signal']:
                bullish_factors += 1
                reasoning_parts.append("MACD bullish")
            else:
                bearish_factors += 1
                reasoning_parts.append("MACD bearish")
            
            # Trend
            if latest['close'] > latest['sma_20'] > latest['sma_50']:
                bullish_factors += 2
                reasoning_parts.append("Uptrend confirmed")
            elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                bearish_factors += 2
                reasoning_parts.append("Downtrend confirmed")
            
            # Volatility
            volatility_score = volatility_data['volatility_score']
            if volatility_score > 10:
                reasoning_parts.append(f"High volatility ({volatility_score:.1f}%)")
            
            # Decision
            net_score = bullish_factors - bearish_factors
            
            if net_score >= 3:
                signal_type = "LONG"
                confidence = min(0.95, 0.6 + (net_score * 0.1))
            elif net_score <= -3:
                signal_type = "SHORT"
                confidence = min(0.95, 0.6 + (abs(net_score) * 0.1))
            else:
                signal_type = "HOLD"
                confidence = 0.4
                reasoning_parts.append("Conflicting signals")
            
            # Calculate levels
            atr = latest['atr']
            
            if signal_type == "LONG":
                entry_price = current_price
                stop_loss = entry_price - (atr * 2)
                tp1 = entry_price + (atr * 1.5)
                tp2 = entry_price + (atr * 3)
                
                if sr_levels['resistance']:
                    nearest_resistance = min([r for r in sr_levels['resistance'] if r > entry_price], default=tp2)
                    if nearest_resistance < tp2:
                        tp2 = nearest_resistance * 0.995
            
            elif signal_type == "SHORT":
                entry_price = current_price
                stop_loss = entry_price + (atr * 2)
                tp1 = entry_price - (atr * 1.5)
                tp2 = entry_price - (atr * 3)
                
                if sr_levels['support']:
                    nearest_support = max([s for s in sr_levels['support'] if s < entry_price], default=tp2)
                    if nearest_support > tp2:
                        tp2 = nearest_support * 1.005
            
            else:
                entry_price = current_price
                stop_loss = current_price * 0.95
                tp1 = current_price * 1.02
                tp2 = current_price * 1.05
            
            # Risk-reward ratio
            if signal_type == "LONG":
                risk = abs(entry_price - stop_loss)
                reward = abs(tp2 - entry_price)
            elif signal_type == "SHORT":
                risk = abs(stop_loss - entry_price)
                reward = abs(entry_price - tp2)
            else:
                risk = reward = 1
            
            rr_ratio = reward / risk if risk > 0 else 0
            
            recommendation = TradingRecommendation(
                symbol=symbol,
                signal=signal_type,
                confidence=confidence,
                entry_price=entry_price,
                tp1=tp1,
                tp2=tp2,
                stop_loss=stop_loss,
                risk_reward_ratio=rr_ratio,
                reasoning=" | ".join(reasoning_parts),
                timestamp=datetime.now()
            )
            
            # Send Telegram notification for high confidence signals
            if signal_type != "HOLD" and confidence > 0.7:
                await self.telegram.send_message(
                    self.telegram.format_recommendation_message(recommendation)
                )
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating recommendation for {symbol}: {e}")
            return None

# Chart Creation Functions
def create_price_chart(df: pd.DataFrame, symbol: str, recommendation: Optional[TradingRecommendation] = None):
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,  # Fixed: changed from shared_xaxis to shared_xaxes
        vertical_spacing=0.05,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=(f'{symbol} Price Chart', 'Volume', 'RSI', 'MACD')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    if 'sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
    
    if 'sma_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', line=dict(color='red')),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'bb_upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper', 
                      line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower', 
                      line=dict(color='gray', dash='dash'),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )
    
    # Add recommendation lines
    if recommendation and recommendation.signal != "HOLD":
        fig.add_hline(y=recommendation.entry_price, line=dict(color="blue", width=2), 
                     annotation_text=f"Entry: ${recommendation.entry_price:.4f}", row=1, col=1)
        fig.add_hline(y=recommendation.tp1, line=dict(color="green", width=1, dash="dash"), 
                     annotation_text=f"TP1: ${recommendation.tp1:.4f}", row=1, col=1)
        fig.add_hline(y=recommendation.tp2, line=dict(color="green", width=2, dash="dash"), 
                     annotation_text=f"TP2: ${recommendation.tp2:.4f}", row=1, col=1)
        fig.add_hline(y=recommendation.stop_loss, line=dict(color="red", width=2, dash="dot"), 
                     annotation_text=f"SL: ${recommendation.stop_loss:.4f}", row=1, col=1)
    
    # Volume
    colors = ['red' if close < open else 'green' for close, open in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        fig.add_hline(y=70, line=dict(color="red", width=1, dash="dash"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="green", width=1, dash="dash"), row=3, col=1)
        fig.add_hline(y=50, line=dict(color="gray", width=1), row=3, col=1)
    
    # MACD
    if 'macd' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='blue')),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='red')),
            row=4, col=1
        )
        fig.add_trace(
            go.Bar(x=df.index, y=df['macd_hist'], name='Histogram', marker_color='gray'),
            row=4, col=1
        )
    
    fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
    return fig

# Main Data Fetching Function
async def get_market_data():
    scanner = VolatilityScanner()
    ai_engine = AITradingEngine()
    
    try:
        await scanner.initialize()
        await ai_engine.initialize()
        
        # Get top volatile pairs
        top_pairs = await scanner.get_top_volatile_pairs(10)
        
        recommendations = []
        detailed_data = {}
        
        for pair_data in top_pairs:
            symbol = pair_data['symbol']
            
            try:
                # Generate recommendation
                recommendation = await ai_engine.generate_recommendation(symbol, pair_data)
                if recommendation:
                    recommendations.append(recommendation)
                
                # Get chart data
                klines = await ai_engine.api.get_klines(symbol, "1h", 100)
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                df = ai_engine.calculate_indicators(df)
                detailed_data[symbol] = df
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Send market summary to Telegram
        if recommendations and ai_engine.telegram.enabled:
            await ai_engine.telegram.send_message(
                ai_engine.telegram.format_market_summary(recommendations, top_pairs)
            )
        
        return top_pairs, recommendations, detailed_data
        
    except Exception as e:
        logger.error(f"Error in get_market_data: {e}")
        return [], [], {}
    finally:
        await scanner.cleanup()
        await ai_engine.cleanup()

# Streamlit App Configuration
st.set_page_config(
    page_title="AI Crypto RUAS Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B35, #F7931E, #FFD23F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .recommendation-card {
        border: 2px solid;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .bullish { border-color: #22c55e; background: rgba(34, 197, 94, 0.1); }
    .bearish { border-color: #ef4444; background: rgba(239, 68, 68, 0.1); }
    .neutral { border-color: #6b7280; background: rgba(107, 114, 128, 0.1); }
    
    .signal-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 0.875rem;
    }
    
    .long { background-color: #22c55e; color: white; }
    .short { background-color: #ef4444; color: white; }
    .hold { background-color: #6b7280; color: white; }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'market_data' not in st.session_state:
    st.session_state.market_data = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'detailed_data' not in st.session_state:
    st.session_state.detailed_data = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# Header
st.markdown('<h1 class="main-header">🤖 RUAS AI Crypto Trading Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Trading Controls")
    
    # Telegram Configuration
    st.markdown("### 📱 Telegram Notifications")
    telegram_enabled = st.checkbox("Enable Telegram Notifications", value=False)
    
    if telegram_enabled:
        bot_token = st.text_input("Bot Token", type="password", help="Get from @BotFather")
        chat_id = st.text_input("Chat ID", help="Your chat ID or channel ID")
        
        if bot_token and chat_id:
            os.environ['TELEGRAM_BOT_TOKEN'] = bot_token
            os.environ['TELEGRAM_CHAT_ID'] = chat_id
            st.success("✅ Telegram configured!")
    
    st.markdown("---")
    
    # Refresh Controls
    st.markdown("### 🔄 Data Refresh")
    auto_refresh = st.checkbox("Auto Refresh (300s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    if st.button("🔄 Refresh Data", type="primary"):
        st.session_state.market_data = None
        st.rerun()
    
    # Filter Controls
    st.markdown("### 🎛️ Filters")
    min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.6, 0.05)
    signal_filter = st.selectbox("Signal Type", ["ALL", "LONG", "SHORT", "HOLD"])
    min_volatility = st.slider("Min Volatility %", 0.0, 50.0, 5.0, 1.0)
    
    st.markdown("---")
    
    # Info
    st.markdown("### ℹ️ Info")
    st.info("""
    **Features:**
    - Real-time market scanning
    - AI-powered recommendations
    - Technical analysis
    - Telegram notifications
    - Risk management
    """)
    
    if st.session_state.last_update:
        st.success(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")

# Auto-refresh logic
if auto_refresh and st.session_state.last_update:
    time_since_update = datetime.now() - st.session_state.last_update
    if time_since_update.seconds >= 300:
        st.session_state.market_data = None
        st.rerun()

# Main content
async def main():
    # Fetch data if needed
    if st.session_state.market_data is None:
        with st.spinner("🔍 Scanning markets and generating AI recommendations..."):
            try:
                market_data, recommendations, detailed_data = await get_market_data()
                st.session_state.market_data = market_data
                st.session_state.recommendations = recommendations
                st.session_state.detailed_data = detailed_data
                st.session_state.last_update = datetime.now()
            except Exception as e:
                st.error(f"❌ Error fetching data: {e}")
                return
    
    market_data = st.session_state.market_data
    recommendations = st.session_state.recommendations
    detailed_data = st.session_state.detailed_data
    
    if not market_data:
        st.error("❌ No market data available")
        return
    
    # Apply filters
    filtered_recommendations = []
    for rec in recommendations:
        if (rec.confidence >= min_confidence and
            (signal_filter == "ALL" or rec.signal == signal_filter)):
            # Find corresponding market data
            market_info = next((m for m in market_data if m['symbol'] == rec.symbol), None)
            if market_info and market_info['volatility_score'] >= min_volatility:
                filtered_recommendations.append(rec)
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Pairs Scanned</h3>
            <h2>{len(market_data)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_conf = len([r for r in filtered_recommendations if r.confidence > 0.7])
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 High Confidence</h3>
            <h2>{high_conf}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        long_signals = len([r for r in filtered_recommendations if r.signal == "LONG"])
        st.markdown(f"""
        <div class="metric-card">
            <h3>🟢 LONG Signals</h3>
            <h2>{long_signals}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        short_signals = len([r for r in filtered_recommendations if r.signal == "SHORT"])
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔴 SHORT Signals</h3>
            <h2>{short_signals}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        avg_volatility = np.mean([pair['volatility_score'] for pair in market_data])
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Avg Volatility</h3>
            <h2>{avg_volatility:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 AI Recommendations", "📊 Market Overview", "📈 Detailed Charts", "🔥 Top Volatile Pairs"])
    
    with tab1:
        st.markdown("## 🤖 AI Trading Recommendations")
        
        if not filtered_recommendations:
            st.warning("No recommendations match your filters.")
        else:
            # Sort by confidence
            filtered_recommendations.sort(key=lambda x: x.confidence, reverse=True)
            
            for rec in filtered_recommendations:
                # Get market data for this symbol
                market_info = next((m for m in market_data if m['symbol'] == rec.symbol), {})
                
                # Determine card style
                if rec.signal == "LONG":
                    card_class = "bullish"
                    signal_class = "long"
                    signal_emoji = "🟢"
                elif rec.signal == "SHORT":
                    card_class = "bearish"
                    signal_class = "short"
                    signal_emoji = "🔴"
                else:
                    card_class = "neutral"
                    signal_class = "hold"
                    signal_emoji = "🟡"
                
                st.markdown(f"""
                <div class="recommendation-card {card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h3>{signal_emoji} {rec.symbol}</h3>
                        <span class="signal-badge {signal_class}">{rec.signal}</span>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                        <div><strong>Confidence:</strong> {rec.confidence:.1%}</div>
                        <div><strong>Entry:</strong> ${rec.entry_price:.4f}</div>
                        <div><strong>TP1:</strong> ${rec.tp1:.4f}</div>
                        <div><strong>TP2:</strong> ${rec.tp2:.4f}</div>
                        <div><strong>Stop Loss:</strong> ${rec.stop_loss:.4f}</div>
                        <div><strong>R:R Ratio:</strong> {rec.risk_reward_ratio:.2f}:1</div>
                        <div><strong>24h Change:</strong> {market_info.get('change_24h', 0):.2f}%</div>
                        <div><strong>Volatility:</strong> {market_info.get('volatility_score', 0):.1f}%</div>
                    </div>
                    
                    <div style="background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 8px;">
                        <strong>🧠 AI Analysis:</strong> {rec.reasoning}
                    </div>
                    
                    <div style="font-size: 0.875rem; color: #6b7280; margin-top: 0.5rem;">
                        Generated: {rec.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("## 📊 Market Overview")
        
        # Create market overview DataFrame
        overview_data = []
        for pair in market_data:
            rec = next((r for r in recommendations if r.symbol == pair['symbol']), None)
            overview_data.append({
                'Symbol': pair['symbol'],
                'Price': f"${pair['price']:.4f}",
                '24h Change': f"{pair['change_24h']:.2f}%",
                'Volatility': f"{pair['volatility_score']:.1f}%",
                'Volume': f"{pair['volume']:,.0f}",
                'AI Signal': rec.signal if rec else "N/A",
                'Confidence': f"{rec.confidence:.1%}" if rec else "N/A"
            })
        
        df_overview = pd.DataFrame(overview_data)
        st.dataframe(df_overview, use_container_width=True)
        
        # Market heatmap
        st.markdown("### 🔥 Volatility Heatmap")
        fig_heatmap = go.Figure(data=go.Scatter(
            x=[p['change_24h'] for p in market_data],
            y=[p['volatility_score'] for p in market_data],
            mode='markers+text',
            text=[p['symbol'].replace('USDT', '') for p in market_data],
            textposition="middle center",
            marker=dict(
                size=[p['volume']/1000000 for p in market_data],
                color=[p['change_24h'] for p in market_data],
                colorscale='RdYlGn',
                showscale=True,
                sizemode='area',
                sizeref=2.*max([p['volume']/1000000 for p in market_data])/(40.**2),
                sizemin=4
            )
        ))
        
        fig_heatmap.update_layout(
            title="Price Change vs Volatility (Bubble size = Volume)",
            xaxis_title="24h Price Change (%)",
            yaxis_title="Volatility Score (%)",
            height=600
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        st.markdown("## 📈 Detailed Technical Analysis")
        
        if detailed_data:
            selected_symbol = st.selectbox(
                "Select Symbol for Analysis:",
                options=list(detailed_data.keys()),
                key="chart_selector"
            )
            
            if selected_symbol and selected_symbol in detailed_data:
                df = detailed_data[selected_symbol]
                rec = next((r for r in recommendations if r.symbol == selected_symbol), None)
                
                # Create and display chart
                fig = create_price_chart(df, selected_symbol, rec)
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical indicators summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Current Indicators")
                    latest = df.iloc[-1]
                    
                    indicators_data = {
                        'Indicator': ['RSI', 'MACD', 'ADX', 'CMF', 'ATR'],
                        'Value': [
                            f"{latest.get('rsi', 0):.1f}",
                            f"{latest.get('macd', 0):.6f}",
                            f"{latest.get('adx', 0):.1f}",
                            f"{latest.get('cmf', 0):.3f}",
                            f"{latest.get('atr', 0):.6f}"
                        ],
                        'Signal': [
                            "Overbought" if latest.get('rsi', 50) > 70 else "Oversold" if latest.get('rsi', 50) < 30 else "Neutral",
                            "Bullish" if latest.get('macd', 0) > latest.get('macd_signal', 0) else "Bearish",
                            "Strong Trend" if latest.get('adx', 0) > 25 else "Weak Trend",
                            "Buying Pressure" if latest.get('cmf', 0) > 0.1 else "Selling Pressure" if latest.get('cmf', 0) < -0.1 else "Neutral",
                            "High Volatility" if latest.get('atr', 0)/latest.get('close', 1) > 0.02 else "Low Volatility"
                        ]
                    }
                    
                    df_indicators = pd.DataFrame(indicators_data)
                    st.dataframe(df_indicators, use_container_width=True)
                
                with col2:
                    if rec:
                        st.markdown("### 🎯 AI Recommendation Details")
                        st.json({
                            "Symbol": rec.symbol,
                            "Signal": rec.signal,
                            "Confidence": f"{rec.confidence:.1%}",
                            "Entry Price": f"${rec.entry_price:.4f}",
                            "Take Profit 1": f"${rec.tp1:.4f}",
                            "Take Profit 2": f"${rec.tp2:.4f}",
                            "Stop Loss": f"${rec.stop_loss:.4f}",
                            "Risk:Reward": f"{rec.risk_reward_ratio:.2f}:1",
                            "Reasoning": rec.reasoning
                        })
    
    with tab4:
        st.markdown("## 🔥 Top Volatile Pairs")
        
        # Sort by volatility
        sorted_pairs = sorted(market_data, key=lambda x: x['volatility_score'], reverse=True)
        
        for i, pair in enumerate(sorted_pairs[:10], 1):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                change_color = "🟢" if pair['change_24h'] > 0 else "🔴"
                st.markdown(f"""
                ### #{i} {change_color} {pair['symbol']}
                **Price:** ${pair['price']:.4f}  
                **24h:** {pair['change_24h']:.2f}%  
                **Vol:** {pair['volatility_score']:.1f}%
                """)
            
            with col2:
                # Mini chart
                if pair['symbol'] in detailed_data:
                    df_mini = detailed_data[pair['symbol']].tail(24)  # Last 24 hours
                    fig_mini = go.Figure()
                    fig_mini.add_trace(go.Scatter(
                        x=df_mini.index,
                        y=df_mini['close'],
                        mode='lines',
                        line=dict(color='#00ff88' if pair['change_24h'] > 0 else '#ff4444', width=2),
                        fill='tozeroy',
                        fillcolor=f"rgba({'0,255,136' if pair['change_24h'] > 0 else '255,68,68'},0.1)"
                    ))
                    fig_mini.update_layout(
                        height=150,
                        margin=dict(l=0, r=0, t=0, b=0),
                        showlegend=False,
                        xaxis=dict(showgrid=False, showticklabels=False),
                        yaxis=dict(showgrid=False, showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_mini, use_container_width=True)
            
            st.markdown("---")

# Risk Warning
st.markdown("---")
st.warning("""
⚠️ **Risk Warning**: Cryptocurrency trading involves substantial risk and may result in significant losses. 
This AI system provides analysis and recommendations for educational purposes only. Always conduct your own research 
and consider your financial situation before making trading decisions. Past performance does not guarantee future results.
""")

# Footer
st.markdown("""
---
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>🤖 RUAS AI Crypto Trading Dashboard | Built with Streamlit | Real-time Binance Data</p>
    <p>⭐ Enhanced with AI Analysis & Telegram Notifications</p>
</div>
""", unsafe_allow_html=True)

# Run the main function
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        logger.error(f"Main application error: {e}")
