#!/usr/bin/env python3
"""
Advanced Binance Futures AI Market Analyzer Bot
Comprehensive analysis with SMC, ICT, CMF, Pivot Points, Fibonacci
Telegram notifications integrated
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import logging
import warnings
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
import math

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_market_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketSentiment(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    VOLATILE = "VOLATILE"

class TrendStrength(Enum):
    VERY_STRONG = "VERY_STRONG"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"

class ICTConcept(Enum):
    BREAK_OF_STRUCTURE = "BOS"
    CHANGE_OF_CHARACTER = "CHOCH"
    ORDER_BLOCK = "OB"
    FAIR_VALUE_GAP = "FVG"
    INSTITUTIONAL_CANDLE = "INST_CANDLE"

@dataclass
class MarketSignal:
    symbol: str
    signal_type: str
    strength: float
    confidence: float
    timestamp: datetime
    price: float
    target: Optional[float] = None
    stop_loss: Optional[float] = None
    method: str = "AI_ANALYSIS"

@dataclass
class SMCStructure:
    """Smart Money Concepts structure"""
    level: float
    structure_type: str  # HH, HL, LH, LL
    timestamp: datetime
    broken: bool = False

@dataclass
class ICTSetup:
    """ICT trading setup"""
    concept: ICTConcept
    entry_zone: Tuple[float, float]
    target_zone: Tuple[float, float]
    stop_loss: float
    confidence: float

@dataclass
class FibonacciLevel:
    """Fibonacci retracement/extension level"""
    level: float
    percentage: float
    level_type: str  # "retracement" or "extension"

class TelegramNotifier:
    """Telegram notification service"""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.session = None
        
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not found in environment variables")
    
    async def __aenter__(self):
        if self.bot_token and self.chat_id:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def send_message(self, message: str, parse_mode: str = "HTML"):
        """Send message to Telegram"""
        if not self.session or not self.bot_token:
            logger.info(f"Telegram notification: {message}")
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info("Telegram notification sent successfully")
                    return True
                else:
                    logger.error(f"Failed to send Telegram notification: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            return False

class AdvancedTechnicalAnalyzer:
    """Advanced technical analysis without TA-Lib dependency"""
    
    @staticmethod
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        return pd.Series(data).rolling(window=period).mean().values
    
    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        return pd.Series(data).ewm(span=period).mean().values
    
    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
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
        """MACD Indicator"""
        ema_fast = AdvancedTechnicalAnalyzer.ema(data, fast)
        ema_slow = AdvancedTechnicalAnalyzer.ema(data, slow)
        macd = ema_fast - ema_slow
        macd_signal = AdvancedTechnicalAnalyzer.ema(macd, signal)
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands"""
        sma = AdvancedTechnicalAnalyzer.sma(data, period)
        std = pd.Series(data).rolling(window=period).std().values
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return pd.Series(tr).rolling(window=period).mean().values
    
    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator"""
        lowest_low = pd.Series(low).rolling(window=k_period).min()
        highest_high = pd.Series(high).rolling(window=k_period).max()
        k_percent = 100 * ((pd.Series(close) - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent.values, d_percent.values
    
    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average Directional Index"""
        plus_dm = pd.Series(high).diff()
        minus_dm = pd.Series(low).diff() * -1
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = AdvancedTechnicalAnalyzer.atr(high, low, close, 1)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / pd.Series(tr).rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / pd.Series(tr).rolling(window=period).mean())
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx.values
    
    @staticmethod
    def cmf(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 20) -> np.ndarray:
        """Chaikin Money Flow"""
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_multiplier = np.nan_to_num(mf_multiplier)
        mf_volume = mf_multiplier * volume
        
        cmf = pd.Series(mf_volume).rolling(window=period).sum() / pd.Series(volume).rolling(window=period).sum()
        return cmf.values

class SMCAnalyzer:
    """Smart Money Concepts Analyzer"""
    
    @staticmethod
    def identify_market_structure(df: pd.DataFrame) -> List[SMCStructure]:
        """Identify market structure points (HH, HL, LH, LL)"""
        structures = []
        
        try:
            # Find local highs and lows
            highs = df['high'].rolling(window=5, center=True).max()
            lows = df['low'].rolling(window=5, center=True).min()
            
            high_points = []
            low_points = []
            
            for i in range(len(df)):
                if df['high'].iloc[i] == highs.iloc[i] and not pd.isna(highs.iloc[i]):
                    high_points.append((i, df['high'].iloc[i], df.index[i]))
                if df['low'].iloc[i] == lows.iloc[i] and not pd.isna(lows.iloc[i]):
                    low_points.append((i, df['low'].iloc[i], df.index[i]))
            
            # Classify structure points
            for i in range(1, len(high_points)):
                current_high = high_points[i][1]
                prev_high = high_points[i-1][1]
                
                structure_type = "HH" if current_high > prev_high else "LH"
                structure = SMCStructure(
                    level=current_high,
                    structure_type=structure_type,
                    timestamp=high_points[i][2]
                )
                structures.append(structure)
            
            for i in range(1, len(low_points)):
                current_low = low_points[i][1]
                prev_low = low_points[i-1][1]
                
                structure_type = "HL" if current_low > prev_low else "LL"
                structure = SMCStructure(
                    level=current_low,
                    structure_type=structure_type,
                    timestamp=low_points[i][2]
                )
                structures.append(structure)
            
            return sorted(structures, key=lambda x: x.timestamp)[-10:]  # Return last 10 structures
            
        except Exception as e:
            logger.error(f"Error identifying market structure: {e}")
            return []
    
    @staticmethod
    def identify_order_blocks(df: pd.DataFrame) -> List[Dict]:
        """Identify institutional order blocks"""
        order_blocks = []
        
        try:
            for i in range(2, len(df) - 2):
                current = df.iloc[i]
                prev = df.iloc[i-1]
                next_candle = df.iloc[i+1]
                
                # Bullish order block: strong move up after consolidation
                if (current['close'] > current['open'] and 
                    (current['close'] - current['open']) / current['open'] > 0.02 and
                    current['volume'] > df['volume'].rolling(20).mean().iloc[i] * 1.5):
                    
                    order_blocks.append({
                        'type': 'bullish_ob',
                        'high': current['high'],
                        'low': current['low'],
                        'timestamp': df.index[i],
                        'strength': (current['close'] - current['open']) / current['open']
                    })
                
                # Bearish order block: strong move down after consolidation
                elif (current['close'] < current['open'] and
                      (current['open'] - current['close']) / current['open'] > 0.02 and
                      current['volume'] > df['volume'].rolling(20).mean().iloc[i] * 1.5):
                    
                    order_blocks.append({
                        'type': 'bearish_ob',
                        'high': current['high'],
                        'low': current['low'],
                        'timestamp': df.index[i],
                        'strength': (current['open'] - current['close']) / current['open']
                    })
            
            return order_blocks[-5:]  # Return last 5 order blocks
            
        except Exception as e:
            logger.error(f"Error identifying order blocks: {e}")
            return []
    
    @staticmethod
    def identify_fair_value_gaps(df: pd.DataFrame) -> List[Dict]:
        """Identify Fair Value Gaps (FVGs)"""
        fvgs = []
        
        try:
            for i in range(1, len(df) - 1):
                prev_candle = df.iloc[i-1]
                current = df.iloc[i]
                next_candle = df.iloc[i+1]
                
                # Bullish FVG: gap between previous low and next high
                if (prev_candle['low'] > next_candle['high'] and
                    current['close'] > current['open']):
                    
                    fvgs.append({
                        'type': 'bullish_fvg',
                        'upper': prev_candle['low'],
                        'lower': next_candle['high'],
                        'timestamp': df.index[i],
                        'filled': False
                    })
                
                # Bearish FVG: gap between previous high and next low
                elif (prev_candle['high'] < next_candle['low'] and
                      current['close'] < current['open']):
                    
                    fvgs.append({
                        'type': 'bearish_fvg',
                        'upper': next_candle['low'],
                        'lower': prev_candle['high'],
                        'timestamp': df.index[i],
                        'filled': False
                    })
            
            return fvgs[-3:]  # Return last 3 FVGs
            
        except Exception as e:
            logger.error(f"Error identifying FVGs: {e}")
            return []

class ICTAnalyzer:
    """Inner Circle Trader concepts analyzer"""
    
    @staticmethod
    def identify_break_of_structure(df: pd.DataFrame, structures: List[SMCStructure]) -> List[Dict]:
        """Identify Break of Structure (BOS)"""
        bos_signals = []
        
        try:
            current_price = df['close'].iloc[-1]
            
            for structure in structures[-5:]:  # Check recent structures
                if structure.structure_type in ['HH', 'LH'] and current_price > structure.level:
                    bos_signals.append({
                        'type': 'bullish_bos',
                        'level_broken': structure.level,
                        'structure_type': structure.structure_type,
                        'confirmation_price': current_price,
                        'timestamp': datetime.now()
                    })
                elif structure.structure_type in ['LL', 'HL'] and current_price < structure.level:
                    bos_signals.append({
                        'type': 'bearish_bos',
                        'level_broken': structure.level,
                        'structure_type': structure.structure_type,
                        'confirmation_price': current_price,
                        'timestamp': datetime.now()
                    })
            
            return bos_signals
            
        except Exception as e:
            logger.error(f"Error identifying BOS: {e}")
            return []
    
    @staticmethod
    def identify_change_of_character(df: pd.DataFrame) -> List[Dict]:
        """Identify Change of Character (CHOCH)"""
        choch_signals = []
        
        try:
            # Look for momentum divergences and trend changes
            recent_data = df.tail(20)
            
            # Calculate momentum using price changes
            momentum = recent_data['close'].pct_change(5)
            
            # Identify potential CHOCH points
            for i in range(5, len(recent_data) - 1):
                if (momentum.iloc[i] < -0.02 and momentum.iloc[i-1] > 0.01):  # Bullish to bearish
                    choch_signals.append({
                        'type': 'bearish_choch',
                        'price': recent_data['close'].iloc[i],
                        'timestamp': recent_data.index[i],
                        'momentum_change': momentum.iloc[i] - momentum.iloc[i-1]
                    })
                elif (momentum.iloc[i] > 0.02 and momentum.iloc[i-1] < -0.01):  # Bearish to bullish
                    choch_signals.append({
                        'type': 'bullish_choch',
                        'price': recent_data['close'].iloc[i],
                        'timestamp': recent_data.index[i],
                        'momentum_change': momentum.iloc[i] - momentum.iloc[i-1]
                    })
            
            return choch_signals
            
        except Exception as e:
            logger.error(f"Error identifying CHOCH: {e}")
            return []

class PivotPointAnalyzer:
    """Pivot Point analysis"""
    
    @staticmethod
    def calculate_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate traditional pivot points"""
        try:
            pivot = (high + low + close) / 3
            
            # Support and Resistance levels
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'pivot': pivot,
                'r1': r1, 'r2': r2, 'r3': r3,
                's1': s1, 's2': s2, 's3': s3
            }
            
        except Exception as e:
            logger.error(f"Error calculating pivot points: {e}")
            return {}
    
    @staticmethod
    def calculate_fibonacci_levels(high: float, low: float, trend: str = 'uptrend') -> List[FibonacciLevel]:
        """Calculate Fibonacci retracement levels"""
        try:
            levels = []
            fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
            
            range_val = high - low
            
            for ratio in fib_ratios:
                if trend == 'uptrend':
                    level = high - (range_val * ratio)
                else:
                    level = low + (range_val * ratio)
                
                levels.append(FibonacciLevel(
                    level=level,
                    percentage=ratio * 100,
                    level_type='retracement'
                ))
            
            # Extension levels
            extension_ratios = [1.272, 1.414, 1.618, 2.0]
            for ratio in extension_ratios:
                if trend == 'uptrend':
                    level = high + (range_val * (ratio - 1))
                else:
                    level = low - (range_val * (ratio - 1))
                
                levels.append(FibonacciLevel(
                    level=level,
                    percentage=ratio * 100,
                    level_type='extension'
                ))
            
            return levels
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            return []

class BinanceFuturesAPI:
    """Binance Futures API client for real-time data"""
    
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
        """Get kline/candlestick data"""
        url = f"{self.BASE_URL}/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def get_24hr_ticker(self, symbol: str = None) -> Dict:
        """Get 24hr ticker price change statistics"""
        url = f"{self.BASE_URL}/fapi/v1/ticker/24hr"
        params = {"symbol": symbol} if symbol else {}
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def get_funding_rate(self, symbol: str = None) -> Dict:
        """Get current funding rate"""
        url = f"{self.BASE_URL}/fapi/v1/premiumIndex"
        params = {"symbol": symbol} if symbol else {}
        async with self.session.get(url, params=params) as response:
            return await response.json()

class AdvancedAIMarketAnalyzer:
    """Advanced AI-powered market analyzer with SMC, ICT, CMF concepts"""
    
    def __init__(self):
        self.tech_analyzer = AdvancedTechnicalAnalyzer()
        self.smc_analyzer = SMCAnalyzer()
        self.ict_analyzer = ICTAnalyzer()
        self.pivot_analyzer = PivotPointAnalyzer()
        self.api = None
        self.notifier = None
    
    async def initialize(self):
        """Initialize API connection and notifier"""
        self.api = BinanceFuturesAPI()
        await self.api.__aenter__()
        
        self.notifier = TelegramNotifier()
        await self.notifier.__aenter__()
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.api:
            await self.api.__aexit__(None, None, None)
        if self.notifier:
            await self.notifier.__aexit__(None, None, None)
    
    def calculate_comprehensive_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators without TA-Lib"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            # Moving Averages
            df['sma_20'] = self.tech_analyzer.sma(close, 20)
            df['sma_50'] = self.tech_analyzer.sma(close, 50)
            df['ema_12'] = self.tech_analyzer.ema(close, 12)
            df['ema_26'] = self.tech_analyzer.ema(close, 26)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = self.tech_analyzer.macd(close)
            
            # RSI
            df['rsi'] = self.tech_analyzer.rsi(close, 14)
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.tech_analyzer.bollinger_bands(close)
            
            # Stochastic
            df['stoch_k'], df['stoch_d'] = self.tech_analyzer.stochastic(high, low, close)
            
            # ADX
            df['adx'] = self.tech_analyzer.adx(high, low, close)
            
            # ATR
            df['atr'] = self.tech_analyzer.atr(high, low, close)
            
            # CMF - Chaikin Money Flow
            df['cmf'] = self.tech_analyzer.cmf(high, low, close, volume)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def ai_sentiment_analysis(self, df: pd.DataFrame, smc_data: Dict, ict_data: Dict) -> MarketSentiment:
        """Advanced AI sentiment analysis incorporating SMC and ICT"""
        try:
            latest = df.iloc[-1]
            
            # Technical sentiment score
            tech_score = 0
            
            # RSI analysis
            if latest['rsi'] > 70:
                tech_score -= 2
            elif latest['rsi'] < 30:
                tech_score += 2
            elif 40 < latest['rsi'] < 60:
                tech_score += 1
            
            # MACD analysis
            if latest['macd'] > latest['macd_signal']:
                tech_score += 1
            else:
                tech_score -= 1
            
            # Moving average analysis
            if latest['close'] > latest['sma_20'] > latest['sma_50']:
                tech_score += 2
            elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                tech_score -= 2
            
            # CMF analysis
            if latest['cmf'] > 0.2:
                tech_score += 1
            elif latest['cmf'] < -0.2:
                tech_score -= 1
            
            # SMC analysis
            smc_score = 0
            order_blocks = smc_data.get('order_blocks', [])
            for ob in order_blocks:
                if ob['type'] == 'bullish_ob':
                    smc_score += 1
                else:
                    smc_score -= 1
            
            # ICT analysis
            ict_score = 0
            bos_signals = ict_data.get('bos_signals', [])
            for bos in bos_signals:
                if bos['type'] == 'bullish_bos':
                    ict_score += 2
                else:
                    ict_score -= 2
            
            # Combined AI score
            total_score = tech_score + smc_score + ict_score
            
            # Volatility check
            volatility = latest['atr'] / latest['close'] * 100
            if volatility > 5:
                return MarketSentiment.VOLATILE
            
            # Sentiment classification
            if total_score >= 3:
                return MarketSentiment.BULLISH
            elif total_score <= -3:
                return MarketSentiment.BEARISH
            else:
                return MarketSentiment.NEUTRAL
                
        except Exception as e:
            logger.error(f"Error in AI sentiment analysis: {e}")
            return MarketSentiment.NEUTRAL
    
    def generate_advanced_signals(self, df: pd.DataFrame, symbol: str, smc_data: Dict, ict_data: Dict, pivot_data: Dict) -> List[MarketSignal]:
        """Generate advanced trading signals"""
        signals = []
        
        try:
            latest = df.iloc[-1]
            current_price = latest['close']
            
            # SMC Order Block Signal
            order_blocks = smc_data.get('order_blocks', [])
            for ob in order_blocks:
                if (ob['type'] == 'bullish_ob' and 
                    ob['low'] <= current_price <= ob['high'] and
                    latest['close'] > latest['open']):
                    
                    signal = MarketSignal(
                        symbol=symbol,
                        signal_type="SMC_BULLISH_ORDER_BLOCK",
                        strength=ob['strength'],
                        confidence=0.8,
                        timestamp=datetime.now(),
                        price=current_price,
                        target=current_price * 1.025,
                        stop_loss=ob['low'],
                        method="SMC_ANALYSIS"
                    )
                    signals.append(signal)
            
            # ICT Break of Structure Signal
            bos_signals = ict_data.get('bos_signals', [])
            for bos in bos_signals:
                if bos['type'] == 'bullish_bos':
                    signal = MarketSignal(
                        symbol=symbol,
                        signal_type="ICT_BULLISH_BOS",
                        strength=0.8,
                        confidence=0.85,
                        timestamp=datetime.now(),
                        price=current_price,
                        target=current_price * 1.03,
                        stop_loss=bos['level_broken'],
                        method="ICT_ANALYSIS"
                    )
                    signals.append(signal)
            
            # Fibonacci + CMF Signal
            if latest['cmf'] > 0.1 and latest['rsi'] > 50:
                fib_levels = pivot_data.get('fibonacci', [])
                for fib in fib_levels:
                    if (fib.level_type == 'retracement' and 
                        fib.percentage == 61.8 and
                        abs(current_price - fib.level) / current_price < 0.005):
                        
                        signal = MarketSignal(
                            symbol=symbol,
                            signal_type="FIBONACCI_CMF_CONFLUENCE",
                            strength=0.75,
                            confidence=0.7,
                            timestamp=datetime.now(),
                            price=current_price,
                            target=current_price * 1.02,
                            stop_loss=current_price * 0.985,
                            method="FIBONACCI_CMF"
                        )
                        signals.append(signal)
                        break
            
            # Pivot Point Breakout Signal
            pivot_points = pivot_data.get('pivot_points', {})
            if pivot_points:
                if current_price > pivot_points.get('r1', 0) and latest['volume'] > df['volume'].rolling(20).mean().iloc[-1]:
                    signal = MarketSignal(
                        symbol=symbol,
                        signal_type="PIVOT_RESISTANCE_BREAKOUT",
                        strength=0.7,
                        confidence=0.75,
                        timestamp=datetime.now(),
                        price=current_price,
                        target=pivot_points.get('r2', current_price * 1.02),
                        stop_loss=pivot_points.get('pivot', current_price * 0.99),
                        method="PIVOT_ANALYSIS"
                    )
                    signals.append(signal)
                elif current_price < pivot_points.get('s1', 0) and latest['volume'] > df['volume'].rolling(20).mean().iloc[-1]:
                    signal = MarketSignal(
                        symbol=symbol,
                        signal_type="PIVOT_SUPPORT_BREAKDOWN",
                        strength=0.7,
                        confidence=0.75,
                        timestamp=datetime.now(),
                        price=current_price,
                        target=pivot_points.get('s2', current_price * 0.98),
                        stop_loss=pivot_points.get('pivot', current_price * 1.01),
                        method="PIVOT_ANALYSIS"
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating advanced signals: {e}")
            return []
    
    async def comprehensive_symbol_analysis(self, symbol: str) -> Dict:
        """Comprehensive analysis of a symbol with all advanced concepts"""
        try:
            # Get market data
            klines = await self.api.get_klines(symbol, "1h", 200)
            ticker_data = await self.api.get_24hr_ticker(symbol)
            funding_data = await self.api.get_funding_rate(symbol)
            
            # Process klines data
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate technical indicators
            df = self.calculate_comprehensive_indicators(df)
            
            # SMC Analysis
            structures = self.smc_analyzer.identify_market_structure(df)
            order_blocks = self.smc_analyzer.identify_order_blocks(df)
            fvgs = self.smc_analyzer.identify_fair_value_gaps(df)
            
            smc_data = {
                'structures': structures,
                'order_blocks': order_blocks,
                'fair_value_gaps': fvgs
            }
            
            # ICT Analysis
            bos_signals = self.ict_analyzer.identify_break_of_structure(df, structures)
            choch_signals = self.ict_analyzer.identify_change_of_character(df)
            
            ict_data = {
                'bos_signals': bos_signals,
                'choch_signals': choch_signals
            }
            
            # Pivot Points and Fibonacci
            recent_data = df.tail(1).iloc[0]
            pivot_points = self.pivot_analyzer.calculate_pivot_points(
                recent_data['high'], recent_data['low'], recent_data['close']
            )
            
            # Determine trend for Fibonacci
            trend = 'uptrend' if recent_data['close'] > df['sma_50'].iloc[-1] else 'downtrend'
            high_period = df['high'].rolling(20).max().iloc[-1]
            low_period = df['low'].rolling(20).min().iloc[-1]
            fibonacci_levels = self.pivot_analyzer.calculate_fibonacci_levels(high_period, low_period, trend)
            
            pivot_data = {
                'pivot_points': pivot_points,
                'fibonacci': fibonacci_levels
            }
            
            # AI Sentiment Analysis
            sentiment = self.ai_sentiment_analysis(df, smc_data, ict_data)
            
            # Generate advanced signals
            signals = self.generate_advanced_signals(df, symbol, smc_data, ict_data, pivot_data)
            
            # Calculate risk metrics
            latest = df.iloc[-1]
            volatility = latest['atr'] / latest['close'] * 100
            momentum_20 = (latest['close'] - df['close'].iloc[-20]) / df['close'].iloc[-20] * 100
            
            # Calculate AI confidence score
            confidence_factors = []
            
            # Volume confirmation
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            if latest['volume'] > avg_volume:
                confidence_factors.append(0.1)
            
            # Multiple timeframe alignment (simulated)
            if latest['sma_20'] > latest['sma_50']:
                confidence_factors.append(0.1)
            
            # CMF confirmation
            if abs(latest['cmf']) > 0.1:
                confidence_factors.append(0.1)
            
            ai_confidence = sum(confidence_factors) + 0.5  # Base confidence
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price_data': {
                    'current_price': latest['close'],
                    'change_24h': float(ticker_data.get('priceChangePercent', 0)),
                    'volume_24h': float(ticker_data.get('volume', 0)),
                    'high_24h': float(ticker_data.get('highPrice', 0)),
                    'low_24h': float(ticker_data.get('lowPrice', 0))
                },
                'technical_indicators': {
                    'rsi': latest['rsi'],
                    'macd': latest['macd'],
                    'macd_signal': latest['macd_signal'],
                    'cmf': latest['cmf'],
                    'adx': latest['adx'],
                    'bb_position': (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
                },
                'smc_analysis': smc_data,
                'ict_analysis': ict_data,
                'pivot_data': pivot_data,
                'ai_analysis': {
                    'sentiment': sentiment,
                    'confidence': ai_confidence,
                    'volatility': volatility,
                    'momentum_20': momentum_20,
                    'trend_strength': 'STRONG' if latest['adx'] > 25 else 'WEAK'
                },
                'signals': signals,
                'funding_rate': float(funding_data.get('lastFundingRate', 0)) if isinstance(funding_data, dict) else 0
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            return None

class AdvancedMarketAnalyzerBot:
    """Main bot class with Telegram notifications"""
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['ASTERUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        self.analyzer = AdvancedAIMarketAnalyzer()
        self.running = False
        self.analysis_results = {}
        self.last_notification_time = {}
    
    async def initialize(self):
        """Initialize the bot"""
        await self.analyzer.initialize()
        logger.info("Advanced Market Analyzer Bot initialized successfully")
        
        # Send initialization message
        await self.send_telegram_notification("🤖 Advanced Market Analyzer Bot Started!\n\nMonitoring symbols: " + ", ".join(self.symbols))
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.analyzer.cleanup()
        logger.info("Bot cleanup completed")
    
    async def send_telegram_notification(self, message: str):
        """Send Telegram notification"""
        try:
            if self.analyzer.notifier:
                await self.analyzer.notifier.send_message(message)
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
    
    def format_analysis_message(self, analysis: Dict) -> str:
        """Format analysis data for Telegram message"""
        try:
            symbol = analysis['symbol']
            price_data = analysis['price_data']
            ai_analysis = analysis['ai_analysis']
            signals = analysis['signals']
            
            message = f"📊 <b>{symbol} Analysis Report</b>\n\n"
            message += f"💰 Price: ${price_data['current_price']:.4f}\n"
            message += f"📈 24h Change: {price_data['change_24h']:.2f}%\n"
            message += f"🎯 Sentiment: <b>{ai_analysis['sentiment'].value}</b>\n"
            message += f"💪 Trend: {ai_analysis['trend_strength']}\n"
            message += f"📊 AI Confidence: {ai_analysis['confidence']:.2f}\n"
            message += f"⚡ Volatility: {ai_analysis['volatility']:.2f}%\n"
            message += f"🚀 Momentum (20h): {ai_analysis['momentum_20']:.2f}%\n\n"
            
            # Technical indicators
            tech = analysis['technical_indicators']
            message += f"📋 <b>Technical Indicators:</b>\n"
            message += f"RSI: {tech['rsi']:.1f}\n"
            message += f"CMF: {tech['cmf']:.3f}\n"
            message += f"ADX: {tech['adx']:.1f}\n\n"
            
            # SMC Analysis
            smc = analysis['smc_analysis']
            if smc['order_blocks']:
                message += f"🏦 <b>SMC Order Blocks:</b>\n"
                for ob in smc['order_blocks'][:2]:
                    message += f"• {ob['type'].replace('_', ' ').title()}: {ob['strength']:.3f}\n"
                message += "\n"
            
            # Signals
            if signals:
                message += f"🔔 <b>Trading Signals ({len(signals)}):</b>\n"
                for signal in signals[:3]:  # Limit to 3 signals
                    message += f"• <b>{signal.signal_type.replace('_', ' ')}</b>\n"
                    message += f"  Strength: {signal.strength:.2f} | Confidence: {signal.confidence:.2f}\n"
                    if signal.target:
                        message += f"  🎯 Target: ${signal.target:.4f}\n"
                    if signal.stop_loss:
                        message += f"  🛑 Stop Loss: ${signal.stop_loss:.4f}\n"
                    message += f"  Method: {signal.method}\n\n"
            
            # Pivot Points
            pivots = analysis['pivot_data']['pivot_points']
            if pivots:
                message += f"📍 <b>Pivot Points:</b>\n"
                message += f"Pivot: ${pivots['pivot']:.4f}\n"
                message += f"R1: ${pivots['r1']:.4f} | S1: ${pivots['s1']:.4f}\n\n"
            
            message += f"⏰ <i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting analysis message: {e}")
            return f"Error formatting analysis for {analysis.get('symbol', 'Unknown')}"
    
    def should_send_notification(self, symbol: str, signals: List[MarketSignal]) -> bool:
        """Determine if notification should be sent"""
        try:
            # Send notification if there are high-confidence signals
            high_confidence_signals = [s for s in signals if s.confidence > 0.7 and s.strength > 0.6]
            
            if not high_confidence_signals:
                return False
            
            # Rate limiting: don't send notifications too frequently
            last_time = self.last_notification_time.get(symbol, datetime.min)
            if (datetime.now() - last_time).seconds < 1800:  # 30 minutes
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking notification criteria: {e}")
            return False
    
    def display_analysis_console(self, analysis: Dict):
        """Display analysis results in console"""
        try:
            symbol = analysis['symbol']
            print(f"\n{'='*80}")
            print(f"🚀 ADVANCED AI MARKET ANALYSIS: {symbol}")
            print(f"{'='*80}")
            
            # Price information
            price_data = analysis['price_data']
            print(f"💰 Current Price: ${price_data['current_price']:.4f}")
            print(f"📈 24h Change: {price_data['change_24h']:.2f}%")
            print(f"📊 24h Volume: {price_data['volume_24h']:,.0f}")
            
            # AI Analysis
            ai_analysis = analysis['ai_analysis']
            print(f"\n🤖 AI ANALYSIS:")
            print(f"   Sentiment: {ai_analysis['sentiment'].value}")
            print(f"   Confidence: {ai_analysis['confidence']:.2f}")
            print(f"   Volatility: {ai_analysis['volatility']:.2f}%")
            print(f"   Momentum (20h): {ai_analysis['momentum_20']:.2f}%")
            print(f"   Trend Strength: {ai_analysis['trend_strength']}")
            
            # Technical Indicators
            tech = analysis['technical_indicators']
            print(f"\n📊 TECHNICAL INDICATORS:")
            print(f"   RSI: {tech['rsi']:.1f}")
            print(f"   MACD: {tech['macd']:.4f}")
            print(f"   CMF: {tech['cmf']:.3f}")
            print(f"   ADX: {tech['adx']:.1f}")
            print(f"   BB Position: {tech['bb_position']:.2f}")
            
            # SMC Analysis
            smc = analysis['smc_analysis']
            print(f"\n🏦 SMART MONEY CONCEPTS:")
            print(f"   Market Structures: {len(smc['structures'])}")
            print(f"   Order Blocks: {len(smc['order_blocks'])}")
            print(f"   Fair Value Gaps: {len(smc['fair_value_gaps'])}")
            
            # ICT Analysis
            ict = analysis['ict_analysis']
            print(f"\n⚡ ICT CONCEPTS:")
            print(f"   Break of Structure: {len(ict['bos_signals'])}")
            print(f"   Change of Character: {len(ict['choch_signals'])}")
            
            # Pivot Points
            pivots = analysis['pivot_data']['pivot_points']
            if pivots:
                print(f"\n📍 PIVOT POINTS:")
                print(f"   Pivot: ${pivots['pivot']:.4f}")
                print(f"   Resistance: R1=${pivots['r1']:.4f}, R2=${pivots['r2']:.4f}")
                print(f"   Support: S1=${pivots['s1']:.4f}, S2=${pivots['s2']:.4f}")
            
            # Fibonacci Levels
            fib_levels = analysis['pivot_data']['fibonacci']
            if fib_levels:
                print(f"\n🌀 FIBONACCI LEVELS:")
                key_levels = [f for f in fib_levels if f.percentage in [38.2, 50.0, 61.8]]
                for level in key_levels[:3]:
                    print(f"   {level.percentage}% {level.level_type}: ${level.level:.4f}")
            
            # Trading Signals
            signals = analysis['signals']
            if signals:
                print(f"\n🔔 TRADING SIGNALS ({len(signals)}):")
                for i, signal in enumerate(signals, 1):
                    print(f"   {i}. {signal.signal_type.replace('_', ' ')}")
                    print(f"      Method: {signal.method}")
                    print(f"      Strength: {signal.strength:.2f} | Confidence: {signal.confidence:.2f}")
                    if signal.target and signal.stop_loss:
                        print(f"      Target: ${signal.target:.4f} | Stop Loss: ${signal.stop_loss:.4f}")
            else:
                print(f"\n🔔 No active trading signals")
            
            # Funding Rate
            funding = analysis.get('funding_rate', 0)
            print(f"\n💸 Funding Rate: {funding*100:.4f}%")
            
            print(f"\n⏰ Analysis Time: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")
            
        except Exception as e:
            logger.error(f"Error displaying console analysis: {e}")
    
    async def analyze_all_symbols(self):
        """Analyze all symbols"""
        try:
            logger.info("Starting comprehensive market analysis...")
            
            for symbol in self.symbols:
                try:
                    analysis = await self.analyzer.comprehensive_symbol_analysis(symbol)
                    
                    if analysis:
                        self.analysis_results[symbol] = analysis
                        self.display_analysis_console(analysis)
                        
                        # Check for notification-worthy signals
                        if self.should_send_notification(symbol, analysis['signals']):
                            message = self.format_analysis_message(analysis)
                            await self.send_telegram_notification(message)
                            self.last_notification_time[symbol] = datetime.now()
                    
                    # Rate limiting
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
    
    def display_market_summary(self):
        """Display market summary"""
        if not self.analysis_results:
            return
        
        print(f"\n{'='*80}")
        print("📋 MARKET SUMMARY")
        print(f"{'='*80}")
        
        total_symbols = len(self.analysis_results)
        sentiment_counts = {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0, 'VOLATILE': 0}
        total_signals = 0
        high_confidence_signals = 0
        avg_confidence = 0
        
        for symbol, analysis in self.analysis_results.items():
            sentiment = analysis['ai_analysis']['sentiment'].value
            sentiment_counts[sentiment] += 1
            
            signals = analysis['signals']
            total_signals += len(signals)
            high_confidence_signals += len([s for s in signals if s.confidence > 0.7])
            avg_confidence += analysis['ai_analysis']['confidence']
        
        avg_confidence /= total_symbols
        
        print(f"📊 Analyzed Symbols: {total_symbols}")
        print(f"🎯 Market Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total_symbols) * 100
            print(f"   {sentiment}: {count} ({percentage:.1f}%)")
        
        print(f"🔔 Total Signals: {total_signals}")
        print(f"⭐ High Confidence Signals: {high_confidence_signals}")
        print(f"🤖 Average AI Confidence: {avg_confidence:.2f}")
        print(f"⏰ Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
    
    async def run_continuous(self, interval_minutes: int = 60):
        """Run continuous analysis"""
        self.running = True
        logger.info(f"Starting continuous analysis every {interval_minutes} minutes")
        
        try:
            while self.running:
                await self.analyze_all_symbols()
                self.display_market_summary()
                
                # Send summary notification every 4 hours
                current_time = datetime.now()
                if current_time.hour % 4 == 0 and current_time.minute < 5:
                    summary_message = f"📋 <b>Market Summary Update</b>\n\n"
                    summary_message += f"Analyzed {len(self.analysis_results)} symbols\n"
                    summary_message += f"Total active signals: {sum(len(a['signals']) for a in self.analysis_results.values())}\n"
                    summary_message += f"Time: {current_time.strftime('%H:%M:%S')}"
                    
                    await self.send_telegram_notification(summary_message)
                
                # Wait for next analysis cycle
                logger.info(f"Waiting {interval_minutes} minutes for next analysis cycle...")
                await asyncio.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("Stopping continuous analysis...")
            self.running = False
        except Exception as e:
            logger.error(f"Error in continuous analysis: {e}")
            self.running = False
    
    def stop(self):
        """Stop the bot"""
        self.running = False

async def main():
    """Main function"""
    
    print("""
    🚀 ADVANCED BINANCE FUTURES AI MARKET ANALYZER
    =============================================
    Features:
    ✅ Smart Money Concepts (SMC)
    ✅ Inner Circle Trader (ICT) Analysis
    ✅ Chaikin Money Flow (CMF)
    ✅ Pivot Points & Fibonacci
    ✅ Advanced AI Sentiment Analysis
    ✅ Telegram Notifications
    ✅ No TA-Lib dependency
    
    Setup Telegram Notifications:
    - Set TELEGRAM_BOT_TOKEN environment variable
    - Set TELEGRAM_CHAT_ID environment variable
    
    Starting analysis...
    """)
    
    # Symbols to analyze
    symbols = [
        'ASTERUSDT', 'INUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
        'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'AVAXUSDT', 'MATICUSDT'
    ]
    
    # Initialize bot
    bot = AdvancedMarketAnalyzerBot(symbols)
    
    try:
        await bot.initialize()
        
        # Run single analysis first
        print("🔍 Running initial market analysis...")
        await bot.analyze_all_symbols()
        bot.display_market_summary()
        
        # Ask for continuous mode
        print("\n💡 Single analysis completed.")
        print("🔄 For continuous monitoring, modify the script to call:")
        print("await bot.run_continuous(interval_minutes=60)")
        
        # Demo: Run a few more cycles
        print("\n🔄 Running 2 more analysis cycles for demo...")
        for i in range(2):
            print(f"\n--- Analysis Cycle {i+2}/3 ---")
            await asyncio.sleep(30)  # Wait 30 seconds
            await bot.analyze_all_symbols()
            bot.display_market_summary()
        
    except Exception as e:
        logger.error(f"Error running bot: {e}")
    finally:
        await bot.cleanup()

if __name__ == "__main__":
    print("📦 Required packages: aiohttp pandas numpy asyncio logging")
    print("🔑 Set environment variables: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
    print("🚀 Starting Advanced Market Analyzer Bot...\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot error: {e}")
