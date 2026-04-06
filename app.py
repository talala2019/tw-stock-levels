import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from FinMind.data import DataLoader
from datetime import datetime, timedelta
import re
import warnings

warnings.filterwarnings('ignore')

# [NEW] 輔助函式：判斷顏色與正負號 (台股習慣：漲紅跌綠)
def get_clr(val):
    return "#ff4b4b" if val > 0 else ("#00ad00" if val < 0 else "#333")

def get_sign(val):
    return "+" if val > 0 else ""

# ==========================================
# 核心邏輯區：保留原始邏輯內容
# ==========================================

def filter_smart_levels(df_levels, is_support=True):
    if df_levels.empty: return df_levels
    
    # 【關鍵修正 1】：距離絕對優先 (Distance 升序排第一)
    df_levels = df_levels.sort_values(['Distance', 'Date'], ascending=[True, False])
    
    selected = []
    for _, row in df_levels.iterrows():
        # 過濾掉距離超過 20% 的位置
        if abs(row['Pct']) > 20.0:
            continue
            
        is_duplicate = False
        for s in selected:
            # 保持 0.8% 的濾波門檻，避免重複
            if abs(row['Price'] - s['Price']) / s['Price'] < 0.008:
                is_duplicate = True
                break
        
        if not is_duplicate:
            selected.append(row)
        
        if len(selected) >= 5: break
            
    res_df = pd.DataFrame(selected)
    return res_df.sort_values('Price', ascending=not is_support)
    
@st.cache_data(ttl=3600)
def get_full_data(stock_id, days=380): 
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime('%Y-%m-%d')

    df = pd.DataFrame()
    found_ticker = f"{stock_id}.TW"
    
    try:
        df = yf.download(found_ticker, start=start_str, progress=False)
        if df.empty or len(df) < 5:
            found_ticker = f"{stock_id}.TWO"
            df = yf.download(found_ticker, start=start_str, progress=False)
    except Exception:
        found_ticker = f"{stock_id}.TWO"
        df = yf.download(found_ticker, start=start_str, progress=False)

    if df.empty: return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    dl = DataLoader()
    try:
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_str)
        if not df_inst.empty:
            df_inst['Date'] = pd.to_datetime(df_inst['date']).dt.date
            if 'buy_sell' not in df_inst.columns:
                df_inst['buy_sell'] = df_inst['buy'] - df_inst['sell']
            inst_sum = (df_inst.groupby('Date')['buy_sell'].sum() / 1000).reset_index()
            inst_sum.rename(columns={'buy_sell': 'Net_Buy'}, inplace=True)
            df = pd.merge(df, inst_sum, on='Date', how='left').fillna(0)
        else: 
            df['Net_Buy'] = 0
    except: 
        df['Net_Buy'] = 0
    
    return df

def analyze(df, cur_price):
    supports, resistances = [], []
    latest_date = str(df['Date'].iloc[-1])
    
    ma_list = ['MA5', 'MA10', 'MA20', 'MA60', 'MA120', 'MA240']
    for ma in ma_list:
        days = int(ma.replace('MA', ''))
        df[ma] = df['Close'].rolling(days).mean()
        val = round(df[ma].iloc[-1], 2)
        if not np.isnan(val):
            if val < cur_price:
                supports.append({'Date': latest_date, 'Price': val, 'Type': f'{ma}位置'})
            else:
                resistances.append({'Date': latest_date, 'Price': val, 'Type': f'{ma}位置'})
    
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    if len(df) >= 2:
        yesterday = df.iloc[-2]
        y_open, y_low, y_high, y_close, y_vol, y_vol_ma20 = yesterday['Open'], yesterday['Low'], yesterday['High'], yesterday['Close'], yesterday['Volume'], yesterday['Vol_MA20']
        y_date = str(yesterday['Date'])

        if y_vol > y_vol_ma20:
            if (y_low < cur_price) and (y_close > y_open): 
                supports.append({'Date': y_date, 'Price': y_low, 'Type': '前一交易日低點(量增紅K)'})
            if (y_high > cur_price) and (y_close < y_open): 
                resistances.append({'Date': y_date, 'Price': y_high, 'Type': '前一交易日高點(量增黑K)'})

    if 'Net_Buy' in df.columns:
        chip_recent_df = df.tail(60)
        std = chip_recent_df['Net_Buy'].std()
        if std > 0:
            big_b = chip_recent_df[(chip_recent_df['Net_Buy'] > 1.0 * std) & (chip_recent_df['Close'] > chip_recent_df['Open'])]
            for _, r in big_b.iterrows(): 
                supports.append({'Date': str(r['Date']), 'Price': r['Low'], 'Type': '法人大買紅K底部'})
            big_s = chip_recent_df[(chip_recent_df['Net_Buy'] < -1.0 * std) & (chip_recent_df['Close'] < chip_recent_df['Open'])]
            for _, r in big_s.iterrows(): 
                resistances.append({'Date': str(r['Date']), 'Price': r['High'], 'Type': '法人大賣黑K頂部'})

    recent_df = df.tail(20) 
    for _, r in recent_df.iterrows():
        if r['Volume'] > r['Vol_MA20'] * 1.5:
            d_str = str(r['Date'])
            if r['Close'] > r['Open']:
                supports.append({'Date': d_str, 'Price': r['Low'], 'Type': '帶量紅K底部'})
            elif r['Close'] < r['Open']:
                resistances.append({'Date': d_str, 'Price': r['High'], 'Type': '帶量黑K頂部'})

    n = 5
    max_idx = argrelextrema(df['High'].values, np.greater_equal, order=n)[0]
    min_idx = argrelextrema(df['Low'].values, np.less_equal, order=n)[0]
    for i in max_idx[-10:]: resistances.append({'Date': str(df['Date'].iloc[i]), 'Price': df['High'].iloc[i], 'Type': '波段相對高點'})
    for i in min_idx[-10:]: supports.append({'Date': str(df['Date'].iloc[i]), 'Price': df['Low'].iloc[i], 'Type': '波段相對低點'})

    step = 10 if cur_price < 100 else (50 if cur_price < 500 else 100)
    lower_round = (cur_price // step) * step
    upper_round = lower_round + step
    supports.append({'Date': latest_date, 'Price': float(lower_round), 'Type': '整數心理關卡'})
    resistances.append({'Date': latest_date, 'Price': float(upper_round), 'Type': '整數心理關卡'})

    df_s = pd.DataFrame(supports).drop_duplicates()
    if not df_s.empty:
        df_s = df_s[df_s['Price'] < cur_price].copy()
        df_s['Distance'] = cur_price - df_s['Price']
        df_s['Pct'] = ((df_s['Price'] - cur_price) / cur_price * 100).round(1)
    
    df_r = pd.DataFrame(resistances).drop_duplicates()
    if not df_r.empty:
        df_r = df_r[df_r['Price'] > cur_price].copy()
        df_r['Distance'] = df_r['Price'] - cur_price
        df_r['Pct'] = ((df_r['Price'] - cur_price) / cur_price * 100).round(1)

    final_r = filter_smart_levels(df_r, is_support=False)
    final_s = filter_smart_levels(df_s, is_support=True)

    return final_r, final_s

# ==========================================
# Streamlit UI 區
# ==========================================

def main():
    st.markdown("<h3 style='margin-bottom: 0px;'>📈 台股支撐壓力分析</h3>", unsafe_allow_html=True)
    
    # 建立輸入區
    fav_list = {
        "自定義輸入": "", "2301 光寶科": "2301", "2308 台達電": "2308", "2313 華通": "2313",
        "2317 鴻海": "2317", "2330 台積電": "2330", "2337 旺宏": "2337", "2449 京元電": "2449", 
        "2451 創見": "2451", "2454 聯發科": "2454", "2455 全新": "2455", "3017 奇鋐": "3017", 
        "3037 欣興": "3037", "3081 聯亞": "3081", "3105 穩懋": "3105", "3163 波若威": "3163", 
        "3231 緯創": "3231", "3260 威剛": "3260", "3293 鈊象": "3293", "3324 雙鴻": "3324",
        "3363 上詮": "3363", "3450 聯鈞": "3450", "3711 日月光": "3711", "4979 華星光": "4979",
        "5340 建榮": "5340", "5475 德宏": "5475", "6285 啟碁": "6285", "6442 光聖": "6442", 
        "6451 訊芯-KY": "6451", "6669 緯穎": "6669", "6770 力積電": "6770", "8021 尖點": "8021", "8271 宇瞻": "8271"
    }
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_label = st.selectbox("選擇標的", list(fav_list.keys()), label_visibility="collapsed")
        stock_id = st.text_input("輸入 4 位代碼", value="") if selected_label == "自定義輸入" else fav_list[selected_label]
    with col2:
        btn = st.button("執行分析", use_container_width=True)

    if btn or (selected_label != "自定義輸入" and stock_id != ""):
        try:
            with st.spinner('計算中...'):
                df = get_full_data(stock_id, days=380)
                if df.empty: return st.error("查無資料")
                
                # --- [周轉率抓取邏輯] ---
                import yfinance as yf
                ticker = yf.Ticker(f"{stock_id}.TW")
                info = ticker.info
                shares = info.get('sharesOutstanding') or info.get('floatShares') or 0
                if shares == 0:
                    try:
                        shares_series = ticker.get_shares_full(start="2025-01-01")
                        if not shares_series.empty: shares = shares_series.iloc[-1]
                    except: shares = 0
                
                cur_vol_shares = df['Volume'].iloc[-1]
                turnover_rate = (cur_vol_shares / shares * 100) if shares > 0 else 0
                
                # --- [數據計算區] ---
                cur_close = float(df['Close'].iloc[-1])
                prev_close = float(df['Close'].iloc[-2])
                y_open, y_high, y_low = df['Open'].iloc[-1], df['High'].iloc[-1], df['Low'].iloc[-1]
                
                pct_3d = ((cur_close - df['Close'].iloc[-4]) / df['Close'].iloc[-4] * 100) if len(df)>=4 else 0
                pct_10d = ((cur_close - df['Close'].iloc[-11]) / df['Close'].iloc[-11] * 100) if len(df)>=11 else 0
                pct_60d = ((cur_close - df['Close'].iloc[-61]) / df['Close'].iloc[-61] * 100) if len(df)>=61 else 0
                
                high_60, low_60 = df.tail(60)['High'].max(), df.tail(60)['Low'].min()
                
                # 均線與乖離計算
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA240'] = df['Close'].rolling(240).mean()
                bias_20 = ((cur_close - df['MA20'].iloc[-1]) / df['MA20'].iloc[-1] * 100) if not pd.isna(df['MA20'].iloc[-1]) else 0
                bias_240 = ((cur_close - df['MA240'].iloc[-1]) / df['MA240'].iloc[-1] * 100) if not pd.isna(df['MA240'].iloc[-1]) else 0
                
                # 月線斜率與量比
                df['Vol_MA20'] = df['Volume'].rolling(20).mean()
                ma20_slope_pct = ((df['MA20'].iloc[-1] - df['MA20'].iloc[-2]) / df['MA20'].iloc[-2] * 100) if len(df)>20 else 0
                vol_ratio = (df['Volume'].iloc[-1] / df['Vol_MA20'].iloc[-1]) if df['Vol_MA20'].iloc[-1] > 0 else 0
                
                r, s = analyze(df, cur_close)

                # --- [UI 顯示區] ---
                st.caption(f"📅 數據日期：{df['Date'].iloc[-1]}")
                
                diff = cur_close - prev_close
                pct = (diff / prev_close) * 100
                st.markdown(
                    f"""
                    <div style="font-size: 1.1rem; font-weight: bold; margin-top: -10px; margin-bottom: 5px;">
                        {stock_id} ｜ 收盤 : <span style="font-size: 1.2rem;">{cur_close:.1f}</span> 
                        <span style="color: {get_clr(diff)}; margin-left: 8px;">
                            {get_sign(diff)}{diff:.1f} ({get_sign(pct)}{pct:.1f}%)
                        </span>
                    </div>
                    """, unsafe_allow_html=True
                )
                
                total_range = y_high - y_low if y_high != y_low else 1
                body_top, body_bottom = max(y_open, cur_close), min(y_open, cur_close)
                up_shadow_p = ((y_high - body_top) / total_range) * 100
                body_p = ((body_top - body_bottom) / total_range) * 100
                low_shadow_p = ((body_bottom - y_low) / total_range) * 100
                
                k_color = "#ff4b4b" if cur_close >= y_open else "#00ad00"
                slope_clr = "#ff4b4b" if ma20_slope_pct > 0 else "#00ad00"
                vol_clr = "#1f77b4" if vol_ratio > 1.2 else ("#999" if vol_ratio < 0.8 else "#666")
                turnover_style = "color:#ff4b4b; font-weight:bold;" if turnover_rate > 5 else "color:#666;"

                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; background-color: #f8f9fa; padding: 10px; border-radius: 8px; margin-bottom: 0px;">
                        <div style="width: 25px; height: 95px; display: flex; flex-direction: column; align-items: center; margin-right: 15px;">
                            <div style="width: 2px; height: {up_shadow_p}%; background-color: #333;"></div>
                            <div style="width: 12px; height: {max(body_p, 5)}%; background-color: {k_color}; border-radius: 1px;"></div>
                            <div style="width: 2px; height: {low_shadow_p}%; background-color: #333;"></div>
                        </div>
                        <div style="flex-grow: 1; line-height: 1.4;">
                            <div style="font-size: 0.95rem; color: #444;">
                                <span style="color: {k_color}; font-weight:bold;">{ "紅K" if cur_close >= y_open else "黑K" }</span> 
                                開: {y_open:.1f}  高: {y_high:.1f} / 低: {y_low:.1f}
                            </div>
                            <div style="font-size: 0.82rem; color: #666;">
                                漲跌 3天: <span style="color:{get_clr(pct_3d)};">{pct_3d:+.1f}%</span> | 10天: <span style="color:{get_clr(pct_10d)};">{pct_10d:+.1f}%</span> | 60天: <span style="color:{get_clr(pct_60d)};">{pct_60d:+.1f}%</span>
                            </div>
                            <div style="font-size: 0.82rem; color: #444; border-top: 1px dashed #ddd; margin-top: 2px;">
                                60日 高: {high_60:.1f} / 低: {low_60:.1f}
                            </div>
                            <div style="font-size: 0.82rem; color: #444;">
                                乖離Ma20: <span style="color:{get_clr(bias_20)}; font-weight:bold;">{bias_20:+.1f}%</span> ｜ 乖離Ma240: <span style="color:{get_clr(bias_240)};">{bias_240:+.1f}%</span>
                            </div>
                            <div style="font-size: 0.82rem; color: #666; border-top: 1px solid #eee; margin-top: 2px; padding-top: 2px;">
                                月線斜率: <span style="color: {slope_clr}; font-weight:bold;">{ma20_slope_pct:+.2f}%</span> ｜ 量比: <span style="color: {vol_clr}; font-weight:bold;">{vol_ratio:.2f}x</span> ｜ 周轉率: <span style="{turnover_style}">{turnover_rate:.1f}%</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True
                )

                st.markdown("<div style='margin-bottom: 5px;'></div>", unsafe_allow_html=True)
                
                st.success("🟢 【上漲壓力區】")
                if not r.empty:
                    for i, (_, row) in enumerate(r.iterrows()):
                        header = f"P{i+1}： {row['Price']:.1f} 元 ➜ :red[(+{row['Pct']}%)] | {row['Type']}"
                        with st.expander(header): st.markdown(f"**日期：** `{row['Date']}`")
                
                st.error("🔴 【下跌支撐區】")
                if not s.empty:
                    for i, (_, row) in enumerate(s.iterrows()):
                        header = f"S{i+1}： {row['Price']:.1f} 元 ➜ :green[({row['Pct']}%)] | {row['Type']}"
                        with st.expander(header): st.markdown(f"**日期：** `{row['Date']}`")

        except Exception as e:
            st.error(f"分析錯誤: {e}")

if __name__ == "__main__":
    main()