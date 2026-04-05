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

# ==========================================
# 核心邏輯區：保留原始邏輯內容
# ==========================================

def filter_smart_levels(df_levels, is_support=True):
    if df_levels.empty: return df_levels
    df_levels = df_levels.sort_values(['Date', 'Distance'], ascending=[False, True])
    selected = []
    for _, row in df_levels.iterrows():
        is_duplicate = False
        for s in selected:
            if abs(row['Price'] - s['Price']) / s['Price'] < 0.008:
                is_duplicate = True
                break
        if not is_duplicate:
            selected.append(row)
        if len(selected) >= 5: break
    res_df = pd.DataFrame(selected)
    return res_df.sort_values('Price', ascending=not is_support)

@st.cache_data(ttl=3600)
def get_full_data(stock_id, days=120):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime('%Y-%m-%d')
    ticker_str = f"{stock_id}.TW"
    df = yf.download(ticker_str, start=start_str, progress=False)
    if df.empty:
        ticker_str = f"{stock_id}.TWO"
        df = yf.download(ticker_str, start=start_str, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    # 籌碼資料抓取
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
        else: df['Net_Buy'] = 0
    except: df['Net_Buy'] = 0
    return df

def analyze(df, cur_price):
    supports, resistances = [], []
    latest_date = str(df['Date'].iloc[-1])
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    for ma in ['MA5', 'MA10', 'MA20']:
        val = round(df[ma].iloc[-1], 2)
        if not np.isnan(val):
            if val < cur_price: supports.append({'Date': latest_date, 'Price': val, 'Type': f'{ma}位置'})
            else: resistances.append({'Date': latest_date, 'Price': val, 'Type': f'{ma}位置'})
    
    if len(df) >= 2:
        y_low, y_high = df['Low'].iloc[-2], df['High'].iloc[-2]
        y_date = str(df['Date'].iloc[-2])
        if y_low < cur_price: supports.append({'Date': y_date, 'Price': y_low, 'Type': '前一交易日低點'})
        if y_high > cur_price: resistances.append({'Date': y_date, 'Price': y_high, 'Type': '前一交易日高點'})

    if 'Net_Buy' in df.columns:
        std = df['Net_Buy'].std()
        if std > 0:
            big_b = df[(df['Net_Buy'] > 1.0 * std) & (df['Close'] > df['Open'])]
            for _, r in big_b.iterrows(): supports.append({'Date': str(r['Date']), 'Price': r['Low'], 'Type': '法人大買紅K底部'})
            big_s = df[(df['Net_Buy'] < -1.0 * std) & (df['Close'] < df['Open'])]
            for _, r in big_s.iterrows(): resistances.append({'Date': str(r['Date']), 'Price': r['High'], 'Type': '法人大賣黑K頂部'})

    recent_df = df.tail(20) 
    for _, r in recent_df.iterrows():
        if r['Volume'] > r['Vol_MA20'] * 1.5:
            d_str = str(r['Date'])
            if r['Close'] > r['Open']:
                supports.append({'Date': d_str, 'Price': r['Low'], 'Type': '帶量紅K底部'})
                resistances.append({'Date': d_str, 'Price': r['High'], 'Type': '帶量紅K頂部'})
            elif r['Close'] < r['Open']:
                resistances.append({'Date': d_str, 'Price': r['High'], 'Type': '帶量黑K頂部'})
                supports.append({'Date': d_str, 'Price': r['Low'], 'Type': '帶量黑K底部'})

    n = 5
    max_idx = argrelextrema(df['High'].values, np.greater_equal, order=n)[0]
    min_idx = argrelextrema(df['Low'].values, np.less_equal, order=n)[0]
    for i in max_idx[-10:]: resistances.append({'Date': str(df['Date'].iloc[i]), 'Price': df['High'].iloc[i], 'Type': '波段相對高點'})
    for i in min_idx[-10:]: supports.append({'Date': str(df['Date'].iloc[i]), 'Price': df['Low'].iloc[i], 'Type': '波段相對低點'})

    step = 5 if cur_price < 100 else (10 if cur_price < 500 else 50)
    lower_round = (cur_price // step) * step
    upper_round = lower_round + step
    supports.append({'Date': latest_date, 'Price': float(lower_round), 'Type': '整數心理關卡'})
    resistances.append({'Date': latest_date, 'Price': float(upper_round), 'Type': '整數心理關卡'})

    df_s = pd.DataFrame(supports).drop_duplicates()
    if not df_s.empty:
        df_s = df_s[df_s['Price'] < cur_price].copy()
        df_s['Distance'] = cur_price - df_s['Price']
    
    df_r = pd.DataFrame(resistances).drop_duplicates()
    if not df_r.empty:
        df_r = df_r[df_r['Price'] > cur_price].copy()
        df_r['Distance'] = df_r['Price'] - cur_price

    final_r = filter_smart_levels(df_r, is_support=False)
    final_s = filter_smart_levels(df_s, is_support=True)

    if not final_r.empty: final_r['Pct'] = ((final_r['Price'] - cur_price) / cur_price * 100).round(1)
    if not final_s.empty: final_s['Pct'] = ((final_s['Price'] - cur_price) / cur_price * 100).round(1)

    return final_r, final_s

# ==========================================
# Streamlit UI 區 (紅漲綠跌視覺強化版)
# ==========================================

def main():
    st.set_page_config(page_title="台股分析", layout="wide")
    
    with st.sidebar:
        st.header("🔍 設定")
        fav_list = {
            "自定義輸入": "", 
            "2301 光寶科": "2301", "2313 華通": "2313", "2317 鴻海": "2317", 
            "2330 台積電": "2330", "3017 奇鋐": "3017", "3037 欣興": "3037",  
            "3081 聯亞": "3081",  "3231 緯創": "3231", "3260 威剛": "3260", 
            "3363 上詮": "3363", "3711 日月光": "3711", "6285 啟碁": "6285", 
            "6669 緯穎": "6669"
        }
        selected_label = st.selectbox("標的選擇", list(fav_list.keys()))
        stock_id = st.text_input("輸入代碼", value="3037") if selected_label == "自定義輸入" else fav_list[selected_label]
        lookback = st.slider("回測天數", 60, 250, 120)
        btn = st.button("執行分析")

    if btn or (selected_label != "自定義輸入" and stock_id != ""):
        try:
            with st.spinner('計算中...'):
                df = get_full_data(stock_id, days=lookback)
                if df.empty: return st.error("查無資料")
                
                cur = float(df['Close'].iloc[-1])
                last_date = str(df['Date'].iloc[-1])
                r, s = analyze(df, cur)

                # 置頂資訊
                st.caption(f"📅 數據日期：{last_date}")
                st.markdown(f"## **{stock_id}** ｜ 當前：**{cur:.2f}** 元")
                st.divider()

                # --- 壓力區 (綠色區塊內容) ---
                st.success("🟢 【上漲壓力區】")
                if r.empty: 
                    st.info("無明顯壓力位")
                else:
                    for i, (_, row) in enumerate(r.iterrows()):
                        # 壓力區：百分比變紅色 :red[內容]
                        pct_text = f":red[(+{row['Pct']}%)]"
                        header_text = f"P{i+1}： {row['Price']:.2f} 元 ➜ {pct_text} | {row['Type']}"
                        
                        with st.expander(header_text):
                            st.markdown(f"**發生日期：** `{row['Date']}`")

                st.write("") 

                # --- 支撐區 (紅色區塊內容) ---
                st.error("🔴 【下跌支撐區】")
                if s.empty: 
                    st.info("無明顯支撐位")
                else:
                    for i, (_, row) in enumerate(s.iterrows()):
                        # 支撐區：百分比變綠色 :green[內容]
                        pct_text = f":green[({row['Pct']}%)]"
                        header_text = f"S{i+1}： {row['Price']:.2f} 元 ➜ {pct_text} | {row['Type']}"
                        
                        with st.expander(header_text):
                            st.markdown(f"**發生日期：** `{row['Date']}`")

        except Exception as e:
            st.error(f"分析錯誤: {e}")

if __name__ == "__main__":
    main()