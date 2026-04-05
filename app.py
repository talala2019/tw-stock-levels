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
    
    # 【關鍵修正 1】：距離絕對優先 (Distance 升序排第一)
    # 不再讓最新日期搶佔版面，而是讓「最接近現在價格」的訊號排前面
    df_levels = df_levels.sort_values(['Distance', 'Date'], ascending=[True, False])
    
    # 【關鍵修正 2】：過濾掉太遠的訊號 (例如超過 20% 的不顯示，避免手機版訊息冗長)
    # 算一下百分比，如果距離太遠就捨棄
    
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
    # 最終輸出排序：壓力由小到大，支撐由大到小
    return res_df.sort_values('Price', ascending=not is_support)
    
@st.cache_data(ttl=3600)
def get_full_data(stock_id, days=380): # 為了年線 MA240，預設拉 360 天
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime('%Y-%m-%d')

    # 策略：先試 .TW，若失敗或資料太少則試 .TWO
    df = pd.DataFrame()
    found_ticker = f"{stock_id}.TW"
    
    try:
        # 第一嘗試：上市 (.TW)
        df = yf.download(found_ticker, start=start_str, progress=False)
        
        # 如果 df 為空，或是最近一筆資料不完整，嘗試上櫃 (.TWO)
        if df.empty or len(df) < 5:
            found_ticker = f"{stock_id}.TWO"
            df = yf.download(found_ticker, start=start_str, progress=False)
            
    except Exception:
        # 萬一發生報錯，最後保險嘗試上櫃
        found_ticker = f"{stock_id}.TWO"
        df = yf.download(found_ticker, start=start_str, progress=False)

    if df.empty:
        return pd.DataFrame()

    # 處理 MultiIndex 欄位問題 (yfinance 新版特性)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    # 籌碼資料抓取 (維持原邏輯)
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
    
    # 1. 基礎指標 (加入長天期均線 MA60, MA120, MA240)
    ma_list = ['MA5', 'MA10', 'MA20', 'MA60', 'MA120', 'MA240']
    for ma in ma_list:
        days = int(ma.replace('MA', ''))
        df[ma] = df['Close'].rolling(days).mean()
        
        val = round(df[ma].iloc[-1], 2)
        if not np.isnan(val):
            # 判斷是支撐還是壓力
            if val < cur_price:
                supports.append({'Date': latest_date, 'Price': val, 'Type': f'{ma}位置'})
            else:
                resistances.append({'Date': latest_date, 'Price': val, 'Type': f'{ma}位置'})
    
    # 20日成交量均線 (維持原邏輯)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    # --- 以下維持你原本的邏輯 (昨日高低、法人、量價 K 線、波段、整數關卡) ---
    # --- 加入成交量檢查的前一交易日高低點 ---
    if len(df) >= 2:
        # 取得前一交易日的資料與 20 日均量
        yesterday = df.iloc[-2]
        y_open = yesterday['Open']
        y_low = yesterday['Low']
        y_high = yesterday['High']
        y_close = yesterday['Close']
        y_vol = yesterday['Volume']
        y_vol_ma20 = yesterday['Vol_MA20']
        y_date = str(yesterday['Date'])

        # 【關鍵修改】：增加成交量必須大於 20 日均量的條件
        if y_vol > y_vol_ma20:
            # 支撐：前一交易日是紅K (收 > 開) 且 低點低於當前價
            if (y_low < cur_price) and (y_close > y_open): 
                supports.append({'Date': y_date, 'Price': y_low, 'Type': '前一交易日低點(量增紅K)'})
            # 壓力：前一交易日是黑K (收 < 開) 且 高點高於當前價
            if (y_high > cur_price) and (y_close < y_open): 
                resistances.append({'Date': y_date, 'Price': y_high, 'Type': '前一交易日高點(量增黑K)'})

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
                #resistances.append({'Date': d_str, 'Price': r['High'], 'Type': '帶量紅K頂部'})
            elif r['Close'] < r['Open']:
                resistances.append({'Date': d_str, 'Price': r['High'], 'Type': '帶量黑K頂部'})
                #supports.append({'Date': d_str, 'Price': r['Low'], 'Type': '帶量黑K底部'})

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

    # --- 修正後的整理與距離計算 ---
    df_s = pd.DataFrame(supports).drop_duplicates()
    if not df_s.empty:
        df_s = df_s[df_s['Price'] < cur_price].copy()
        df_s['Distance'] = cur_price - df_s['Price']
        df_s['Pct'] = ((df_s['Price'] - cur_price) / cur_price * 100).round(1) # 先算百分比供過濾
    
    df_r = pd.DataFrame(resistances).drop_duplicates()
    if not df_r.empty:
        df_r = df_r[df_r['Price'] > cur_price].copy()
        df_r['Distance'] = df_r['Price'] - cur_price
        df_r['Pct'] = ((df_r['Price'] - cur_price) / cur_price * 100).round(1) # 先算百分比供過濾

    # 執行智慧過濾 (現在會優先選近的)
    final_r = filter_smart_levels(df_r, is_support=False)
    final_s = filter_smart_levels(df_s, is_support=True)

    return final_r, final_s

# ==========================================
# Streamlit UI 區 (手機優先 - 無側邊欄版)
# ==========================================

def main():
    # 設定頁面，不再需要側邊欄
    st.set_page_config(page_title="台股分析", layout="wide", initial_sidebar_state="collapsed")
    
    # 直接在主畫面上方顯示標題與輸入區
    st.title("📈 台股支撐壓力分析")
    
    # 建立輸入區
    fav_list = {
        "自定義輸入": "", 
        "2301 光寶科": "2301", "2313 華通": "2313", "2317 鴻海": "2317", 
        "2330 台積電": "2330", "3017 奇鋐": "3017", "3037 欣興": "3037",  
        "3081 聯亞": "3081",  "3231 緯創": "3231", "3260 威剛": "3260", 
        "3363 上詮": "3363", "3711 日月光": "3711", "6285 啟碁": "6285", 
        "6669 緯穎": "6669"
    }
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_label = st.selectbox("選擇標的", list(fav_list.keys()), label_visibility="collapsed")
        if selected_label == "自定義輸入":
            stock_id = st.text_input("輸入 4 位代碼", value="")
        else:
            stock_id = fav_list[selected_label]
    
    with col2:
        # 固定回測天數為 380
        lookback = 380
        btn = st.button("執行分析", use_container_width=True)

    if btn or (selected_label != "自定義輸入" and stock_id != ""):
        try:
            with st.spinner('計算中...'):
                df = get_full_data(stock_id, days=lookback)
                if df.empty: return st.error("查無資料，請檢查代碼是否正確")
                
                cur = float(df['Close'].iloc[-1])
                last_date = str(df['Date'].iloc[-1])
                r, s = analyze(df, cur)

                # 置頂資訊：日期與價格
                st.caption(f"📅 數據日期：{last_date} ( 回測380天，已計算至 MA240 年線 )")
                st.markdown(f"## **{stock_id}** ｜ 收盤：**{cur:.2f}** 元")
                st.divider()

                # --- 壓力區 (綠色) ---
                st.success("🟢 【上漲壓力區】")
                if r.empty: 
                    st.info("上方無明顯壓力位")
                else:
                    for i, (_, row) in enumerate(r.iterrows()):
                        pct_text = f":red[(+{row['Pct']}%)]"
                        header_text = f"P{i+1}： {row['Price']:.2f} 元 ➜ {pct_text} | {row['Type']}"
                        with st.expander(header_text):
                            st.markdown(f"**發生日期：** `{row['Date']}`")

                st.write("") 

                # --- 支撐區 (紅色) ---
                st.error("🔴 【下跌支撐區】")
                if s.empty: 
                    st.info("下方無明顯支撐位")
                else:
                    for i, (_, row) in enumerate(s.iterrows()):
                        pct_text = f":green[({row['Pct']}%)]"
                        header_text = f"S{i+1}： {row['Price']:.2f} 元 ➜ {pct_text} | {row['Type']}"
                        with st.expander(header_text):
                            st.markdown(f"**發生日期：** `{row['Date']}`")

        except Exception as e:
            st.error(f"分析錯誤: {e}")

if __name__ == "__main__":
    main()