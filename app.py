import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from FinMind.data import DataLoader
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 輔助函式：判斷顏色與正負號
def get_clr(val):
    return "#ff4b4b" if val > 0 else ("#00ad00" if val < 0 else "#333")

def get_sign(val):
    return "+" if val > 0 else ""

# ==========================================
# 核心邏輯區
# ==========================================

def filter_smart_levels(df_levels, cur_price, is_support=True, dynamic_threshold=0.015):
    if df_levels.empty: return df_levels
    
    df_levels = df_levels[abs(df_levels['Pct']) <= 20.0]
    df_levels = df_levels.sort_values(['Distance', 'Date'], ascending=[True, False])
    
    clusters = []
    for _, row in df_levels.iterrows():
        price, typ, date = row['Price'], row['Type'], row['Date']
        merged = False
        for cluster in clusters:
            if abs(price - cluster['mean_price']) / cluster['mean_price'] <= dynamic_threshold:
                cluster['prices'].append(price)
                cluster['types'].append(typ)
                cluster['dates'].append(date)
                cluster['mean_price'] = sum(cluster['prices']) / len(cluster['prices'])
                merged = True
                break
        if not merged:
            clusters.append({'prices': [price], 'mean_price': price, 'types': [typ], 'dates': [date]})
            
    selected = []
    for c in clusters:
        mean_p = c['mean_price']
        dist = abs(cur_price - mean_p)
        pct = ((mean_p - cur_price) / cur_price) * 100
        
        # --- 核心修改：日期去重、轉物件、降序排序 ---
        date_objs = pd.to_datetime(list(set(c['dates'])))
        sorted_dates = sorted(date_objs, reverse=True) 
        unique_dates_str = [d.strftime('%Y-%m-%d') for d in sorted_dates]
        
        unique_types = list(dict.fromkeys(c['types']))
                
        selected.append({
            'Price': round(mean_p, 2),
            'Pct': round(pct, 1),
            'Distance': dist,
            'All_Dates': " | ".join(unique_dates_str), # 這是給 UI 用的新欄位
            'Signal_Count': len(c['prices']),
            'Types_Merged': "、".join(unique_types)
        })
        
    res_df = pd.DataFrame(selected)
    if res_df.empty: return res_df
    
    res_df = res_df.sort_values('Distance', ascending=True).head(5)
    return res_df.sort_values('Price', ascending=not is_support)

@st.cache_data(ttl=3600)
def get_full_data(stock_id, days=380): 
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime('%Y-%m-%d')

    df = pd.DataFrame()
    for ext in [".TW", ".TWO"]:
        try:
            df = yf.download(f"{stock_id}{ext}", start=start_str, progress=False)
            if not df.empty and len(df) >= 5: break
        except: pass

    if df.empty: return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    dl = DataLoader()
    try:
        df_inst = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_str)
        if not df_inst.empty:
            df_inst['Date'] = pd.to_datetime(df_inst['date']).dt.date
            if 'buy_sell' not in df_inst.columns: df_inst['buy_sell'] = df_inst['buy'] - df_inst['sell']
            inst_sum = (df_inst.groupby('Date')['buy_sell'].sum() / 1000).reset_index()
            inst_sum.rename(columns={'buy_sell': 'Net_Buy'}, inplace=True)
            df = pd.merge(df, inst_sum, on='Date', how='left').fillna(0)
        else: df['Net_Buy'] = 0
    except: df['Net_Buy'] = 0
    
    return df

def analyze(df, cur_price):
    supports, resistances = [], []
    latest_date = str(df['Date'].iloc[-1])
    current_idx = len(df) - 1
    
    # 1. 均線系統
    for ma in [5, 10, 20, 60, 120, 240]:
        df[f'MA{ma}'] = df['Close'].rolling(ma).mean()
        val = round(df[f'MA{ma}'].iloc[-1], 2)
        if not np.isnan(val):
            target_list = supports if val < cur_price else resistances
            target_list.append({'Date': latest_date, 'Price': val, 'Type': f'MA{ma}位置'})
    
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    # 2. 近期 K 線型態與法人買賣超紅黑K
    if len(df) >= 2:
        yesterday = df.iloc[-2]
        if yesterday['Volume'] > yesterday['Vol_MA20']:
            if (yesterday['Low'] < cur_price) and (yesterday['Close'] > yesterday['Open']): 
                supports.append({'Date': str(yesterday['Date']), 'Price': yesterday['Low'], 'Type': '前日量增紅K低點'})
            if (yesterday['High'] > cur_price) and (yesterday['Close'] < yesterday['Open']): 
                resistances.append({'Date': str(yesterday['Date']), 'Price': yesterday['High'], 'Type': '前日量增黑K高點'})

    if 'Net_Buy' in df.columns:
        chip_recent_df = df.tail(60)
        std = chip_recent_df['Net_Buy'].std()
        if std > 0:
            for _, r in chip_recent_df[(chip_recent_df['Net_Buy'] > 1.0 * std) & (chip_recent_df['Close'] > chip_recent_df['Open'])].iterrows(): 
                supports.append({'Date': str(r['Date']), 'Price': r['Low'], 'Type': '法人大買紅K底部'})
            for _, r in chip_recent_df[(chip_recent_df['Net_Buy'] < -1.0 * std) & (chip_recent_df['Close'] < chip_recent_df['Open'])].iterrows(): 
                resistances.append({'Date': str(r['Date']), 'Price': r['High'], 'Type': '法人大賣黑K頂部'})

        # 【新增 1】：法人成本防線 (60日均建倉成本 VWAP)
        buy_days = chip_recent_df[chip_recent_df['Net_Buy'] > 0].copy()
        if not buy_days.empty and buy_days['Net_Buy'].sum() > 0:
            buy_days['Typical'] = (buy_days['High'] + buy_days['Low'] + buy_days['Close']) / 3
            inst_cost = (buy_days['Typical'] * buy_days['Net_Buy']).sum() / buy_days['Net_Buy'].sum()
            if inst_cost > 0:
                target_list = supports if inst_cost < cur_price else resistances
                target_list.append({'Date': latest_date, 'Price': round(inst_cost, 2), 'Type': '60日法人成本防線'})

    # 【新增 2】：大量套牢區 / 大量換手區 (近 120 天前 3 大成交量)
    top_vol_days = df.tail(120).nlargest(3, 'Volume')
    for _, row in top_vol_days.iterrows():
        date_str = str(row['Date'])
        if row['Close'] > cur_price:
            resistances.append({'Date': date_str, 'Price': row['Close'], 'Type': '大量套牢區'})
        else:
            supports.append({'Date': date_str, 'Price': row['Close'], 'Type': '大量換手支撐區'})

    # 【新增 3 & 4】：波段支撐/壓力 & 近期點連線 (趨勢線投影)
    # 將 order 設為 10 來抓取較明顯的波段轉折
    n_swing = 10
    max_idx = argrelextrema(df['High'].values, np.greater_equal, order=n_swing)[0]
    min_idx = argrelextrema(df['Low'].values, np.less_equal, order=n_swing)[0]
    
    # 波段支撐/壓力
    for i in max_idx[-3:]: 
        resistances.append({'Date': str(df['Date'].iloc[i]), 'Price': df['High'].iloc[i], 'Type': '波段壓力'})
    for i in min_idx[-3:]: 
        supports.append({'Date': str(df['Date'].iloc[i]), 'Price': df['Low'].iloc[i], 'Type': '波段支撐'})
        
    # 近期高點連線 (下降壓力線)
    if len(max_idx) >= 2:
        idx1, idx2 = max_idx[-2], max_idx[-1]
        if idx1 != idx2 and idx2 < current_idx: # 確保不是同一天且有投影空間
            slope = (df['High'].iloc[idx2] - df['High'].iloc[idx1]) / (idx2 - idx1)
            proj_r = df['High'].iloc[idx2] + slope * (current_idx - idx2)
            if proj_r > cur_price:
                resistances.append({'Date': latest_date, 'Price': round(proj_r, 2), 'Type': '近期高點連線'})

    # 近期低點連線 (上升支撐線)
    if len(min_idx) >= 2:
        idx1, idx2 = min_idx[-2], min_idx[-1]
        if idx1 != idx2 and idx2 < current_idx:
            slope = (df['Low'].iloc[idx2] - df['Low'].iloc[idx1]) / (idx2 - idx1)
            proj_s = df['Low'].iloc[idx2] + slope * (current_idx - idx2)
            if proj_s > 0 and proj_s < cur_price:
                supports.append({'Date': latest_date, 'Price': round(proj_s, 2), 'Type': '近期低點連線'})

    # 整數關卡
    step = 10 if cur_price < 100 else (50 if cur_price < 500 else 100)
    lower_round, upper_round = (cur_price // step) * step, (cur_price // step) * step + step
    supports.append({'Date': latest_date, 'Price': float(lower_round), 'Type': '整數心理關卡'})
    resistances.append({'Date': latest_date, 'Price': float(upper_round), 'Type': '整數心理關卡'})

    # 執行清洗與動態分群
    df_s = pd.DataFrame(supports).drop_duplicates(subset=['Price', 'Type'])
    if not df_s.empty:
        df_s = df_s[df_s['Price'] < cur_price].copy()
        df_s['Distance'], df_s['Pct'] = cur_price - df_s['Price'], ((df_s['Price'] - cur_price) / cur_price * 100).round(1)
    
    df_r = pd.DataFrame(resistances).drop_duplicates(subset=['Price', 'Type'])
    if not df_r.empty:
        df_r = df_r[df_r['Price'] > cur_price].copy()
        df_r['Distance'], df_r['Pct'] = df_r['Price'] - cur_price, ((df_r['Price'] - cur_price) / cur_price * 100).round(1)

    final_r = filter_smart_levels(df_r, cur_price, is_support=False)
    final_s = filter_smart_levels(df_s, cur_price, is_support=True)

    return final_r, final_s

# ==========================================
# Streamlit UI 區
# ==========================================

def main():
    st.markdown("<h3 style='margin-bottom: 0px;'>📈 台股支撐壓力分析</h3>", unsafe_allow_html=True)
    
    fav_list = {
        "自定義輸入": "", "2301 光寶科": "2301", "2308 台達電": "2308", "2313 華通": "2313",
        "2317 鴻海": "2317", "2330 台積電": "2330", "2337 旺宏": "2337", "2449 京元電": "2449", 
        "2451 創見": "2451", "2454 聯發科": "2454", "2455 全新": "2455", "3017 奇鋐": "3017", 
        "3037 欣興": "3037", "3081 聯亞": "3081", "3105 穩懋": "3105", "3163 波若威": "3163", 
        "3231 緯創": "3231", "3260 威剛": "3260", "3293 鈊象": "3293", "3324 雙鴻": "3324",
        "3363 上詮": "3363", "3450 聯鈞": "3450", "3711 日月光": "3711", "4722 國精化": "4722", 
        "4979 華星光": "4979","5340 建榮": "5340", "5475 德宏": "5475", "6285 啟碁": "6285", "6442 光聖": "6442", 
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
                
                # --- 周轉率抓取邏輯 ---
                ticker = yf.Ticker(f"{stock_id}.TW")
                shares = ticker.info.get('sharesOutstanding') or ticker.info.get('floatShares') or 0
                if shares == 0:
                    try:
                        shares_series = ticker.get_shares_full(start="2025-01-01")
                        if not shares_series.empty: shares = shares_series.iloc[-1]
                    except: shares = 0
                
                cur_vol_shares = df['Volume'].iloc[-1]
                turnover_rate = (cur_vol_shares / shares * 100) if shares > 0 else 0
                
                # --- 數據計算區 ---
                cur_close = float(df['Close'].iloc[-1])
                prev_close = float(df['Close'].iloc[-2])
                y_open, y_high, y_low = df['Open'].iloc[-1], df['High'].iloc[-1], df['Low'].iloc[-1]
                
                # 【關鍵修正】：先執行分析，讓 df 產生 MA20, MA240, Vol_MA20 等欄位
                r, s = analyze(df, cur_close) 

                # 現在這些欄位已經存在了，可以安全計算
                pct_3d = ((cur_close - df['Close'].iloc[-4]) / df['Close'].iloc[-4] * 100) if len(df)>=4 else 0
                pct_10d = ((cur_close - df['Close'].iloc[-11]) / df['Close'].iloc[-11] * 100) if len(df)>=11 else 0
                pct_60d = ((cur_close - df['Close'].iloc[-61]) / df['Close'].iloc[-61] * 100) if len(df)>=61 else 0
                
                high_60, low_60 = df.tail(60)['High'].max(), df.tail(60)['Low'].min()
                
                # 安全讀取 MA 數據（確保 analyze 已運行）
                bias_20 = ((cur_close - df['MA20'].iloc[-1]) / df['MA20'].iloc[-1] * 100) if 'MA20' in df.columns else 0
                bias_240 = ((cur_close - df['MA240'].iloc[-1]) / df['MA240'].iloc[-1] * 100) if 'MA240' in df.columns else 0
                
                ma20_slope_pct = ((df['MA20'].iloc[-1] - df['MA20'].iloc[-2]) / df['MA20'].iloc[-2] * 100) if len(df)>20 else 0
                vol_ratio = (df['Volume'].iloc[-1] / df['Vol_MA20'].iloc[-1]) if 'Vol_MA20' in df.columns and df['Vol_MA20'].iloc[-1] > 0 else 0

                # --- UI 顯示區 ---
                st.caption(f"📅 數據日期：{df['Date'].iloc[-1]}")
                
                diff, pct = cur_close - prev_close, ((cur_close - prev_close) / prev_close) * 100
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
                
                total_range = max(y_high - y_low, 1)
                body_top, body_bottom = max(y_open, cur_close), min(y_open, cur_close)
                up_shadow_p, body_p, low_shadow_p = ((y_high - body_top) / total_range) * 100, ((body_top - body_bottom) / total_range) * 100, ((body_bottom - y_low) / total_range) * 100
                
                k_color = "#ff4b4b" if cur_close >= y_open else "#00ad00"
                slope_clr, vol_clr = "#ff4b4b" if ma20_slope_pct > 0 else "#00ad00", "#1f77b4" if vol_ratio > 1.2 else ("#999" if vol_ratio < 0.8 else "#666")
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
                        with st.expander(f"P{i+1}： {row['Price']:.1f} 元 ➜ :red[(+{row['Pct']}%)] | 🔥融合 {row['Signal_Count']} 個訊號"): 
                            st.markdown(f"**訊號來源：** `{row['Types_Merged']}`\n\n**觸發歷史(新→舊)：**\n`{row['All_Dates']}`")
                
                st.error("🔴 【下跌支撐區】")
                if not s.empty:
                    for i, (_, row) in enumerate(s.iterrows()):
                        with st.expander(f"S{i+1}： {row['Price']:.1f} 元 ➜ :green[({row['Pct']}%)] | 🔥融合 {row['Signal_Count']} 個訊號"): 
                            st.markdown(f"**訊號來源：** `{row['Types_Merged']}`\n\n**觸發歷史(新→舊)：**\n`{row['All_Dates']}`")

        except Exception as e:
            st.error(f"分析錯誤: {e}")

if __name__ == "__main__":
    main()