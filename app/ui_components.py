import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from config import ZONE_COLORS, ZONE_BG, ZONE_TEXT, MONTH_NAMES, WMO_CODES

def render_sidebar(M, df, helpers):
    station = M['station']
    zone_names = M['zone_names']
    
    with st.sidebar:
        st.markdown("""
        <div style='padding:20px 0 28px;text-align:center'>
            <div style='font-size:2.8rem'>🌿</div>
            <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:600;
                        color:#7ab648;letter-spacing:0.08em'>Intelli-Climate</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#5a7a50;margin-bottom:10px'>Dataset Overview</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:0.83rem;color:#8aaa78;line-height:2'>
            📍 30 Cities across Sri Lanka<br>
            📅 Jan 2010 – Jun 2023<br>
            📊 {len(df):,} daily records<br>
            🌧️ {df['is_rainy'].mean():.1%} rainy days average
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#5a7a50;margin-bottom:10px'>Climate Regions</div>", unsafe_allow_html=True)
        for zone_id, zone_name in sorted(zone_names.items()):
            cities_in = station[station['climate_zone']==zone_id]['city'].tolist()
            if not cities_in: continue
            color = ZONE_COLORS.get(zone_id, '#9a9088')
            st.markdown(f"""
            <div style='margin-bottom:10px;padding:10px 12px;
                        background:rgba(255,255,255,0.05);border-radius:8px;
                        border-left:3px solid {color}'>
                <div style='color:{color};font-size:0.8rem;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.05em'>{zone_name}</div>
                <div style='color:#6a8a60;font-size:0.72rem;margin-top:3px'>{len(cities_in)} stations</div>
            </div>""", unsafe_allow_html=True)

def render_header():
    st.markdown("""
    <div style='padding:10px 0 8px'>
        <div style='font-family:Roboto,sans-serif;font-size:2.8rem;font-weight:800;
                    color:#1a1a1a;line-height:1.1;margin-bottom:8px'>
            Sri Lankan Climate &amp; Weather Predictor 
        </div>
        <div style='font-size:1rem;color:#7a7068;max-width: 800px;line-height:1.6'>
            Explore climatic patterns and predict weather across 30 Sri Lankan cities
            built from 13 years of historical weather data.
        </div>
    </div>
    <hr style='margin:24px 0 0'>
    """, unsafe_allow_html=True)

def render_climate_zones(station, zone_names):
    st.markdown('<div class="sec-title">Climate Zone Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Distinct climate regions discovered across 30 Sri Lankan cities based on temperature, precipitation, and wind patterns.</div>', unsafe_allow_html=True)

    map_col, detail_col = st.columns([2, 1], gap="large")

    with map_col:
        fig = go.Figure()
        for zone_id in sorted(station['climate_zone'].unique()):
            sub = station[station['climate_zone'] == zone_id]
            zname = zone_names.get(zone_id, f'Zone {zone_id}')
            color = ZONE_COLORS.get(zone_id, '#9a9088')
            hover = sub.apply(lambda r:
                f"<b>{r['city']}</b><br>"
                f"Region: {zname}<br>"
                f"Avg Temp: {r['temperature_2m_mean']:.1f}°C<br>"
                f"Avg Rain: {r['precipitation_sum']:.1f} mm/day<br>"
                f"Avg Wind: {r['windspeed_10m_max']:.1f} km/h", axis=1)
            fig.add_trace(go.Scattergeo(
                lon=sub['longitude'], lat=sub['latitude'],
                mode='markers+text',
                name=zname,
                text=sub['city'],
                textposition='top center',
                textfont=dict(size=8.5, color='#333'),
                hovertemplate=hover + '<extra></extra>',
                marker=dict(
                    size=20 if zone_id != -1 else 14,
                    color=color,
                    symbol='circle',
                    line=dict(width=2.5, color='white'),
                    opacity=0.92
                )
            ))
        fig.update_layout(
            geo=dict(
                scope='asia', showland=True, landcolor='#e8e2d9',
                showocean=True, oceancolor='#d0e8f5',
                showcountries=True, countrycolor='#b0a898',
                showcoastlines=True, coastlinecolor='#8a7a6a',
                showlakes=True, lakecolor='#d0e8f5',
                center=dict(lat=7.85, lon=80.7),
                projection_scale=11,
                lataxis_range=[5.5, 10.3],
                lonaxis_range=[79.2, 82.3],
                bgcolor='#f5f2ee',
            ),
            paper_bgcolor='#ffffff', plot_bgcolor='#f5f2ee',
            margin=dict(l=0,r=0,t=0,b=0), height=520,
            legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#e4dfd8',
                        borderwidth=1, font=dict(size=12, color='#333'),
                        x=0.01, y=0.99),
            font=dict(color='#1a1a1a', family='Instrument Sans')
        )
        st.plotly_chart(fig, use_container_width=True)

    with detail_col:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        for zone_id in sorted(zone_names.keys()):
            cities_in = station[station['climate_zone']==zone_id]
            if len(cities_in) == 0: continue
            zname = zone_names[zone_id]
            color = ZONE_COLORS.get(zone_id, '#9a9088')
            bg    = ZONE_BG.get(zone_id, '#f5f2ee')
            avg_r = cities_in['precipitation_sum'].mean()
            avg_t = cities_in['temperature_2m_mean'].mean()
            avg_w = cities_in['windspeed_10m_max'].mean()
            city_list = ', '.join(cities_in['city'].tolist())
            st.markdown(f"""
            <div style='background:{bg};border-radius:14px;padding:16px 18px;
                        margin-bottom:14px;border-left:4px solid {color}'>
                <div style='color:{color};font-family:Syne,sans-serif;font-size:0.95rem;
                            font-weight:700;text-transform:uppercase;letter-spacing:0.05em'>
                    {zname}
                </div>
                <div style='display:flex;gap:16px;margin:8px 0;flex-wrap:wrap'>
                    <span style='font-size:0.78rem;color:#5a5048'>
                        🌡️ {avg_t:.1f}°C
                    </span>
                    <span style='font-size:0.78rem;color:#5a5048'>
                        🌧️ {avg_r:.1f}mm/day
                    </span>
                    <span style='font-size:0.78rem;color:#5a5048'>
                        💨 {avg_w:.1f}km/h
                    </span>
                </div>
                <div style='font-size:0.76rem;color:#7a7068;line-height:1.6'>
                    {city_list}
                </div>
            </div>""", unsafe_allow_html=True)

def render_predictor(M, helpers):
    st.markdown('<div class="sec-title">Weather Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Select a city and configure weather parameters to predict the likelihood of rainfall.</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")
    
    with left:
        st.markdown("#### 📍 Location & Date")
        city_opt = helpers['in_model']
        city = st.selectbox("City", city_opt,
                            index=city_opt.index('Colombo') if 'Colombo' in city_opt else 0)
        
        zone = helpers['zone_map'].get(city, 0)
        zname = M['zone_names'].get(zone, '')
        color = ZONE_COLORS.get(zone, '#9a9088')
        bg    = ZONE_BG.get(zone, '#f5f2ee')
        
        st.markdown(f"""
        <div style='background:{bg};border-left:4px solid {color};border-radius:0 10px 10px 0;
                    padding:10px 16px;margin-bottom:16px'>
            <span style='color:{color};font-weight:700;font-size:0.85rem'>{zname}</span>
            <span style='color:#9a9088;font-size:0.78rem;margin-left:8px'>
                {helpers['coords'][city]['latitude']:.2f}°N · {helpers['coords'][city]['longitude']:.2f}°E
            </span>
        </div>""", unsafe_allow_html=True)

        month = st.select_slider("Month", options=list(range(1,13)),
                                  format_func=lambda x: MONTH_NAMES[x], value=6)

        st.markdown("#### 🌡️ Temperature")
        temp_mean = st.slider("Mean Temperature (°C)", 14.0, 34.0, 26.5, 0.5)
        c1, c2 = st.columns(2)
        with c1:
            temp_max = st.slider("Max (°C)", float(temp_mean), 40.0, min(temp_mean+4,40.0), 0.5)
        with c2:
            temp_min = st.slider("Min (°C)", 10.0, float(temp_mean), max(temp_mean-5,10.0), 0.5)

        st.markdown("#### 🌬️ Atmospheric")
        wind      = st.slider("Max Wind Speed (km/h)", 0.0, 50.0, 14.0, 0.5)
        radiation = st.slider("Solar Radiation (MJ/m²)", 4.0, 30.0, 17.0, 0.5)
        et0       = st.slider("Evapotranspiration (mm)", 1.0, 9.0, 4.2, 0.1)

        temp_range = round(temp_max - temp_min, 2)
        st.caption(f"Computed temp range: **{temp_range:.1f}°C** (max − min)")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        predict = st.button("⚡  Run Forecast", type="primary")

    with right:
        if predict:
            feats = np.array([[temp_mean, temp_max, temp_min,
                                wind, radiation, et0, temp_range, month, zone]])
            feats_sc = M['scaler_cls'].transform(feats)
            pred  = M['gnb'].predict(feats_sc)[0]
            proba = M['gnb'].predict_proba(feats_sc)[0]
            p_rain = proba[1]
            p_dry  = proba[0]

            if pred == 1:
                st.markdown(f"""
                <div class='result-rainy'>
                    <div class='result-icon'>🌧️</div>
                    <div class='result-title'>RAIN FORECASTED</div>
                    <div class='result-conf'>Likelihood: <b>{p_rain*100:.1f}%</b></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-dry'>
                    <div class='result-icon'>☀️</div>
                    <div class='result-title'>DRY WEATHER FORECASTED</div>
                    <div class='result-conf'>Likelihood: <b>{p_dry*100:.1f}%</b></div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            fig_prob = go.Figure(go.Bar(
                x=[p_dry*100, p_rain*100],
                y=['☀️ Dry', '🌧️ Rainy'],
                orientation='h',
                marker_color=['#c47d2d','#2d6abf'],
                text=[f'{p_dry*100:.1f}%', f'{p_rain*100:.1f}%'],
                textposition='inside',
                textfont=dict(color='white', size=13, family='Syne'),
            ))
            fig_prob.update_layout(
                height=110, margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor='white', plot_bgcolor='white',
                xaxis=dict(range=[0,100], showgrid=False, visible=False),
                yaxis=dict(showgrid=False),
                font=dict(color='#1a1a1a', family='Instrument Sans'),
                showlegend=False, bargap=0.3,
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # Historical context
            st.markdown(f"#### 📊 Historical Trends: {city} in {MONTH_NAMES[month]}")
            df = M['df']
            hist = df[(df['city']==city) & (df['month']==month)]
            if len(hist):
                h_rain = hist['is_rainy'].mean()
                h_avg  = hist['precipitation_sum'].mean()
                c1,c2,c3 = st.columns(3)
                c1.metric("Historical Rain Days", f"{h_rain:.0%}")
                c2.metric("Avg Daily Rain", f"{h_avg:.1f} mm")
                c3.metric("Current Forecast", "🌧️ Rain" if pred==1 else "☀️ Dry")
        else:
            st.markdown("""
            <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
                        height:500px;border:2px dashed #d4cfc8;border-radius:20px;
                        background:#faf8f5;text-align:center;padding:40px'>
                <div style='font-size:4rem;margin-bottom:16px'>⚡</div>
                <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:700;
                            color:#1a1a1a;margin-bottom:8px'>Configure & Forecast</div>
                <div style='font-size:0.88rem;color:#9a9088;max-width:280px;line-height:1.6'>
                    Select a city and set the weather parameters on the left, then
                    click <b>Run Forecast</b> for a rainfall prediction.
                </div>
            </div>""", unsafe_allow_html=True)

def render_explorer(M, helpers):
    st.markdown('<div class="sec-title">City Weather Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Explore historical weather patterns, seasonal cycles, and trends for any city.</div>', unsafe_allow_html=True)

    df = M['df']
    all_cities = helpers['all_cities']
    
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
    with ctrl1:
        sel_city = st.selectbox("City", all_cities, key='explorer_city',
                                 index=all_cities.index('Colombo') if 'Colombo' in all_cities else 0)
    with ctrl2:
        chart_type = st.selectbox("Chart Type", [
            "Monthly Rainfall Pattern",
            "Temperature Range by Month",
            "Rainy Day Frequency",
            "Annual Rainfall Trend",
            "Weather Code Distribution",
            "All-City Rainfall Heatmap",
        ])
    with ctrl3:
        yr_min, yr_max = st.select_slider("Year Range", options=list(range(2010,2024)),
                                            value=(2010, 2023))

    city_df = df[(df['city']==sel_city) &
                 (df['time'].dt.year.between(yr_min, yr_max))].copy()
    city_df['year'] = city_df['time'].dt.year

    MNAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig_style = dict(paper_bgcolor='white', plot_bgcolor='#fafaf8',
                     font=dict(color='#1a1a1a', family='Instrument Sans'),
                     margin=dict(l=60,r=20,t=50,b=50))

    fig_c = None
    if chart_type == "Monthly Rainfall Pattern":
        monthly = city_df.groupby('month')['precipitation_sum'].mean().reset_index()
        avg = monthly['precipitation_sum'].mean()
        bar_c = ['#2d6abf' if v >= avg else '#7ab6df' for v in monthly['precipitation_sum']]
        fig_c = go.Figure([
            go.Bar(x=[MNAMES[m-1] for m in monthly['month']],
                   y=monthly['precipitation_sum'], marker_color=bar_c,
                   name='Avg Daily Rain', hovertemplate='%{x}: %{y:.2f} mm<extra></extra>'),
            go.Scatter(x=MNAMES, y=[avg]*12, mode='lines',
                       line=dict(color='#c47d2d', dash='dash', width=2),
                       name=f'Annual avg ({avg:.2f}mm)')
        ])
    elif chart_type == "Temperature Range by Month":
        monthly = city_df.groupby('month').agg(
            max_t=('temperature_2m_max','mean'),
            mean_t=('temperature_2m_mean','mean'),
            min_t=('temperature_2m_min','mean')).reset_index()
        mx = [MNAMES[m-1] for m in monthly['month']]
        fig_c = go.Figure([
            go.Scatter(x=mx, y=monthly['max_t'], mode='lines+markers', name='Max', line=dict(color='#c47d2d',width=2.5)),
            go.Scatter(x=mx+mx[::-1], y=list(monthly['max_t'])+list(monthly['min_t'])[::-1], fill='toself', fillcolor='rgba(45,106,191,0.1)', line=dict(color='rgba(0,0,0,0)'), showlegend=False),
            go.Scatter(x=mx, y=monthly['mean_t'], mode='lines+markers', name='Mean', line=dict(color='#2d6abf',width=2.5)),
            go.Scatter(x=mx, y=monthly['min_t'], mode='lines+markers', name='Min', line=dict(color='#7ab648',width=2)),
        ])
    elif chart_type == "Rainy Day Frequency":
        monthly = city_df.groupby('month')['is_rainy'].mean().reset_index()
        fig_c = go.Figure(go.Bar(x=[MNAMES[m-1] for m in monthly['month']], y=monthly['is_rainy']*100, marker_color='#2d6abf'))
    elif chart_type == "Annual Rainfall Trend":
        annual = city_df.groupby('year')['precipitation_sum'].sum().reset_index()
        fig_c = go.Figure(go.Scatter(x=annual['year'], y=annual['precipitation_sum'], mode='lines+markers', marker_color='#2d6abf'))
    elif chart_type == "Weather Code Distribution":
        wc = city_df['weathercode'].value_counts().reset_index()
        wc.columns = ['code','count']
        wc['label'] = wc['code'].map(WMO_CODES).fillna(wc['code'].astype(str))
        fig_c = go.Figure(go.Bar(x=wc['label'], y=wc['count'], marker_color='#2d6abf'))
    elif chart_type == "All-City Rainfall Heatmap":
        dfm = df.copy()
        dfm['month'] = dfm['time'].dt.month
        pivot = dfm.groupby(['city','month'])['precipitation_sum'].mean().unstack()
        pivot.columns = MNAMES
        fig_c = go.Figure(go.Heatmap(z=pivot.values, x=MNAMES, y=pivot.index.tolist(), colorscale='Blues'))

    if fig_c:
        fig_c.update_layout(title=f'{chart_type} — {sel_city}', **fig_style)
        st.plotly_chart(fig_c, use_container_width=True)

    if chart_type != "All-City Rainfall Heatmap":
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Rainy Days", f"{city_df['is_rainy'].mean():.1%}")
        s2.metric("Avg Daily Rain", f"{city_df['precipitation_sum'].mean():.2f} mm")
        s3.metric("Avg Temp", f"{city_df['temperature_2m_mean'].mean():.1f}°C")
        s4.metric("Avg Wind Speed", f"{city_df['windspeed_10m_max'].mean():.1f} km/h")
        s5.metric("Total Records", f"{len(city_df):,}")
