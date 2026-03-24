import streamlit as st
import ee
import folium
from streamlit_folium import folium_static
import pandas as pd
import plotly.express as px
import json
import google.generativeai as genai
import re

# 페이지 설정
st.set_page_config(layout="wide", page_title="🛰️ GEE AI 탐사선")

# CSS로 스타일링
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTitle { color: #1e88e5; font-family: 'Suit', sans-serif; }
    .sidebar-content { padding: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛰️ Earth Engine AI 탐사선")

# 사이드바 설정
st.sidebar.title("⚙️ 설정 및 도구")
gemini_key = st.sidebar.text_input("Gemini API Key를 입력하세요", type="password")
gemini_model = st.sidebar.text_input("Gemini 모델명", value="gemini-3-flash-preview")
mode = st.sidebar.radio("분석 모드 선택", ["기본 분석 (한국 기온)", "GEDI 산림 정밀 분석", "AI 탐사선 (자연어 질문)"])

# GEE 초기화
PROJECT_ID = 'basic-perigee-384507'

@st.cache_resource
def init_ee():
    try:
        key_file = r'd:\AI_Class\Earth Engine\basic-perigee-384507-4673a11d6682.json'
        
        # 1. 로컬의 JSON 키 파일 사용 (우선 순위)
        import os
        if os.path.exists(key_file):
            credentials = ee.ServiceAccountCredentials('', key_file=key_file)
            ee.Initialize(credentials, project=PROJECT_ID)
            return True

        # 2. Streamlit Secrets (배포 환경용)
        if "GEE_JSON_KEY" in st.secrets:
            key_data = st.secrets["GEE_JSON_KEY"]
            
            # 딕셔너리 또는 문자열(JSON) 형태 모두 지원
            if isinstance(key_data, str):
                import json
                key_dict = json.loads(key_data)
            else:
                key_dict = key_data
            
            email = key_dict['client_email']
            p_key = key_dict['private_key']
            
            # 비공개 키 형식 정규화 (줄바꿈 인식 개선)
            if isinstance(p_key, str):
                p_key = p_key.replace('\\n', '\n').strip()
            
            credentials = ee.ServiceAccountCredentials(email, key_data=p_key)
            ee.Initialize(credentials, project=PROJECT_ID)
            return True
        else:
            # 3. 기본 초기화 시도
            ee.Initialize(project=PROJECT_ID)
            return True
    except Exception as e:
        st.error(f"❌ GEE 초기화 오류: {e}")
        st.info(f"""
        **해결 방법:**
        1. [Google Cloud Console](https://console.cloud.google.com/iam-admin/iam?project={PROJECT_ID})에서 서비스 계정에 'Service Usage Consumer' 및 'Earth Engine Resource Viewer' 권한을 추가하세요.
        2. [Earth Engine API](https://console.cloud.google.com/apis/library/earthengine.googleapis.com?project={PROJECT_ID})가 활성화되어 있는지 확인하세요.
        """)
        st.stop()

if not init_ee():
    st.stop()

# -- 1. 기본 분석 모드 (기존 코드) --
if mode == "기본 분석 (한국 기온)":
    st.sidebar.info("1980년대 대비 최근 10년의 한국 기온 변화를 분석합니다.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ 지난 40년간 기온 상승 분포")
        dataset = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        past = dataset.filterDate('1980-01-01', '1989-12-31').select('temperature_2m').mean()
        current = dataset.filterDate('2014-01-01', '2023-12-31').select('temperature_2m').mean()
        diff = current.subtract(past)
        
        vis_params = {'min': 0, 'max': 3, 'palette': ['ffffb2', 'fed976', 'feb24c', 'fd8d3c', 'f03b20', 'bd0026']}
        map_id_dict = ee.data.getMapId({'image': diff, 'visParams': vis_params})
        
        m = folium.Map(location=[36.5, 127.5], zoom_start=7)
        folium.TileLayer(tiles=map_id_dict['tile_fetcher'].url_format, attr='Google Earth Engine', name='Temp Rise', overlay=True).add_to(m)
        folium_static(m, height=600)

    with col2:
        st.subheader("📈 서울 기온 변화 추이")
        if st.button("데이터 불러오기"):
            point = ee.Geometry.Point([126.97, 37.56])
            dataset_ts = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate('1980-01-01', '2023-12-31').select('temperature_2m')
            
            def extract_val(img):
                mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=9000).get('temperature_2m')
                return ee.Feature(None, {'date': img.date().format('YYYY-MM'), 'temp': ee.Number(mean).subtract(273.15)})

            features = dataset_ts.map(extract_val).getInfo()['features']
            df = pd.DataFrame([f['properties'] for f in features])
            df['date'] = pd.to_datetime(df['date'])
            df_annual = df.resample('YE', on='date').mean().reset_index()
            
            fig = px.line(df_annual, x='date', y='temp', title='서울 연평균 기온 변화')
            st.plotly_chart(fig, use_container_width=True)

# -- 2. GEDI 산림 정밀 분석 모드 (신규) --
elif mode == "GEDI 산림 정밀 분석":
    st.sidebar.info("NASA GEDI 레이저 데이터를 활용하여 수관(나무) 높이와 산림 구조를 분석합니다.")
    
    # 분석 지역 설정
    area_options = {
        "광릉수목원 (대한민국)": {"center": [37.75, 127.16], "zoom": 13},
        "설악산 국립공원 (대한민국)": {"center": [38.12, 128.46], "zoom": 12},
        "아마존 열대우림 (브라질)": {"center": [-3.11, -60.02], "zoom": 10},
        "콩고 분지 (아프리카)": {"center": [-0.5, 18.0], "zoom": 8}
    }
    selected_area = st.selectbox("분석 지역 선택", list(area_options.keys()))
    analysis_type = st.radio("분석 데이터 선택", ["수관 상단 높이 (Canopy Height)", "지면 고도 (Ground Elevation)"])
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"🌲 {selected_area} 정밀 분석 지도")
        
        # GEDI 데이터 로드 (L2A Monthly)
        dataset = ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY")
        
        # 최신 데이터 위주로 평균값 계산
        if analysis_type == "수관 상단 높이 (Canopy Height)":
            # selfMask()로 데이터가 없는 지점(0 또는 null)을 투명하게 처리 (검은 배경 제거)
            data_layer = dataset.select('rh98').mean().selfMask()
            
            # 팔레트 개선: 검은색 느낌이 없는 선명하고 밝은 색상 (연두-녹색-노랑-주황-빨강)
            vis_params = {
                'min': 3, # 3m 이하의 지면 노이즈 제외
                'max': 45, 
                'palette': ['#d9f0a3', '#78c679', '#238b45', '#ffff33', '#fe9929', '#e31a1c']
            }
            label = "Canopy Height (m)"
        else:
            # 지면 고도 데이터도 투명화 및 시각화 개선
            data_layer = dataset.select('elev_lowestmode').mean().selfMask()
            vis_params = {
                'min': 0, 'max': 1500, 
                'palette': ['#0000ff', '#00ffff', '#ffff00', '#ff0000', '#ffffff'] # 고도에 따른 색상 시퀀스
            }
            label = "Elevation (m)"

        # 지도 설정
        loc = area_options[selected_area]["center"]
        zoom = area_options[selected_area]["zoom"]
        
        map_id_dict = ee.data.getMapId({'image': data_layer, 'visParams': vis_params})
        m_gedi = folium.Map(location=loc, zoom_start=zoom, tiles="cartodbpositron")
        folium.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format, 
            attr='NASA GEDI / Google Earth Engine', 
            name=label, 
            overlay=True
        ).add_to(m_gedi)
        
        # 범례 추가 (V2.1 대응)
        folium.LayerControl().add_to(m_gedi)
        folium_static(m_gedi, height=650)

    with col2:
        st.subheader("📊 정밀 분석 지표")
        st.write(f"**현재 위치:** {selected_area}")
        st.write(f"**데이터:** NASA GEDI (Latest)")
        st.divider()

        # 1. 중심점 기준 수치 추출 및 대시보드 표시
        try:
            with st.spinner("수치 데이터를 계산 중..."):
                point = ee.Geometry.Point([loc[1], loc[0]])
                region = point.buffer(2000).bounds() # 2km 영역
                
                # 해당 영역의 평균값 계산
                stat = data_layer.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=1000,
                    maxPixels=1e9
                )
                
                # rh98 또는 elev_lowestmode 키 동적으로 가져오기
                keys = stat.keys().getInfo()
                if keys:
                    val = stat.get(keys[0]).getInfo()
                    if val is not None:
                        st.metric(label=f"주변 {analysis_type} 평균", value=f"{val:.2f} m")
                    else:
                        st.warning("이 지역의 최근 유효 데이터가 부족합니다.")
                else:
                    st.warning("데이터 로딩 중...")
        except Exception as e:
            st.error(f"통계 추출 오류: {e}")

        st.divider()

        # 2. 색상 범례(Legend) 추가
        st.markdown("**🎨 색상 범례 (m)**")
        if analysis_type == "수관 상단 높이 (Canopy Height)":
            legend_html = """
            <div style="font-size: 13px; font-weight: bold;">
                <span style="background:#bd0026; padding: 2px 10px; color:white;"></span> 40m+ (매우 높은 숲)<br>
                <span style="background:#fd8d3c; padding: 2px 10px;"></span> 30~40m<br>
                <span style="background:#fed976; padding: 2px 10px;"></span> 20~30m (일반 숲)<br>
                <span style="background:#ffffcc; padding: 2px 10px;"></span> 10~20m (낮은 숲)<br>
                <span style="background:#74c476; padding: 2px 10px;"></span> 0~10m (관목/지면)<br>
                <span style="background:#f7fcf5; border:1px solid gray; padding: 2px 10px;"></span> 0m (데이터 없음)
            </div>
            """
            st.markdown(legend_html, unsafe_allow_html=True)
            st.info("💡 **RH98:** 상위 98% 지점의 높이(수관 키)")
        else:
            st.info("💡 **Elevation:** 지면의 실제 해발 고도입니다.")

        st.divider()
        st.caption("참고: GEDI 데이터는 위성 궤도 데이터이므로 지점 간 1km 격자 데이터가 없는 구역은 빈 칸으로 나타날 수 있습니다.")

# -- 3. AI 탐사선 모드 --
else:
    st.subheader("🤖 무엇이든 물어보세요 (GEE AI Agent)")
    query = st.text_area("분석하고 싶은 내용을 입력하세요", placeholder="예: 지난 5년간 아마존의 산불 발생 지역을 보여줘", height=100)
    
    if st.button("AI 분석 시작"):
        if not gemini_key:
            st.warning("먼저 사이드바에 Gemini API Key를 입력해주세요.")
        elif not query:
            st.warning("분석 질문을 입력해주세요.")
        else:
            with st.spinner("AI가 GEE 코드를 생성하고 데이터를 분석 중입니다..."):
                try:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel(gemini_model)
                    
                    prompt = f"""
                    You are a Google Earth Engine expert. Write only valid Python code for the following request.
                    
                    [Requirements]
                    1. DO NOT call ee.Initialize() or any imports (ee, pd, px are already provided).
                    2. Store the final ee.Image or ee.ImageCollection in a variable named 'result'.
                    3. Define 'vis_params' as a dictionary with min, max, and palette.
                    4. Define 'center' as [lat, lon].
                    5. Define 'description' as a short Korean string describing the analysis.
                    
                    [User Request]: {query}
                    
                    Output only the Python code itself.
                    """
                    
                    response = model.generate_content(prompt)
                    code = response.text
                    
                    # 코드 정제
                    code = re.sub(r'```(?:python)?', '', code).replace('```', '').strip()
                    lines = [line for line in code.split('\n') if not line.startswith('설명:')]
                    code = '\n'.join(lines)
                    
                    # 코드 실행 환경 준비
                    exec_globals = {"ee": ee, "pd": pd, "px": px}
                    try:
                        exec(code, exec_globals)
                    except Exception as exec_e:
                        st.error(f"코드 실행 중 문법 오류 발생: {exec_e}")
                        with st.expander("AI가 생성한 원본 코드 보기"):
                            st.code(code, language='python')
                        st.stop()
                    
                    result = exec_globals.get("result")
                    vis_params = exec_globals.get("vis_params")
                    center = exec_globals.get("center", [36.5, 127.5])
                    description = exec_globals.get("description", "분석이 완료되었습니다.")
                    
                    st.success("✅ 분석 완료!")
                    st.write(f"ℹ️ {description}")
                    
                    # 지도 렌더링
                    if isinstance(result, ee.ImageCollection):
                        result = result.mean()
                    
                    map_id = ee.data.getMapId({'image': result, 'visParams': vis_params})
                    m_ai = folium.Map(location=center, zoom_start=6)
                    folium.TileLayer(tiles=map_id['tile_fetcher'].url_format, attr='Google Earth Engine', name='AI Analysis', overlay=True).add_to(m_ai)
                    folium_static(m_ai, height=600)
                    
                    with st.expander("실행된 AI 코드 보기"):
                        st.code(code, language='python')
                        
                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다: {e}")

st.caption("Powered by Google Earth Engine & Gemini AI")
