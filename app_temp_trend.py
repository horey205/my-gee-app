import streamlit as st
import ee
import folium
from streamlit_folium import folium_static
import pandas as pd
import plotly.express as px
import json

# 페이지 설정
st.set_page_config(layout="wide", page_title="🌡️ 한국 기온 변화 탐사선")

# CSS로 스타일링
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTitle { color: #e63946; font-family: 'Suit', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌡️ 한국은 얼마나 뜨거워졌을까? (1980 - 2023)")
st.sidebar.markdown("### 🛰️ 분석 가이드")
st.sidebar.info(
    "지도의 색상은 1980년대 대비 최근 10년의 기온 상승 폭을 나타냅니다.\n"
    "붉은색이 진할수록 기온이 많이 오른 지역입니다."
)

# GEE 초기화 (서버/로컬용 통합 로직)
@st.cache_resource
def init_ee():
    try:
        # 기본 프로젝트 ID
        project_id = 'basic-perigee-384507'
        
        # 1. 단일 JSON 뭉치 방식 (가장 권장: GEE_JSON_KEY 하나에 파일 내용 전체 입력)
        if "GEE_JSON_KEY" in st.secrets:
            key_data = st.secrets["GEE_JSON_KEY"]
            
            # 데이터 타입 판별 및 로드
            if isinstance(key_data, str):
                key_dict = json.loads(key_data)
            else:
                key_dict = key_data
            
            # **핵심 수정**: key_data에는 JSON 전체가 아닌 'private_key' 문자열만 들어가야 함
            email = key_dict['client_email']
            p_key = key_dict['private_key'].replace('\\n', '\n').strip()
            p_id = key_dict.get('project_id', project_id)
            
            credentials = ee.ServiceAccountCredentials(email, key_data=p_key)
            ee.Initialize(credentials, project=p_id)
            
        # 2. 개별 필드 방식 (하위 호환성)
        elif "private_key" in st.secrets:
            email = st.secrets["client_email"]
            p_id = st.secrets.get("project_id", project_id)
            p_key = st.secrets["private_key"].replace('\\n', '\n').strip()
            
            credentials = ee.ServiceAccountCredentials(email, key_data=p_key)
            ee.Initialize(credentials, project=p_id)
            
        # 3. 로컬 테스트용
        else:
            ee.Initialize(project=project_id)
            if not hasattr(ee.data, '_credentials'):
                class MockCredentials: pass
                ee.data._credentials = MockCredentials()
    except Exception as e:
        st.error(f"인증 오류 발생: {e}")
        st.info("💡 해결 방법: Streamlit Secrets에 'GEE_JSON_KEY'라는 이름으로 JSON 파일 내용 전체를 다시 한 번 정확히 넣어주세요.")

init_ee()

# 데이터 처리 함수
def get_temp_diff_map():
    dataset = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
    past = dataset.filterDate('1980-01-01', '1989-12-31').select('temperature_2m').mean()
    current = dataset.filterDate('2014-01-01', '2023-12-31').select('temperature_2m').mean()
    diff = current.subtract(past)
    return diff

def get_timeseries(lon, lat):
    point = ee.Geometry.Point([lon, lat])
    dataset = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
                .filterDate('1980-01-01', '2023-12-31') \
                .select('temperature_2m')
    
    def extract_val(img):
        mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=9000).get('temperature_2m')
        date = img.date().format('YYYY-MM')
        return ee.Feature(None, {'date': date, 'temp': ee.Number(mean).subtract(273.15)})

    features = dataset.map(extract_val).getInfo()['features']
    data = [f['properties'] for f in features]
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df

# 레이아웃
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ 지난 40년간 기온 상승 분포")
    try:
        diff_layer = get_temp_diff_map()
        vis_params = {'min': 0, 'max': 3, 'palette': ['ffffb2', 'fed976', 'feb24c', 'fd8d3c', 'f03b20', 'bd0026']}
        map_id_dict = ee.Image(diff_layer).getMapId(vis_params)
        
        m = folium.Map(location=[36.5, 127.5], zoom_start=7)
        folium.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            name='Temperature Rise',
            overlay=True,
            control=True
        ).add_to(m)
        folium_static(m, height=600)
    except Exception as e:
        st.error(f"지도를 로드할 수 없습니다: {e}")

with col2:
    st.subheader("📈 변화 추이 분석")
    if st.button("서울 지역 기온 변화 불러오기"):
        with st.spinner("데이터를 분석 중입니다..."):
            df = get_timeseries(126.97, 37.56)
            df_annual = df.resample('YE', on='date').mean().reset_index()
            
            fig = px.line(df_annual, x='date', y='temp', 
                         title='서울 연평균 기온 변화 (1980-2023)',
                         labels={'temp': '평균 기온 (°C)', 'date': '연도'},
                         template='plotly_white')
            fig.add_scatter(x=df_annual['date'], y=df_annual['temp'].rolling(window=5).mean(), name='5년 이동평균')
            st.plotly_chart(fig, use_container_width=True)
            
            rise = df_annual.iloc[-1]['temp'] - df_annual.iloc[0]['temp']
            st.metric("총 상승 폭", f"{rise:.2f} °C", delta=f"{rise:.2f} °C", delta_color="inverse")
    else:
        st.info("버튼을 눌러 분석 결과를 확인해보세요.")

st.caption("Data Source: ERA5-Land. Powered by Google Earth Engine.")
