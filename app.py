import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="네이버 쇼핑 트렌드 대시보드", layout="wide")

# 데이터 경로 설정
DATA_DIR = "data"

def load_data(file_pattern):
    """파일 패턴에 맞는 최신 CSV 파일을 로드"""
    files = [f for f in os.listdir(DATA_DIR) if file_pattern in f]
    if not files:
        return None
    latest_file = sorted(files)[-1]
    return pd.read_csv(os.path.join(DATA_DIR, latest_file))

def clean_html(text):
    """HTML 태그 제거"""
    if not isinstance(text, str):
        return ""
    return re.sub('<.*?>', '', text)

# 사이드바 구성
st.sidebar.title("🔍 데이터 설정")
keywords = ["딸기", "딸기 빙수", "딸기 케이크", "딸기 라떼"]
selected_keywords = st.sidebar.multiselect("분석할 키워드를 선택하세요", keywords, default=keywords)

st.title("📊 네이버 쇼핑 트렌드 및 EDA 대시보드")
st.markdown("수집된 네이버 API 데이터를 기반으로 쇼핑 트렌드와 검색 인사이트를 분석합니다.")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📈 쇼핑 트렌드 비교", "📝 블로그 검색 분석", "🛒 쇼핑 검색 분석"])

# --- Tab 1: 쇼핑 트렌드 비교 ---
with tab1:
    st.header("키워드별 쇼핑 클릭 트렌드")
    
    df_trend = load_data("딸기_키워드_쇼핑트랜드")
    if df_trend is not None:
        df_trend['period'] = pd.to_datetime(df_trend['period'])
        
        # 1. 시계열 트렌드 그래프 (Plotly)
        fig_trend = px.line(df_trend, x='period', y=selected_keywords, 
                            title="키워드별 클릭 비중 추이 (최근 1년)",
                            labels={'value': '클릭 비중 (Ratio)', 'period': '날짜', 'variable': '키워드'})
        st.plotly_chart(fig_trend, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 2. 요일별 집계 분석
            df_trend['weekday'] = df_trend['period'].dt.day_name()
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            df_weekday = df_trend.groupby('weekday')[selected_keywords].mean().reindex(weekday_order)
            
            fig_weekday = px.bar(df_weekday, barmode='group', 
                                 title="요일별 평균 클릭 비중",
                                 labels={'value': '평균 클릭 비중', 'weekday': '요일', 'variable': '키워드'})
            st.plotly_chart(fig_weekday, use_container_width=True)
            
        with col2:
            # 3. 트렌드 기술 통계표
            st.subheader("트렌드 요약 통계량")
            st.table(df_trend[selected_keywords].describe())
            
        # 4. 요일별 통계표
        st.subheader("요일별 상세 통계")
        st.dataframe(df_weekday)
    else:
        st.error("쇼핑 트렌드 데이터를 찾을 수 없습니다.")

# --- Tab 2: 블로그 검색 분석 ---
with tab2:
    st.header("블로그 검색 인사이트")
    
    for kw in selected_keywords:
        st.subheader(f"'{kw}' 블로그 분석")
        df_blog = load_data(f"{kw}_블로그")
        
        if df_blog is not None:
            col_b1, col_b2 = st.columns([2, 1])
            
            with col_b1:
                # 5. TF-IDF 키워드 분석
                df_blog['clean_description'] = df_blog['description'].apply(clean_html)
                vectorizer = TfidfVectorizer(max_features=20)
                tfidf_matrix = vectorizer.fit_transform(df_blog['clean_description'])
                feature_names = vectorizer.get_feature_names_out()
                sums = tfidf_matrix.sum(axis=0)
                
                kw_df = pd.DataFrame({'keyword': feature_names, 'score': sums.tolist()[0]})
                kw_df = kw_df.sort_values(by='score', ascending=False)
                
                fig_tfidf = px.bar(kw_df, x='score', y='keyword', orientation='h',
                                   title=f"'{kw}' 핵심 키워드 가중치 (TF-IDF)",
                                   color='score', color_continuous_scale='Viridis')
                fig_tfidf.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_tfidf, use_container_width=True)
            
            with col_b2:
                # 6. 키워드 가중치 표
                st.subheader("핵심 키워드 Top 20")
                st.dataframe(kw_df, height=400)
            
            # 7. 블로그 게시물 리스트
            st.subheader(f"'{kw}' 최신 블로그 게시물")
            st.dataframe(df_blog[['title', 'postdate', 'link']].head(10))
        else:
            st.warning(f"'{kw}' 블로그 데이터를 찾을 수 없습니다.")

# --- Tab 3: 쇼핑 검색 분석 ---
with tab3:
    st.header("쇼핑 검색 시장 분석")
    
    for kw in selected_keywords:
        st.subheader(f"'{kw}' 쇼핑 데이터 분석")
        df_shop = load_data(f"{kw}_쇼핑")
        
        if df_shop is not None:
            df_shop['lprice'] = pd.to_numeric(df_shop['lprice'], errors='coerce')
            df_shop = df_shop.dropna(subset=['lprice'])
            
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                # 8. 가격 분포 히스토그램
                fig_price = px.histogram(df_shop, x='lprice', nbins=20, 
                                         title=f"'{kw}' 상품 가격 분포",
                                         labels={'lprice': '최저가 (원)', 'count': '빈도'},
                                         color_discrete_sequence=['salmon'])
                st.plotly_chart(fig_price, use_container_width=True)
                
            with col_s2:
                # 9. 브랜드 점유율 파이 차트
                brand_counts = df_shop['brand'].value_counts().head(10).reset_index()
                brand_counts.columns = ['brand', 'count']
                fig_brand = px.pie(brand_counts, values='count', names='brand', 
                                   title=f"'{kw}' 상위 노출 브랜드 비중")
                st.plotly_chart(fig_brand, use_container_width=True)
            
            col_s3, col_s4 = st.columns([1, 2])
            with col_s3:
                # 10. 가격 기술 통계표
                st.subheader("가격 요약 통계")
                st.table(df_shop['lprice'].describe())
            
            with col_s4:
                # 11. 쇼핑 상품 목록 표
                st.subheader(f"'{kw}' 상위 상품 목록")
                st.dataframe(df_shop[['title', 'lprice', 'brand', 'mallName']].head(10))
        else:
            st.warning(f"'{kw}' 쇼핑 데이터를 찾을 수 없습니다.")

st.sidebar.markdown("---")
st.sidebar.info("이 대시보드는 네이버 오픈 API 데이터를 사용하여 자동 생성되었습니다.")
