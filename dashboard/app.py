"""
Streamlit Dashboard - Bank Customer Churn Prediction
3 ana bölüm: Prediction, Model Performance, Analytics
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# API URL
API_URL = os.getenv('API_URL', 'http://localhost:8000')

# Sayfa yapılandırması
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("🏦 Churn Prediction Dashboard")
page = st.sidebar.selectbox(
    "Sayfa Seçin",
    ["Prediction", "Model Performance", "Analytics"]
)


def predictionPage():
    """Tahmin sayfası"""
    st.title("🔮 Müşteri Churn Tahmini")
    
    st.markdown("---")
    
    # Form
    col1, col2 = st.columns(2)
    
    with col1:
        creditScore = st.number_input(
            "Kredi Skoru",
            min_value=0,
            max_value=850,
            value=619,
            help="0-850 arası"
        )
        
        age = st.number_input(
            "Yaş",
            min_value=18,
            max_value=100,
            value=42
        )
        
        tenure = st.number_input(
            "Banka ile Yıl Sayısı",
            min_value=0,
            max_value=10,
            value=2
        )
        
        balance = st.number_input(
            "Hesap Bakiyesi",
            min_value=0.0,
            value=0.0,
            format="%.2f"
        )
        
        numOfProducts = st.number_input(
            "Ürün Sayısı",
            min_value=1,
            max_value=4,
            value=1
        )
    
    with col2:
        hasCrCard = st.selectbox(
            "Kredi Kartı",
            [0, 1],
            format_func=lambda x: "Var" if x == 1 else "Yok"
        )
        
        isActiveMember = st.selectbox(
            "Aktif Üye",
            [0, 1],
            format_func=lambda x: "Evet" if x == 1 else "Hayır"
        )
        
        estimatedSalary = st.number_input(
            "Tahmini Maaş",
            min_value=0.0,
            value=101348.88,
            format="%.2f"
        )
        
        geography = st.selectbox(
            "Coğrafya",
            ["France", "Spain", "Germany"]
        )
        
        gender = st.selectbox(
            "Cinsiyet",
            ["Male", "Female"]
        )
    
    # Tahmin butonu
    if st.button("🔍 Tahmin Yap", type="primary", use_container_width=True):
        # API'ye istek
        inputData = {
            "creditScore": creditScore,
            "age": age,
            "tenure": tenure,
            "balance": balance,
            "numOfProducts": numOfProducts,
            "hasCrCard": hasCrCard,
            "isActiveMember": isActiveMember,
            "estimatedSalary": estimatedSalary,
            "geography": geography,
            "gender": gender
        }
        
        try:
            # Tahmin isteği
            response = requests.post(f"{API_URL}/predict", json=inputData)
            response.raise_for_status()
            result = response.json()
            
            # Sonuç gösterimi
            st.markdown("---")
            st.subheader("📊 Tahmin Sonuçları")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Tahmin",
                    "Ayrılacak" if result['prediction'] == 1 else "Ayrılmayacak",
                    delta=None
                )
            
            with col2:
                probability = result['probability']
                st.metric(
                    "Ayrılma Olasılığı",
                    f"{probability:.2%}",
                    delta=None
                )
            
            with col3:
                riskScore = result['riskScore']
                riskColor = {
                    "LOW": "🟢",
                    "MEDIUM": "🟡",
                    "HIGH": "🔴"
                }
                st.metric(
                    "Risk Seviyesi",
                    f"{riskColor.get(riskScore, '⚪')} {riskScore}",
                    delta=None
                )
            
            # Olasılık bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Ayrılmayacak', 'Ayrılacak'],
                y=[1 - probability, probability],
                marker_color=['green', 'red'],
                text=[f"{(1-probability):.2%}", f"{probability:.2%}"],
                textposition='auto'
            ))
            fig.update_layout(
                title="Tahmin Olasılıkları",
                xaxis_title="Sonuç",
                yaxis_title="Olasılık",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # SHAP Explanation (opsiyonel)
            try:
                explainResponse = requests.post(f"{API_URL}/explain", json=inputData)
                if explainResponse.status_code == 200:
                    explanation = explainResponse.json()
                    
                    st.markdown("---")
                    st.subheader("🔍 SHAP Açıklaması")
                    
                    # SHAP values bar chart
                    shapValues = explanation['shap_values']
                    shapDf = pd.DataFrame(
                        list(shapValues.items()),
                        columns=['Feature', 'SHAP Value']
                    )
                    shapDf = shapDf.sort_values('SHAP Value', key=abs, ascending=False)
                    
                    fig = px.bar(
                        shapDf,
                        x='SHAP Value',
                        y='Feature',
                        orientation='h',
                        color='SHAP Value',
                        color_continuous_scale='RdBu',
                        title="Feature Importance (SHAP Values)"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"SHAP açıklaması alınamadı: {e}")
        
        except requests.exceptions.RequestException as e:
            st.error(f"API hatası: {e}")
            st.info("API'nin çalıştığından emin olun: `uvicorn api.main:app --reload`")


def modelPerformancePage():
    """Model performans sayfası"""
    st.title("📈 Model Performansı")
    
    st.markdown("---")
    
    try:
        # Tüm metrikleri getir (ROC ve Confusion Matrix dahil)
        response = requests.get(f"{API_URL}/metrics/full")
        response.raise_for_status()
        metrics = response.json()
        
        # Metrikler gösterimi
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
        
        with col5:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        
        # Metrikler tablosu
        st.markdown("---")
        st.subheader("📊 Detaylı Metrikler")
        
        basicMetrics = {
            'accuracy': metrics.get('accuracy'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1_score': metrics.get('f1_score'),
            'roc_auc': metrics.get('roc_auc')
        }
        metricsDf = pd.DataFrame([basicMetrics])
        st.dataframe(metricsDf, use_container_width=True)
        
        # ROC Curve
        if 'y_test' in metrics and 'y_pred_proba' in metrics:
            st.markdown("---")
            st.subheader("📉 ROC Eğrisi")
            
            from sklearn.metrics import roc_curve, auc
            
            y_test = np.array(metrics['y_test'])
            y_pred_proba = np.array(metrics['y_pred_proba'])
            
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.4f})',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', width=2, dash='dash')
            ))
            fig.update_layout(
                title='ROC Eğrisi',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("---")
            st.subheader("📉 ROC Eğrisi")
            st.info("ROC eğrisi için model eğitim sırasında kaydedilen veriler kullanılmalıdır.")
        
        # Confusion Matrix
        if 'confusion_matrix' in metrics:
            st.markdown("---")
            st.subheader("🔲 Confusion Matrix")
            
            cm = np.array(metrics['confusion_matrix'])
            
            # Heatmap ile göster
            fig = px.imshow(
                cm,
                labels=dict(x="Tahmin", y="Gerçek", color="Sayı"),
                x=['Ayrılmayacak', 'Ayrılacak'],
                y=['Ayrılmayacak', 'Ayrılacak'],
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                title='Confusion Matrix',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Sayısal değerler
            st.write("**Confusion Matrix Değerleri:**")
            cmDf = pd.DataFrame(
                cm,
                index=['Gerçek: Ayrılmayacak', 'Gerçek: Ayrılacak'],
                columns=['Tahmin: Ayrılmayacak', 'Tahmin: Ayrılacak']
            )
            st.dataframe(cmDf, use_container_width=True)
            
            # Precision-Recall Curve
            if 'y_test' in metrics and 'y_pred_proba' in metrics:
                st.markdown("---")
                st.subheader("📊 Precision-Recall Eğrisi")
                
                from sklearn.metrics import precision_recall_curve
                
                precision, recall, thresholds = precision_recall_curve(
                    y_test, y_pred_proba
                )
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=recall,
                    y=precision,
                    mode='lines',
                    name='Precision-Recall Curve',
                    line=dict(color='green', width=2)
                ))
                fig.update_layout(
                    title='Precision-Recall Eğrisi',
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("---")
            st.subheader("🔲 Confusion Matrix")
            st.info("Confusion matrix için model eğitim sırasında kaydedilen veriler kullanılmalıdır.")
        
    except requests.exceptions.RequestException as e:
        st.error(f"API hatası: {e}")
        st.info("API'nin çalıştığından ve model eğitiminin yapıldığından emin olun.")


def analyticsPage():
    """Analitik sayfası"""
    st.title("📊 Analitik Dashboard")
    
    st.markdown("---")
    
    # Feature Importance
    st.subheader("📊 Feature Importance (SHAP)")
    
    # Feature isimlerini Türkçe'ye çevir
    feature_name_mapping = {
        'CreditScore': 'Kredi Skoru',
        'Age': 'Yaş',
        'Tenure': 'Müşteri Süresi',
        'Balance': 'Bakiye',
        'NumOfProducts': 'Ürün Sayısı',
        'HasCrCard': 'Kredi Kartı Var',
        'IsActiveMember': 'Aktif Üye',
        'EstimatedSalary': 'Tahmini Maaş',
        'Geography_Germany': 'Coğrafya: Almanya',
        'Geography_Spain': 'Coğrafya: İspanya',
        'Gender_Male': 'Cinsiyet: Erkek'
    }
    
    try:
        response = requests.get(f"{API_URL}/feature-importance", timeout=5)
        if response.status_code == 200:
            data = response.json()
            featureImportance = data.get('feature_importance', {})
            
            if featureImportance:
                # Feature isimlerini Türkçe'ye çevir
                translated_importance = {}
                for feature, importance in featureImportance.items():
                    # Feature ismini çevir (eğer mapping'de yoksa orijinal ismi kullan)
                    translated_name = feature_name_mapping.get(feature, feature)
                    translated_importance[translated_name] = importance
                
                # DataFrame'e çevir
                df = pd.DataFrame(
                    list(translated_importance.items()),
                    columns=['Feature', 'Importance']
                )
                df = df.sort_values('Importance', ascending=True)
                
                # Horizontal bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df['Importance'],
                    y=df['Feature'],
                    orientation='h',
                    marker=dict(
                        color=df['Importance'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Importance")
                    ),
                    text=[f'{val:.4f}' for val in df['Importance']],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
                ))
                fig.update_layout(
                    title='Feature Importance (SHAP Değerleri)',
                    xaxis_title='SHAP Importance (Ortalama Mutlak Değer)',
                    yaxis_title='Özellik',
                    height=max(400, len(df) * 30),
                    hovermode='y',
                    margin=dict(l=150)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Tablo olarak da göster
                st.markdown("---")
                st.subheader("📋 Feature Importance Tablosu")
                df_display = df.copy()
                df_display['Importance'] = df_display['Importance'].apply(lambda x: f'{x:.6f}')
                df_display = df_display.sort_values('Importance', ascending=False, key=lambda x: pd.to_numeric(x))
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                
                # İstatistikler
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Toplam Özellik", len(df))
                with col2:
                    st.metric("En Yüksek Importance", f"{df['Importance'].max():.4f}")
                with col3:
                    st.metric("Ortalama Importance", f"{df['Importance'].mean():.4f}")
            else:
                st.warning("Feature importance verisi bulunamadı.")
        elif response.status_code == 404:
            st.warning("Feature importance bulunamadı. Model eğitimi yapıldığından ve SHAP feature importance hesaplandığından emin olun.")
            st.info("Model eğitimi için: `python src/train.py` komutunu çalıştırın.")
        else:
            st.error(f"API hatası: {response.status_code}")
            st.info("API'nin çalıştığından emin olun (http://localhost:8000)")
    except requests.exceptions.RequestException as e:
        st.error(f"API bağlantı hatası: {e}")
        st.info("API'nin çalıştığından emin olun (http://localhost:8000)")
    
    st.markdown("---")
    
    # Risk Dağılımı
    st.subheader("📈 Risk Dağılımı")
    st.info("Bu sayfa veritabanından geçmiş tahminleri gösterir. Veritabanı entegrasyonu tamamlandığında aktif olacaktır.")
    
    # Placeholder chart
    riskData = pd.DataFrame({
        'Risk': ['LOW', 'MEDIUM', 'HIGH'],
        'Count': [65, 25, 10]
    })
    
    fig = px.pie(
        riskData,
        values='Count',
        names='Risk',
        title="Risk Seviyesi Dağılımı"
    )
    st.plotly_chart(fig, use_container_width=True)


# Ana uygulama
def main():
    if page == "Prediction":
        predictionPage()
    elif page == "Model Performance":
        modelPerformancePage()
    elif page == "Analytics":
        analyticsPage()


if __name__ == "__main__":
    main()
