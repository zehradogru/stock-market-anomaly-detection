import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import dtw
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
# from statsmodels.tsa.arima.model import ARIMA # ARIMA şimdilik yorum satırı olarak bırakıldı.
import warnings

# Uyarıları gizle
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None # Chained assignment uyarılarını gizle

# 1. PARAMETRELER
# -----------------------------------------------------------------------------
# Daha yönetilebilir bir hisse senedi listesi (test için, istediğiniz gibi genişletin)
tickers_list = [
   'AEFES.IS', 'AKBNK.IS', 'AKSA.IS', 'ALARK.IS', 'ARCLK.IS',
   'ASELS.IS', 'BIMAS.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS',
   'KCHOL.IS', 'KOZAL.IS', 'KRDMD.IS', 'PETKM.IS', 'PGSUS.IS',
   'SAHOL.IS', 'SASA.IS', 'SISE.IS', 'TCELL.IS', 'THYAO.IS',
   'TOASO.IS', 'TUPRS.IS', 'TTKOM.IS', 'ULKER.IS', 'YKBNK.IS'
]

start_date = '2022-01-01'
end_date = '2024-01-01'   # Veya güncel bir tarih
num_clusters_to_find = 5  # Raporunuza göre ayarlayın, hisse sayısından fazla olamaz.

# Monte Carlo ve Alım Sinyali Parametreleri
SIMULATION_DAYS = 20       # Gelecekte kaç işlem günü simüle edilecek
NUM_SIMULATIONS = 3000     # Her anomali için simülasyon sayısı (daha yüksek = daha doğru ama yavaş)
HISTORICAL_DAYS_FOR_PARAMS = 60 # Drift ve volatilite tahmini için geçmiş gün sayısı

# Alım Sinyali Kriterleri (Raporunuza göre ayarlayın!)
MIN_RECOVERY_PROB = 0.60
MIN_EXPECTED_RETURN = 0.01
MIN_RETURN_CVAR_RATIO = 0.3
CVAR_ALPHA = 0.05
# -----------------------------------------------------------------------------

# 2. VERİ ÇEKME VE HAZIRLAMA
# -----------------------------------------------------------------------------
print("Hisse senedi verileri indiriliyor...")
try:
    # auto_adjust=False kullanarak 'Adj Close'u manuel alıyoruz, çünkü bazen yfinance'in
    # auto_adjust'ı beklenmedik sonuçlar verebilir veya sütun adlarını değiştirebilir.
    raw_data = yf.download(tickers_list, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if raw_data.empty:
        raise ValueError("İndirilen veri boş.")
    adj_close_prices = raw_data['Adj Close'].copy()
except Exception as e:
    print(f"Veri indirme hatası: {e}")
    exit()

# Veri temizleme
adj_close_prices = adj_close_prices.dropna(axis=1, how='all') # Tamamen NaN olan sütunları kaldır
# Belirli bir orandan (%90) daha az verisi olan sütunları at
adj_close_prices = adj_close_prices.dropna(axis=1, thresh=int(len(adj_close_prices) * 0.9))
adj_close_prices = adj_close_prices.fillna(method='ffill').fillna(method='bfill') # Kalan boşlukları doldur

if adj_close_prices.empty or adj_close_prices.shape[1] < 2:
    print(f"Yeterli sayıda ({adj_close_prices.shape[1]}) geçerli hisse senedi verisi bulunamadı (en az 2 gerekli).")
    exit()

# Küme sayısını, geçerli hisse senedi sayısına göre ayarla (gerekirse)
if adj_close_prices.shape[1] < num_clusters_to_find:
    print(f"Uyarı: Bulunan hisse sayısı ({adj_close_prices.shape[1]}) istenen küme sayısından ({num_clusters_to_find}) az. Küme sayısı hisse sayısına düşürüldü.")
    num_clusters_to_find = adj_close_prices.shape[1]
if num_clusters_to_find <=1 and adj_close_prices.shape[1] > 1 : num_clusters_to_find = 2


print(f"{adj_close_prices.shape[1]} adet hisse senedi için veriler başarıyla işlendi.")
valid_tickers = adj_close_prices.columns.tolist()

# Logaritmik Getirileri Hesapla (Anomali tespiti ve DTW için temel)
log_returns = np.log(adj_close_prices / adj_close_prices.shift(1))
log_returns.replace([np.inf, -np.inf], np.nan, inplace=True) # Sonsuz değerleri NaN yap
log_returns = log_returns.dropna(axis=0, how='all') # Tamamen NaN olan satırları kaldır
log_returns = log_returns.dropna(axis=1, how='all') # Tamamen NaN olan sütunları kaldır (hisse gitmişse)
log_returns = log_returns.fillna(0) # Kalan NaN'ları 0 ile doldur (veya ffill/bfill) - ilk gün getirisi için

# Log getirileri sonrası veri tutarlılığını sağla
valid_tickers = log_returns.columns.tolist() # Log getirisi olan geçerli hisseler
adj_close_prices = adj_close_prices.loc[log_returns.index, valid_tickers] # İndeksleri ve sütunları eşitle

if log_returns.empty or log_returns.shape[1] < 2:
    print("Logaritmik getiri hesaplaması sonrası yeterli veri veya hisse kalmadı.")
    exit()
if log_returns.shape[1] < num_clusters_to_find:
     num_clusters_to_find = log_returns.shape[1]
if num_clusters_to_find <=1 and log_returns.shape[1] > 1 : num_clusters_to_find = 2


# Kümeleme için log getirilerini normalleştir
scaler_clustering = StandardScaler()
# Transpoze etmeden önce NaN kontrolü
if log_returns.isnull().values.any():
    print("Uyarı: Kümeleme için normalleştirilecek log getirilerinde NaN değerler var. 0 ile dolduruluyor.")
    log_returns_for_scaling = log_returns.fillna(0)
else:
    log_returns_for_scaling = log_returns

normalized_log_returns_for_clustering_scaled = pd.DataFrame(
    scaler_clustering.fit_transform(log_returns_for_scaling), # Her bir hissenin zaman serisini normalleştir
    columns=valid_tickers,
    index=log_returns.index
)
# -----------------------------------------------------------------------------

# 3. DTW MESAFE MATRİSİNİ HESAPLA
# -----------------------------------------------------------------------------
print("DTW mesafe matrisi hesaplanıyor... Bu işlem hisse sayısına bağlı olarak sürebilir.")
n_series_clustering = normalized_log_returns_for_clustering_scaled.shape[1]
dtw_distance_matrix = np.zeros((n_series_clustering, n_series_clustering))

# DTW için pencere boyutu (hesaplama süresini azaltabilir ve ilgisiz eşleşmeleri önleyebilir)
dtw_window_size = int(len(normalized_log_returns_for_clustering_scaled) * 0.1) # Zaman serisi uzunluğunun %10'u gibi

for i in range(n_series_clustering):
    for j in range(i + 1, n_series_clustering):
        series1 = normalized_log_returns_for_clustering_scaled.iloc[:, i].values.reshape(-1, 1)
        series2 = normalized_log_returns_for_clustering_scaled.iloc[:, j].values.reshape(-1, 1)
        # tslearn.metrics.dtw, default olarak Euclidean mesafesi kullanır
        distance = dtw(series1, series2, sakoe_chiba_radius=dtw_window_size // 2)
        dtw_distance_matrix[i, j] = distance
        dtw_distance_matrix[j, i] = distance # Matris simetriktir
# -----------------------------------------------------------------------------

# 4. HİYERARŞİK KÜMELEME
# -----------------------------------------------------------------------------
print("Hiyerarşik kümeleme yapılıyor...")
if n_series_clustering <= 1: # Kümeleme için yeterli hisse yoksa
    print("Kümeleme için yeterli sayıda (en az 2) hisse senedi bulunmuyor.")
    stock_clusters = {ticker: 0 for ticker in valid_tickers} # Tümünü tek kümeye ata
    num_clusters_to_find = 1
else:
    # `metric`='precomputed' (veya eski scikit-learn'de `affinity`) kullanıldığında,
    # `linkage`='ward' hata verir. 'average', 'complete' veya 'single' kullanılabilir.
    hierarchical_cluster_model = AgglomerativeClustering(
        n_clusters=num_clusters_to_find,
        metric='precomputed', # `metric` argümanını kullanın
        linkage='average'     # 'ward' yerine 'average' veya 'complete'
    )
    try:
        cluster_labels = hierarchical_cluster_model.fit_predict(dtw_distance_matrix)
        stock_clusters = {ticker: label for ticker, label in zip(valid_tickers, cluster_labels)}
    except ValueError as ve:
        print(f"Kümeleme hatası: {ve}. Muhtemelen n_clusters > n_samples. Tümünü tek kümeye atıyorum.")
        stock_clusters = {ticker: 0 for ticker in valid_tickers}
        num_clusters_to_find = 1


print("\nKümeleme Sonuçları:")
for i in range(num_clusters_to_find):
    cluster_members = [ticker for ticker, label in stock_clusters.items() if label == i]
    if cluster_members:
        print(f"Küme {i}: {', '.join(cluster_members)}")
    else:
        print(f"Küme {i}: (Boş)")
# -----------------------------------------------------------------------------

# 5. ANOMALİ TESPİT FONKSİYONLARI VE HİBRİT YAKLAŞIM
# -----------------------------------------------------------------------------
print("\nAnomali tespiti yapılıyor...")

def get_cluster_z_score_anomalies(log_returns_df, stock_ticker, cluster_members_tickers, window=20, threshold=-2.5):
    if stock_ticker not in log_returns_df.columns: return pd.Series(False, index=log_returns_df.index)
    stock_series = log_returns_df[stock_ticker]
    peer_tickers = [p for p in cluster_members_tickers if p != stock_ticker and p in log_returns_df.columns]
    if not peer_tickers: return pd.Series(False, index=log_returns_df.index)
    peer_returns = log_returns_df[peer_tickers]
    # Rolling window'u ortalama ve std hesaplamalarına ekle
    cluster_mean_rolling = peer_returns.mean(axis=1).rolling(window=window, min_periods=1).mean()
    cluster_std_rolling = peer_returns.std(axis=1).rolling(window=window, min_periods=1).mean() # std'nin de ort.
    z_scores = (stock_series - cluster_mean_rolling) / cluster_std_rolling.replace(0, np.nan)
    anomalies = z_scores < threshold
    return anomalies.fillna(False)

def get_isolation_forest_anomalies(log_returns_series, contamination=0.02): # Daha düşük kontaminasyon
    if log_returns_series.dropna().empty or len(log_returns_series.dropna()) < 2:
        return pd.Series(False, index=log_returns_series.index)
    model = IsolationForest(contamination=contamination, random_state=42, bootstrap=False) # bootstrap=False öneriliyor
    # Tek boyutlu veri için reshape ve NaN'ları atla
    valid_data = log_returns_series.dropna()
    if valid_data.empty: return pd.Series(False, index=log_returns_series.index)
    predictions = model.fit_predict(valid_data.values.reshape(-1, 1))
    anomalies_on_dropna = pd.Series(predictions == -1, index=valid_data.index)
    return anomalies_on_dropna.reindex(log_returns_series.index, fill_value=False)

def get_one_class_svm_anomalies(log_returns_series, nu=0.02, kernel="rbf", gamma='scale'): # gamma='scale' daha genel
    if log_returns_series.dropna().empty or len(log_returns_series.dropna()) < 2:
        return pd.Series(False, index=log_returns_series.index)
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    valid_data = log_returns_series.dropna()
    if valid_data.empty: return pd.Series(False, index=log_returns_series.index)
    predictions = model.fit_predict(valid_data.values.reshape(-1, 1))
    anomalies_on_dropna = pd.Series(predictions == -1, index=valid_data.index)
    return anomalies_on_dropna.reindex(log_returns_series.index, fill_value=False)

# Tüm hisseler için anomali skorlarını sakla
all_anomalies = pd.DataFrame(index=log_returns.index)

for cluster_id in range(num_clusters_to_find):
    current_cluster_members = [ticker for ticker, label in stock_clusters.items() if label == cluster_id]
    if not current_cluster_members: continue
    for stock_ticker in current_cluster_members:
        if stock_ticker not in log_returns.columns: continue
        stock_log_returns = log_returns[stock_ticker]
        # Z-Score
        z_anomalies = get_cluster_z_score_anomalies(log_returns, stock_ticker, current_cluster_members)
        all_anomalies[f'{stock_ticker}_z_score'] = z_anomalies
        # Isolation Forest
        if_anomalies = get_isolation_forest_anomalies(stock_log_returns)
        all_anomalies[f'{stock_ticker}_if'] = if_anomalies
        # One-Class SVM
        ocsvm_anomalies = get_one_class_svm_anomalies(stock_log_returns)
        all_anomalies[f'{stock_ticker}_ocsvm'] = ocsvm_anomalies
        # Hibrit Anomali (en az 2 yöntem TRUE ve negatif getiri)
        anomaly_columns = [f'{stock_ticker}_z_score', f'{stock_ticker}_if', f'{stock_ticker}_ocsvm']
        is_negative_return = stock_log_returns < 0
        valid_anomaly_cols = [col for col in anomaly_columns if col in all_anomalies.columns]
        if len(valid_anomaly_cols) > 0:
            hybrid_signal_count = all_anomalies[valid_anomaly_cols].sum(axis=1)
            all_anomalies[f'{stock_ticker}_hybrid'] = (hybrid_signal_count >= 2) & is_negative_return
        else:
            all_anomalies[f'{stock_ticker}_hybrid'] = False
# -----------------------------------------------------------------------------

# 6. MONTE CARLO SİMÜLASYONU VE ALIM SİNYALİ ÜRETİMİ
# -----------------------------------------------------------------------------
print("\nMonte Carlo Simülasyonu ve Alım Sinyali Değerlendirmesi Başlıyor...")

def monte_carlo_simulation_gbm(start_price, drift, volatility, days_to_simulate, num_simulations):
    dt = 1
    price_paths = np.zeros((days_to_simulate + 1, num_simulations))
    price_paths[0] = start_price
    for t in range(1, days_to_simulate + 1):
        Z = np.random.standard_normal(num_simulations)
        price_paths[t] = price_paths[t - 1] * np.exp(
            (drift - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * Z
        )
    return price_paths

buy_signals_mc = pd.DataFrame(False, index=all_anomalies.index, columns=valid_tickers)

for stock_ticker in valid_tickers:
    if f'{stock_ticker}_hybrid' not in all_anomalies.columns: continue
    hybrid_anomalies_for_stock = all_anomalies[f'{stock_ticker}_hybrid']
    anomaly_dates_indices = hybrid_anomalies_for_stock[hybrid_anomalies_for_stock].index

    for anomaly_date in anomaly_dates_indices:
        if anomaly_date not in adj_close_prices.index: continue # Fiyat verisi yoksa atla

        current_price = adj_close_prices.loc[anomaly_date, stock_ticker]
        anomaly_date_loc = adj_close_prices.index.get_loc(anomaly_date)
        if anomaly_date_loc == 0: continue # İlk gün anomaliyse atla
        target_price_recovery = adj_close_prices.iloc[anomaly_date_loc - 1][stock_ticker]

        historical_data_end_idx = log_returns.index.get_loc(anomaly_date)
        historical_data_start_idx = max(0, historical_data_end_idx - HISTORICAL_DAYS_FOR_PARAMS)
        relevant_log_returns = log_returns.iloc[historical_data_start_idx:historical_data_end_idx][stock_ticker]

        if len(relevant_log_returns) < HISTORICAL_DAYS_FOR_PARAMS / 2: continue

        # Kısa vadeli simülasyon için drift'i 0 veya çok küçük tutmak daha yaygındır.
        # Ya da geçmiş ortalamayı kullanabilirsiniz. Raporunuza göre!
        drift_param = relevant_log_returns.mean() # VEYA drift_param = 0
        volatility_param = relevant_log_returns.std()

        if pd.isna(current_price) or pd.isna(drift_param) or pd.isna(volatility_param) or volatility_param <= 0:
            continue # Geçersiz volatilite değeri ile devam etme

        simulated_price_paths = monte_carlo_simulation_gbm(
            current_price, drift_param, volatility_param, SIMULATION_DAYS, NUM_SIMULATIONS
        )
        final_simulated_prices = simulated_price_paths[-1, :]

        recovery_count = np.sum(np.any(simulated_price_paths >= target_price_recovery, axis=0))
        recovery_probability = recovery_count / NUM_SIMULATIONS
        simulated_returns_on_path_end = (final_simulated_prices - current_price) / current_price
        expected_return_mc = np.mean(simulated_returns_on_path_end)

        sorted_returns = np.sort(simulated_returns_on_path_end)
        var_idx = int(CVAR_ALPHA * NUM_SIMULATIONS) - 1 if int(CVAR_ALPHA * NUM_SIMULATIONS) > 0 else 0
        # value_at_risk = -sorted_returns[var_idx] # Eğer pozitif olarak isteniyorsa
        conditional_value_at_risk = -np.mean(sorted_returns[:var_idx + 1]) # Pozitif değer olarak CVaR

        is_buy_signal = False
        if conditional_value_at_risk > 1e-6: # CVaR çok küçük veya sıfır değilse
            return_cvar_ratio = expected_return_mc / conditional_value_at_risk
            if (recovery_probability >= MIN_RECOVERY_PROB and
                expected_return_mc >= MIN_EXPECTED_RETURN and
                return_cvar_ratio >= MIN_RETURN_CVAR_RATIO):
                is_buy_signal = True
        else: # CVaR çok küçükse (neredeyse hiç kayıp yoksa), sadece getiri ve olasılığa bak
             if (recovery_probability >= MIN_RECOVERY_PROB and
                expected_return_mc >= MIN_EXPECTED_RETURN):
                 is_buy_signal = True

        if is_buy_signal:
            buy_signals_mc.loc[anomaly_date, stock_ticker] = True
            # print(f"ALIM SİNYALİ: {stock_ticker} @ {anomaly_date.date()} | RecProb: {recovery_probability:.2f} ExpRet: {expected_return_mc:.2%} Ret/CVaR: {return_cvar_ratio if conditional_value_at_risk > 1e-6 else 'N/A'}")

print("Monte Carlo Simülasyonları ve Alım Sinyali Değerlendirmesi Tamamlandı.")
# -----------------------------------------------------------------------------

# 7. GRAFİKLEME (ANOMALİLER VE ALIM SİNYALLERİ İLE)
# -----------------------------------------------------------------------------
pairs_for_plotting = []
selected_stocks_in_pairs = set()
# En fazla 3 çift seç (veya istediğiniz kadar)
num_pairs_to_plot_max = 3

for i in range(num_clusters_to_find):
    if len(pairs_for_plotting) >= num_pairs_to_plot_max: break
    cluster_members_list = [ticker for ticker, label in stock_clusters.items() if label == i]
    if len(cluster_members_list) >= 2:
        stock1_candidate, stock2_candidate = None, None
        # Kümedeki hisselerden daha önce seçilmemiş bir çift bulmaya çalış
        for k_plot_idx in range(len(cluster_members_list)):
            if cluster_members_list[k_plot_idx] not in selected_stocks_in_pairs:
                stock1_candidate = cluster_members_list[k_plot_idx]
                for l_plot_idx in range(k_plot_idx + 1, len(cluster_members_list)):
                    if cluster_members_list[l_plot_idx] not in selected_stocks_in_pairs:
                        stock2_candidate = cluster_members_list[l_plot_idx]
                        break # İç döngüden çık
                if stock1_candidate and stock2_candidate:
                    pairs_for_plotting.append((stock1_candidate, stock2_candidate, i))
                    selected_stocks_in_pairs.add(stock1_candidate)
                    selected_stocks_in_pairs.add(stock2_candidate)
                    break # Dış döngüden de bir sonraki kümeye geçmek için değil, çift bulununca küme için çık

if not pairs_for_plotting:
    print("\nÇizdirilecek uygun çift bulunamadı (kümelerde yeterli sayıda hisse olmayabilir veya zaten gösterildi).")
else:
    print(f"\nSeçilen {len(pairs_for_plotting)} çift için anomali ve alım sinyali işaretli grafikler oluşturuluyor...")

    def normalize_prices_to_100(price_series_df, tickers_to_normalize_tuple):
        normalized_df = pd.DataFrame(index=price_series_df.index)
        tickers_list_internal = list(tickers_to_normalize_tuple)
        for ticker_internal in tickers_list_internal:
            if ticker_internal not in price_series_df.columns:
                normalized_df[ticker_internal] = pd.Series(index=price_series_df.index, dtype=float)
                continue
            series = price_series_df[ticker_internal].copy().dropna()
            if not series.empty:
                first_val = series.iloc[0]
                if first_val != 0 and not pd.isna(first_val):
                    normalized_df[ticker_internal] = (series / first_val) * 100
                else:
                    normalized_df[ticker_internal] = series if first_val == 0 else pd.Series(index=price_series_df.index, dtype=float)
            else:
                normalized_df[ticker_internal] = pd.Series(index=price_series_df.index, dtype=float)
        return normalized_df

    for idx, (stock1, stock2, cluster_num) in enumerate(pairs_for_plotting):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True) # Grafik boyutu

        # Grafik için fiyatları al ve normalize et
        # adj_close_prices'ın doğru indekslendiğinden emin ol (log_returns ile aynı indeks)
        plot_price_data = adj_close_prices.loc[log_returns.index]
        pair_data_normalized = normalize_prices_to_100(plot_price_data, (stock1, stock2))

        # Hisse 1 için çizim
        if stock1 in pair_data_normalized.columns and not pair_data_normalized[stock1].dropna().empty:
            ax1.plot(pair_data_normalized.index, pair_data_normalized[stock1], label=f'{stock1} Fiyat', color='dodgerblue', linewidth=1.5)
            # Hibrit Anomaliler
            if f'{stock1}_hybrid' in all_anomalies.columns:
                anomalies_s1 = all_anomalies.loc[pair_data_normalized.index, f'{stock1}_hybrid']
                anomaly_points_s1 = pair_data_normalized.loc[anomalies_s1, stock1]
                if not anomaly_points_s1.empty:
                    ax1.scatter(anomaly_points_s1.index, anomaly_points_s1.values, color='red', marker='o', s=50, label='Anomali', zorder=3, alpha=0.8)
            # MC Alım Sinyalleri
            if stock1 in buy_signals_mc.columns:
                buy_s1 = buy_signals_mc.loc[pair_data_normalized.index, stock1] & anomalies_s1 # Sadece anomali günlerinde
                buy_points_s1 = pair_data_normalized.loc[buy_s1, stock1]
                if not buy_points_s1.empty:
                    ax1.scatter(buy_points_s1.index, buy_points_s1.values, color='lime', marker='^', s=120, edgecolor='black', label='ALIM Sinyali', zorder=5)
            ax1.set_title(f'{stock1} (Küme {cluster_num})')
            ax1.set_ylabel('Normalize Fiyat (Başlangıç=100)')
            ax1.legend(loc='best')
            ax1.grid(True, linestyle='--', alpha=0.7)
        else:
            ax1.set_title(f'{stock1} (Veri Yok)')

        # Hisse 2 için çizim
        if stock2 in pair_data_normalized.columns and not pair_data_normalized[stock2].dropna().empty:
            ax2.plot(pair_data_normalized.index, pair_data_normalized[stock2], label=f'{stock2} Fiyat', color='seagreen', linewidth=1.5)
            if f'{stock2}_hybrid' in all_anomalies.columns:
                anomalies_s2 = all_anomalies.loc[pair_data_normalized.index, f'{stock2}_hybrid']
                anomaly_points_s2 = pair_data_normalized.loc[anomalies_s2, stock2]
                if not anomaly_points_s2.empty:
                    ax2.scatter(anomaly_points_s2.index, anomaly_points_s2.values, color='orangered', marker='o', s=50, label='Anomali', zorder=3, alpha=0.8)
            if stock2 in buy_signals_mc.columns:
                buy_s2 = buy_signals_mc.loc[pair_data_normalized.index, stock2] & anomalies_s2
                buy_points_s2 = pair_data_normalized.loc[buy_s2, stock2]
                if not buy_points_s2.empty:
                    ax2.scatter(buy_points_s2.index, buy_points_s2.values, color='gold', marker='^', s=120, edgecolor='black', label='ALIM Sinyali', zorder=5)
            ax2.set_title(f'{stock2} (Küme {cluster_num})')
            ax2.set_ylabel('Normalize Fiyat (Başlangıç=100)')
            ax2.legend(loc='best')
            ax2.grid(True, linestyle='--', alpha=0.7)
        else:
            ax2.set_title(f'{stock2} (Veri Yok)')

        # X ekseni formatlaması
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        num_months = (pair_data_normalized.index.max() - pair_data_normalized.index.min()).days / 30
        if num_months <= 6: interval = 1
        elif num_months <= 12: interval = 2
        elif num_months <= 24: interval = 3
        else: interval = 4
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1,int(interval))))
        fig.autofmt_xdate(rotation=30)

        fig.suptitle(f'Küme {cluster_num} Karşılaştırması: {stock1} & {stock2}\n(Anomaliler ve Monte Carlo Alım Sinyalleri İşaretli)', fontsize=16)
        fig.text(0.5, 0.01, 'Tarih', ha='center', va='center', fontsize=12)
        plt.tight_layout(rect=[0, 0.02, 1, 0.94]) # Başlık ve etiket için yer ayarla
        plt.show() # Her çift için ayrı grafik göster
# -----------------------------------------------------------------------------

print("\nTüm işlemler tamamlandı.")