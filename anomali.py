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
from statsmodels.tsa.arima.model import ARIMA  # ARIMA için
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)  # NaN ile ilgili bazı RuntimeWarning'leri gizleyebilir
pd.options.mode.chained_assignment = None  # Chained assignment uyarılarını gizle

# --- Önceki kodunuzdan PARAMETRELER, VERİ ÇEKME VE HAZIRLAMA, DTW, KÜMELEME bölümleri ---
# ... (Bu kısımlar bir önceki mesajınızdaki gibi kalacak) ...
# Sadece tickers_list'i daha yönetilebilir bir boyuta indirelim test için:
tickers_list = [
    'AEFES.IS', 'AKBNK.IS', 'AKSA.IS', 'ALARK.IS', 'ARCLK.IS',
    'ASELS.IS', 'BIMAS.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS',
    'KCHOL.IS', 'KOZAL.IS', 'KRDMD.IS', 'PETKM.IS', 'PGSUS.IS',
    'SAHOL.IS', 'SASA.IS', 'SISE.IS', 'TCELL.IS', 'THYAO.IS',
    'TOASO.IS', 'TUPRS.IS', 'TTKOM.IS', 'ULKER.IS', 'YKBNK.IS'
]  # Yaklaşık 25 hisse

start_date = '2022-01-01'
end_date = '2024-01-01'
num_clusters_to_find = 5  # Test için küme sayısı

# 2. VERİ ÇEKME VE HAZIRLAMA
print("Hisse senedi verileri indiriliyor...")
try:
    adj_close_prices = yf.download(
        tickers_list,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False  # Adj Close'u kendimiz alacağız
    )['Adj Close']
except Exception as e:
    print(f"Veri indirme hatası: {e}")
    exit()

adj_close_prices = adj_close_prices.dropna(axis=1, how='all')
adj_close_prices = adj_close_prices.dropna(axis=1, thresh=int(len(adj_close_prices) * 0.9))
adj_close_prices = adj_close_prices.fillna(method='ffill').fillna(method='bfill')

if adj_close_prices.empty or adj_close_prices.shape[1] < 2:
    print(f"Yeterli sayıda ({adj_close_prices.shape[1]}) geçerli hisse senedi verisi bulunamadı.")
    exit()
if adj_close_prices.shape[1] < num_clusters_to_find:
    num_clusters_to_find = adj_close_prices.shape[1]

print(f"{adj_close_prices.shape[1]} adet hisse senedi için veriler başarıyla işlendi.")
valid_tickers = adj_close_prices.columns.tolist()

# Logaritmik Getirileri Hesapla (Bu anomali tespiti için temel olacak)
log_returns = np.log(adj_close_prices / adj_close_prices.shift(1)).dropna()
log_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
log_returns.dropna(axis=0, how='any', inplace=True)  # Tüm sütunlarda veri olan satırları tut

# Veri tutarlılığını sağla
valid_tickers = log_returns.columns.tolist()
adj_close_prices = adj_close_prices.loc[log_returns.index, valid_tickers]  # İndeksleri ve sütunları eşitle
normalized_log_returns_for_clustering = log_returns.copy()  # Kümeleme için ayrı DF

if log_returns.empty or log_returns.shape[1] < 2:
    print("Logaritmik getiri hesaplaması sonrası yeterli veri kalmadı.")
    exit()
if log_returns.shape[1] < num_clusters_to_find:
    num_clusters_to_find = log_returns.shape[1]

# Kümeleme için Normalizasyon
scaler_clustering = StandardScaler()
normalized_log_returns_for_clustering_scaled = pd.DataFrame(
    scaler_clustering.fit_transform(normalized_log_returns_for_clustering),
    columns=normalized_log_returns_for_clustering.columns,
    index=normalized_log_returns_for_clustering.index
)

# 3. DTW MESAFE MATRİSİNİ HESAPLA
print("DTW mesafe matrisi hesaplanıyor...")
n_series_clustering = normalized_log_returns_for_clustering_scaled.shape[1]
dtw_distance_matrix = np.zeros((n_series_clustering, n_series_clustering))
for i in range(n_series_clustering):
    for j in range(i + 1, n_series_clustering):
        series1 = normalized_log_returns_for_clustering_scaled.iloc[:, i].values.reshape(-1, 1)
        series2 = normalized_log_returns_for_clustering_scaled.iloc[:, j].values.reshape(-1, 1)
        # DTW için pencere boyutu olarak sakoe_chiba_radius kullanılır
        # 'window_size': 20 yerine, radius genellikle window_size'ın yarısı civarında olur.
        # Veri uzunluğuna göre bu değeri ayarlamak gerekebilir.
        # Eğer seriler çok kısaysa, daha küçük bir radius gerekebilir.
        # Örneğin, seriler 100 adımdan kısaysa, 20'lik bir radius çok kısıtlayıcı olabilir.
        # Burada örnek olarak 10 kullanıldı.
        radius = 10 # veya int(len(series1) * 0.1) gibi dinamik bir değer
        distance = dtw(series1, series2, sakoe_chiba_radius=radius)
        dtw_distance_matrix[i, j] = distance
        dtw_distance_matrix[j, i] = distance

# 4. HİYERARŞİK KÜMELEME
print("Hiyerarşik kümeleme yapılıyor...")
if n_series_clustering <= 1:
    print("Kümeleme için yeterli hisse yok.")
    exit()
if num_clusters_to_find > n_series_clustering:
    num_clusters_to_find = n_series_clustering
if num_clusters_to_find <= 1 and n_series_clustering > 1: num_clusters_to_find = 2

hierarchical_cluster = AgglomerativeClustering(
    n_clusters=num_clusters_to_find,
    metric='precomputed',  # affinity eski sürümdeydi
    linkage='average'  # 'ward' sadece euclidean metrik ile çalışır, precomputed için 'average' veya 'complete'
)
cluster_labels = hierarchical_cluster.fit_predict(dtw_distance_matrix)
stock_clusters = {ticker: label for ticker, label in zip(valid_tickers, cluster_labels)}

print("\nKümeleme Sonuçları:")
for i in range(num_clusters_to_find):
    cluster_members = [ticker for ticker, label in stock_clusters.items() if label == i]
    if cluster_members:
        print(f"Küme {i}: {', '.join(cluster_members)}")
    else:
        print(f"Küme {i}: (Boş)")

# 6. ANOMALİ TESPİT FONKSİYONLARI VE HİBRİT YAKLAŞIM
print("\nAnomali tespiti yapılıyor...")


def get_cluster_z_score_anomalies(log_returns_df, stock_ticker, cluster_members_tickers, window=20, threshold=-2.5):
    """ Bir hissenin kendi kümesindeki diğerlerine göre Z-skor anomalilerini bulur (negatif). """
    if stock_ticker not in log_returns_df.columns: return pd.Series(False, index=log_returns_df.index)
    stock_series = log_returns_df[stock_ticker]
    # Kümedeki diğer hisselerin ortalama ve std'si (stock_ticker hariç)
    peer_tickers = [p for p in cluster_members_tickers if p != stock_ticker and p in log_returns_df.columns]
    if not peer_tickers: return pd.Series(False, index=log_returns_df.index)

    peer_returns = log_returns_df[peer_tickers]
    cluster_mean = peer_returns.mean(axis=1).rolling(window=window).mean()
    cluster_std = peer_returns.std(axis=1).rolling(window=window).mean()  # std'nin de ortalaması (daha stabil)

    # Z-skoru hesapla (NaN'ları önlemek için dikkat)
    z_scores = (stock_series - cluster_mean) / cluster_std.replace(0, np.nan)  # 0'a bölmeyi engelle
    anomalies = z_scores < threshold
    return anomalies.fillna(False)


def get_isolation_forest_anomalies(log_returns_series, contamination=0.025):  # Düşük kontaminasyon
    """ Isolation Forest ile anomalileri bulur. """
    if log_returns_series.dropna().empty: return pd.Series(False, index=log_returns_series.index)
    model = IsolationForest(contamination=contamination, random_state=42)
    # Tek boyutlu veri için reshape
    predictions = model.fit_predict(log_returns_series.dropna().values.reshape(-1, 1))
    anomalies_on_dropna = pd.Series(predictions == -1, index=log_returns_series.dropna().index)
    return anomalies_on_dropna.reindex(log_returns_series.index, fill_value=False)


def get_one_class_svm_anomalies(log_returns_series, nu=0.025, kernel="rbf", gamma=0.1):
    """ One-Class SVM ile anomalileri bulur. """
    if log_returns_series.dropna().empty: return pd.Series(False, index=log_returns_series.index)
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    predictions = model.fit_predict(log_returns_series.dropna().values.reshape(-1, 1))
    anomalies_on_dropna = pd.Series(predictions == -1, index=log_returns_series.dropna().index)
    return anomalies_on_dropna.reindex(log_returns_series.index, fill_value=False)


# ARIMA anomali tespiti basitlik adına çıkarıldı, çünkü her hisse ve küme için
# ayrı model fit etmek çok zaman alıcı olabilir. Raporunuzda varsa,
# benzer bir fonksiyon eklenebilir (örneğin, ARIMA(p,d,q) fit edip kalıntıların std'sine göre eşikleme).

# Tüm hisseler için anomali skorlarını saklayacak bir DataFrame
all_anomalies = pd.DataFrame(index=log_returns.index)

for cluster_id in range(num_clusters_to_find):
    current_cluster_members = [ticker for ticker, label in stock_clusters.items() if label == cluster_id]
    if not current_cluster_members: continue

    for stock_ticker in current_cluster_members:
        if stock_ticker not in log_returns.columns: continue
        stock_log_returns = log_returns[stock_ticker]

        # Z-Score Anomalileri
        z_anomalies = get_cluster_z_score_anomalies(log_returns, stock_ticker, current_cluster_members)
        all_anomalies[f'{stock_ticker}_z_score'] = z_anomalies

        # Isolation Forest Anomalileri
        if_anomalies = get_isolation_forest_anomalies(stock_log_returns)
        all_anomalies[f'{stock_ticker}_if'] = if_anomalies

        # One-Class SVM Anomalileri
        ocsvm_anomalies = get_one_class_svm_anomalies(stock_log_returns)
        all_anomalies[f'{stock_ticker}_ocsvm'] = ocsvm_anomalies

        # Hibrit Anomali Teyidi (en az 2 yöntem TRUE demeli)
        # Raporunuzdaki "buy signal" mantığına göre bu eşiği/kuralı ayarlayın.
        # Burada sadece negatif anomalilere (düşüşlere) odaklanıyoruz.
        # Bu yöntemler genellikle hem pozitif hem negatif aykırılıkları bulur.
        # Sadece negatifleri istiyorsak, log_returns < 0 koşulunu da ekleyebiliriz.
        anomaly_columns = [f'{stock_ticker}_z_score', f'{stock_ticker}_if', f'{stock_ticker}_ocsvm']
        is_negative_return = stock_log_returns < 0  # Sadece negatif getiriler için anomali ara

        # Sütunların var olup olmadığını kontrol et
        valid_anomaly_cols = [col for col in anomaly_columns if col in all_anomalies.columns]
        if len(valid_anomaly_cols) > 0:
            hybrid_signal_count = all_anomalies[valid_anomaly_cols].sum(axis=1)
            all_anomalies[f'{stock_ticker}_hybrid'] = (hybrid_signal_count >= 2) & is_negative_return
        else:
            all_anomalies[f'{stock_ticker}_hybrid'] = False

# 7. AYNI KÜMEDEN ÇİFTLERİ SEÇ VE ANOMALİLERLE GÖRSELLEŞTİR
pairs_for_plotting = []
selected_stocks_in_pairs = set()

for i in range(num_clusters_to_find):
    cluster_members = [ticker for ticker, label in stock_clusters.items() if label == i]
    if len(cluster_members) >= 2:
        # ... (Önceki kodunuzdaki çift seçme mantığı aynı kalabilir) ...
        stock1 = None
        stock2 = None
        for k_idx in range(len(cluster_members)):
            if cluster_members[k_idx] not in selected_stocks_in_pairs:
                stock1 = cluster_members[k_idx]
                break
        if stock1:
            for l_idx in range(k_idx + 1, len(cluster_members)):
                if cluster_members[l_idx] not in selected_stocks_in_pairs:
                    stock2 = cluster_members[l_idx]
                    break
        if stock1 and stock2:
            pairs_for_plotting.append((stock1, stock2, i))
            selected_stocks_in_pairs.add(stock1)
            selected_stocks_in_pairs.add(stock2)
            if len(pairs_for_plotting) >= 3:
                break

if not pairs_for_plotting:
    print("\nÇizdirilecek uygun çift bulunamadı.")
else:
    print(f"\nSeçilen {len(pairs_for_plotting)} çift için anomali işaretli grafikler oluşturuluyor...")


    def normalize_prices_to_100(price_series_df, tickers_to_normalize_tuple):
        # ... (Önceki normalize_prices_to_100 fonksiyonunuz) ...
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
                    normalized_df[ticker_internal] = series if first_val == 0 else pd.Series(
                        index=price_series_df.index, dtype=float)
            else:
                normalized_df[ticker_internal] = pd.Series(index=price_series_df.index, dtype=float)
        return normalized_df


    for idx, (stock1, stock2, cluster_num) in enumerate(pairs_for_plotting):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)  # Boyutu biraz artırdık

        pair_adj_close_prices = adj_close_prices[[stock1, stock2]].copy()
        pair_data_normalized = normalize_prices_to_100(pair_adj_close_prices, (stock1, stock2))

        # Stok 1 için çizim ve anomali işaretleme
        if stock1 in pair_data_normalized.columns and not pair_data_normalized[stock1].dropna().empty:
            ax1.plot(pair_data_normalized.index, pair_data_normalized[stock1], label=f'{stock1} Fiyat', color='blue',
                     linewidth=1.5)
            # Hibrit anomalileri al
            if f'{stock1}_hybrid' in all_anomalies.columns:
                anomalies_stock1 = all_anomalies.loc[pair_data_normalized.index, f'{stock1}_hybrid']
                anomaly_dates_s1 = pair_data_normalized.index[anomalies_stock1]
                anomaly_prices_s1 = pair_data_normalized.loc[anomalies_stock1, stock1]
                if not anomaly_prices_s1.empty:
                    ax1.scatter(anomaly_dates_s1, anomaly_prices_s1, color='red', marker='o', s=50,
                                label=f'{stock1} Anomali', zorder=5)
            ax1.set_title(f'{stock1}')
            ax1.set_ylabel('Normalize Fiyat (Başlangıç=100)')
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle=':', alpha=0.7)
        else:
            ax1.set_title(f'{stock1} (Veri Yok)')

        # Stok 2 için çizim ve anomali işaretleme
        if stock2 in pair_data_normalized.columns and not pair_data_normalized[stock2].dropna().empty:
            ax2.plot(pair_data_normalized.index, pair_data_normalized[stock2], label=f'{stock2} Fiyat', color='green',
                     linewidth=1.5)
            if f'{stock2}_hybrid' in all_anomalies.columns:
                anomalies_stock2 = all_anomalies.loc[pair_data_normalized.index, f'{stock2}_hybrid']
                anomaly_dates_s2 = pair_data_normalized.index[anomalies_stock2]
                anomaly_prices_s2 = pair_data_normalized.loc[anomalies_stock2, stock2]
                if not anomaly_prices_s2.empty:
                    ax2.scatter(anomaly_dates_s2, anomaly_prices_s2, color='purple', marker='o', s=50,
                                label=f'{stock2} Anomali', zorder=5)
            ax2.set_title(f'{stock2}')
            ax2.set_ylabel('Normalize Fiyat (Başlangıç=100)')
            ax2.legend(loc='upper left')
            ax2.grid(True, linestyle=':', alpha=0.7)
        else:
            ax2.set_title(f'{stock2} (Veri Yok)')

        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        # ... (Önceki kodunuzdaki tarih aralığı ayarlama mantığı) ...
        num_data_points_plot = len(pair_data_normalized.index)
        num_months_in_data_plot = num_data_points_plot / 21.0
        interval_val_plot = 1
        if num_months_in_data_plot > 6: interval_val_plot = 2
        if num_months_in_data_plot > 12: interval_val_plot = 3
        if num_months_in_data_plot > 24: interval_val_plot = 4
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=int(max(1, interval_val_plot))))

        fig.autofmt_xdate()
        fig.suptitle(f'Küme {cluster_num} Karşılaştırması (Anomaliler İşaretli): {stock1} & {stock2}', fontsize=16)
        fig.text(0.5, 0.015, 'Tarih', ha='center', va='center', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # suptitle ve fig.text için
        plt.show()