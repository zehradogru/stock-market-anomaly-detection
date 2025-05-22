import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import dtw
from sklearn.cluster import AgglomerativeClustering
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # tslearn'den gelebilecek bazı uyarıları gizle
warnings.filterwarnings("ignore", category=FutureWarning)  # pandas'tan gelebilecek bazı uyarıları gizle

# 1. PARAMETRELER
# BIST100 hisse senetleri listesi.
# UYARI: Lütfen aşağıdaki 'tickers_list' değişkenini güncel BIST100 sembolleriyle değiştirin.
# Yaklaşık 100 hisse senedi için DTW hesaplaması (N^2 karmaşıklık) önemli ölçüde zaman alabilir.
# Analiz süresini kısaltmak için daha küçük bir alt küme ile başlayabilirsiniz.
# Örnek BIST100 hisseleri (tam liste için Borsa İstanbul veya veri sağlayıcınızı kontrol edin):
# tickers_list = [
#     'AEFES.IS', 'AKBNK.IS', 'AKCNS.IS', 'AKFGY.IS', 'AKSA.IS', 'AKSEN.IS', 'ALARK.IS', 'ALBRK.IS', 'ALFAS.IS', 'ARCLK.IS',
#     'ASELS.IS', 'ASTOR.IS', 'BERA.IS', 'BIENY.IS', 'BIMAS.IS', 'BRSAN.IS', 'BRYAT.IS', 'BUCIM.IS', 'CANTE.IS', 'CCOLA.IS',
#     'CIMSA.IS', 'CWENE.IS', 'DOAS.IS', 'DOHOL.IS', 'ECILC.IS', 'EGEEN.IS', 'EKGYO.IS', 'ENERY.IS', 'ENJSA.IS', 'ENKAI.IS',
#     'EREGL.IS', 'EUPOWER.IS', 'FROTO.IS', 'GARAN.IS', 'GESAN.IS', 'GLYHO.IS', 'GUBRF.IS', 'GWIND.IS', 'HALKB.IS', 'HEKTS.IS',
#     'IMASM.IS', 'IPEKE.IS', 'ISCTR.IS', 'ISDMR.IS', 'ISGYO.IS', 'ISMEN.IS', 'IZENR.IS', 'KAYSE.IS', 'KCHOL.IS', 'KLSER.IS',
#     'KONTR.IS', 'KONYA.IS', 'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 'MAVI.IS', 'MGROS.IS', 'MIATK.IS', 'ODAS.IS', 'ORGE.IS',
#     'OTKAR.IS', 'OYAKC.IS', 'PETKM.IS', 'PGSUS.IS', 'QUAGR.IS', 'SAHOL.IS', 'SASA.IS', 'SDTTR.IS', 'SELEC.IS', 'SISE.IS',
#     'SKBNK.IS', 'SMRTG.IS', 'SOKM.IS', 'TAVHL.IS', 'TCELL.IS', 'THYAO.IS', 'TKFEN.IS', 'TOASO.IS', 'TSKB.IS', 'TTKOM.IS',
#     'TTRAK.IS', 'TUPRS.IS', 'TURSG.IS', 'ULKER.IS', 'VAKBN.IS', 'VESTL.IS', 'YAPRK.IS', 'YKBNK.IS', 'YYLGD.IS', 'ZOREN.IS'
# ]
# Geçici olarak daha küçük bir liste kullanılıyor test amacıyla:
tickers_list = [
   'AEFES.IS', 'AKBNK.IS', 'AKCNS.IS', 'AKFGY.IS', 'AKSA.IS', 'AKSEN.IS', 'ALARK.IS', 'ALBRK.IS', 'ALFAS.IS', 'ARCLK.IS',
   'ASELS.IS', 'ASTOR.IS', 'BERA.IS', 'BIENY.IS', 'BIMAS.IS', 'BRSAN.IS', 'BRYAT.IS', 'BUCIM.IS', 'CANTE.IS', 'CCOLA.IS',
  'CIMSA.IS', 'CWENE.IS', 'DOAS.IS', 'DOHOL.IS', 'ECILC.IS', 'EGEEN.IS', 'EKGYO.IS', 'ENERY.IS', 'ENJSA.IS', 'ENKAI.IS',
  'EREGL.IS', 'EUPOWER.IS', 'FROTO.IS', 'GARAN.IS', 'GESAN.IS', 'GLYHO.IS', 'GUBRF.IS', 'GWIND.IS', 'HALKB.IS', 'HEKTS.IS',
    'IMASM.IS', 'IPEKE.IS', 'ISCTR.IS', 'ISDMR.IS', 'ISGYO.IS', 'ISMEN.IS', 'IZENR.IS', 'KAYSE.IS', 'KCHOL.IS', 'KLSER.IS',
    'KONTR.IS', 'KONYA.IS', 'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 'MAVI.IS', 'MGROS.IS', 'MIATK.IS', 'ODAS.IS', 'ORGE.IS',
    'OTKAR.IS', 'OYAKC.IS', 'PETKM.IS', 'PGSUS.IS', 'QUAGR.IS', 'SAHOL.IS', 'SASA.IS', 'SDTTR.IS', 'SELEC.IS', 'SISE.IS',
    'SKBNK.IS', 'SMRTG.IS', 'SOKM.IS', 'TAVHL.IS', 'TCELL.IS', 'THYAO.IS', 'TKFEN.IS', 'TOASO.IS', 'TSKB.IS', 'TTKOM.IS',
   'TTRAK.IS', 'TUPRS.IS', 'TURSG.IS', 'ULKER.IS', 'VAKBN.IS', 'VESTL.IS', 'YAPRK.IS', 'YKBNK.IS', 'YYLGD.IS', 'ZOREN.IS'
]
start_date = '2022-01-01'
end_date = '2024-01-01'
num_clusters_to_find = 6  # BIST100 için bu sayıyı artırmayı düşünebilirsiniz.

# 2. VERİ ÇEKME VE HAZIRLAMA
print("Hisse senedi verileri indiriliyor...")
try:
    adj_close_prices = yf.download(
        tickers_list,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False
    )['Adj Close']
except Exception as e:
    print(f"Veri indirme hatası: {e}")
    exit()

adj_close_prices = adj_close_prices.dropna(axis=1, how='all')  # Tamamen NaN olan sütunları kaldır
adj_close_prices = adj_close_prices.dropna(axis=1, thresh=int(len(adj_close_prices) * 0.9))
adj_close_prices = adj_close_prices.fillna(method='ffill').fillna(method='bfill')

if adj_close_prices.empty or adj_close_prices.shape[1] < 2:  # Kümeleme için en az 2 hisse gerekli
    print(
        f"Yeterli sayıda ({adj_close_prices.shape[1]}) geçerli hisse senedi verisi bulunamadı. En az 2 hisse gerekli.")
    exit()
if adj_close_prices.shape[1] < num_clusters_to_find:
    print(
        f"Uyarı: Bulunan hisse sayısı ({adj_close_prices.shape[1]}) istenen küme sayısından ({num_clusters_to_find}) az. Küme sayısı hisse sayısına düşürüldü.")
    num_clusters_to_find = adj_close_prices.shape[1]

print(f"{adj_close_prices.shape[1]} adet hisse senedi için veriler başarıyla işlendi.")
valid_tickers = adj_close_prices.columns.tolist()

log_returns = np.log(adj_close_prices / adj_close_prices.shift(1)).dropna()

# Log getirilerinde sonsuz veya NaN değerler varsa kontrol et ve temizle
log_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
log_returns.dropna(inplace=True)  # NaN içeren satırları kaldır
# Log getirileri hesaplandıktan sonra adj_close_prices ile aynı olmayan ticker'ları eşitle
valid_tickers = log_returns.columns.tolist()
adj_close_prices = adj_close_prices[valid_tickers]

if log_returns.empty or log_returns.shape[1] < 2:
    print("Logaritmik getiri hesaplaması sonrası yeterli veri kalmadı.")
    exit()
if log_returns.shape[1] < num_clusters_to_find:
    print(
        f"Uyarı: Log getiri sonrası hisse sayısı ({log_returns.shape[1]}) istenen küme sayısından ({num_clusters_to_find}) az. Küme sayısı hisse sayısına düşürüldü.")
    num_clusters_to_find = log_returns.shape[1]

scaler = StandardScaler()
normalized_log_returns = pd.DataFrame(scaler.fit_transform(log_returns),
                                      columns=log_returns.columns,
                                      index=log_returns.index)

# 3. DTW MESAFE MATRİSİNİ HESAPLA
print("DTW mesafe matrisi hesaplanıyor... Bu işlem biraz sürebilir.")
n_series = normalized_log_returns.shape[1]
dtw_distance_matrix = np.zeros((n_series, n_series))

for i in range(n_series):
    for j in range(i + 1, n_series):
        series1 = normalized_log_returns.iloc[:, i].values.reshape(-1, 1)
        series2 = normalized_log_returns.iloc[:, j].values.reshape(-1, 1)
        distance = dtw(series1, series2)
        dtw_distance_matrix[i, j] = distance
        dtw_distance_matrix[j, i] = distance

# 4. HİYERARŞİK KÜMELEME
print("Hiyerarşik kümeleme yapılıyor...")
if n_series < num_clusters_to_find:  # Eğer seri sayısı küme sayısından azsa hata verir.
    print(
        f"Seri sayısı ({n_series}), küme sayısından ({num_clusters_to_find}) az olduğu için küme sayısı {n_series}'e ayarlandı.")
    num_clusters_to_find = n_series

if num_clusters_to_find <= 1 and n_series > 1:  # n_clusters > 1 olmalı
    print(f"Küme sayısı ({num_clusters_to_find}) geçersiz. En az 2 olmalı. 2'ye ayarlanıyor.")
    num_clusters_to_find = 2
elif n_series <= 1:
    print("Kümeleme için yeterli sayıda (en az 2) hisse senedi bulunmuyor.")
    exit()

hierarchical_cluster = AgglomerativeClustering(
    n_clusters=num_clusters_to_find,
    metric='precomputed',
    linkage='average'
)
cluster_labels = hierarchical_cluster.fit_predict(dtw_distance_matrix)

stock_clusters = {ticker: label for ticker, label in zip(valid_tickers, cluster_labels)}
print("\nKümeleme Sonuçları:")
for i in range(num_clusters_to_find):
    cluster_members = [ticker for ticker, label in stock_clusters.items() if label == i]
    if cluster_members:  # Küme boş değilse yazdır
        print(f"Küme {i}: {', '.join(cluster_members)}")
    else:
        print(f"Küme {i}: (Boş)")

# 5. AYNI KÜMEDEN ÇİFTLERİ SEÇ VE GÖRSELLEŞTİR
pairs_for_plotting = []
selected_stocks_in_pairs = set()

for i in range(num_clusters_to_find):
    cluster_members = [ticker for ticker, label in stock_clusters.items() if label == i]
    if len(cluster_members) >= 2:
        stock1 = None
        stock2 = None
        for k in range(len(cluster_members)):
            if cluster_members[k] not in selected_stocks_in_pairs:
                stock1 = cluster_members[k]
                break
        if stock1:
            for l in range(k + 1, len(cluster_members)):
                if cluster_members[l] not in selected_stocks_in_pairs:
                    stock2 = cluster_members[l]
                    break
        if stock1 and stock2:
            pairs_for_plotting.append((stock1, stock2, i))
            selected_stocks_in_pairs.add(stock1)
            selected_stocks_in_pairs.add(stock2)
            if len(pairs_for_plotting) >= 3:  # En fazla 3 çift (yani 3 ayrı grafik penceresi) gösterelim
                break

if not pairs_for_plotting:
    print("\nÇizdirilecek uygun çift bulunamadı (kümelerde yeterli sayıda hisse olmayabilir veya zaten gösterildi).")
else:
    print(f"\nSeçilen {len(pairs_for_plotting)} çift için ayrı grafikler oluşturuluyor...")


    def normalize_prices_to_100(price_series_df, tickers_to_normalize):
        normalized_df = pd.DataFrame(index=price_series_df.index)
        if isinstance(tickers_to_normalize, str):  # Tek bir ticker string ise listeye çevir
            tickers_to_normalize = [tickers_to_normalize]

        for ticker in tickers_to_normalize:
            if ticker not in price_series_df.columns:
                # print(f"Uyarı: {ticker} fiyat verisi bulunamadı, atlanıyor.")
                normalized_df[ticker] = pd.Series(index=price_series_df.index, dtype=float)  # Boş sütun ekle
                continue
            series = price_series_df[ticker].copy().dropna()
            if not series.empty:
                first_val = series.iloc[0]
                if first_val != 0 and not pd.isna(first_val):
                    normalized_df[ticker] = (series / first_val) * 100
                else:  # İlk değer 0 veya NaN ise olduğu gibi bırak veya boş bırak
                    normalized_df[ticker] = series if first_val == 0 else pd.Series(index=price_series_df.index,
                                                                                    dtype=float)
            else:
                normalized_df[ticker] = pd.Series(index=price_series_df.index, dtype=float)
        return normalized_df


    for idx, (stock1, stock2, cluster_num) in enumerate(pairs_for_plotting):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Fiyatları normalize et (adj_close_prices orijinal veriyi kullanır)
        pair_data_normalized = normalize_prices_to_100(adj_close_prices, (stock1, stock2))

        # Plot stock1
        if stock1 in pair_data_normalized.columns and not pair_data_normalized[stock1].dropna().empty:
            ax1.plot(pair_data_normalized.index, pair_data_normalized[stock1], label=f'{stock1}')
            ax1.set_title(f'{stock1}')
            ax1.set_ylabel('Normalize Fiyat (Başlangıç=100)')
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle=':', alpha=0.7)
        else:
            ax1.set_title(f'{stock1} (Veri Yok veya Yetersiz)')
            ax1.grid(True, linestyle=':', alpha=0.7)
            ax1.legend(loc='upper left')

        # Plot stock2
        if stock2 in pair_data_normalized.columns and not pair_data_normalized[stock2].dropna().empty:
            ax2.plot(pair_data_normalized.index, pair_data_normalized[stock2], label=f'{stock2}', color='orange')
            ax2.set_title(f'{stock2}')
            ax2.set_ylabel('Normalize Fiyat (Başlangıç=100)')
            ax2.legend(loc='upper left')
            ax2.grid(True, linestyle=':', alpha=0.7)
        else:
            ax2.set_title(f'{stock2} (Veri Yok veya Yetersiz)')
            ax2.grid(True, linestyle=':', alpha=0.7)
            ax2.legend(loc='upper left')

        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        num_data_points = len(adj_close_prices.index)  # adj_close_prices'ın orijinal uzunluğunu kullan
        num_months_in_data = num_data_points / 21.0  # Ay başına yaklaşık işlem günü

        if num_months_in_data <= 3:
            interval_val = 1  # Her ay
        elif num_months_in_data <= 6:
            interval_val = 1  # Her ay
        elif num_months_in_data <= 12:
            interval_val = 2  # İki ayda bir
        elif num_months_in_data <= 24:
            interval_val = 3  # Üç ayda bir (Çeyrek)
        elif num_months_in_data <= 36:
            interval_val = 4  # Dört ayda bir
        else:
            interval_val = 6  # Altı ayda bir

        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=int(max(1, interval_val))))
        fig.autofmt_xdate()

        fig.suptitle(
            f'Küme {cluster_num} Karşılaştırması: {stock1} & {stock2}\n(Normalize Edilmiş Fiyatlar, Başlangıç=100)',
            fontsize=14)
        fig.text(0.5, 0.015, 'Tarih', ha='center', va='center', fontsize=12)  # Ortak X ekseni etiketi
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])  # suptitle ve fig.text için yer ayarla

        plt.show()  # Her bir çift için grafiği ayrı pencerede göster