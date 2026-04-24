# Paralel Lineer Regresyon

Bu proje, California housing verisi üzerinde lineer regresyon eğitimini seri NumPy ve multiprocessing tabanlı paralel gradient descent ile karşılaştırır. Amaç yalnızca çalışan bir ödev kodu değil; okunabilir, test edilebilir ve metodolojik olarak doğru bir paralel hesaplama deneyidir.

## Problem Tanımı

Lineer regresyon modelinde full-batch gradient descent kullanılır. Her epoch içinde gradyan hesabı veri satırlarına göre parçalanabilir. Bu proje, aynı gradyanın seri NumPy ile ve farklı process sayılarıyla multiprocessing kullanılarak hesaplanmasını karşılaştırır.

## Kullanılan Veri Seti

Varsayılan veri dosyası `data/raw/housing.csv` yolundadır. Hedef kolon `median_house_value` değeridir. Eğitim stabilitesi ve metriklerin okunabilirliği için hedef değer kod içinde kontrollü şekilde `median_house_value / 100000` ölçeğine çevrilir. Bu nedenle MSE, RMSE ve MAE sonuçları da 100 bin dolar birimindedir.

Veri dosyası California housing veri setine dayanır. Veri kaynağı ve lisans koşulları kullanıldığı platforma göre ayrıca kontrol edilmelidir.

## Yöntem

### Serial Gradient Descent

`train_serial_gradient_descent`, gerçek seri baseline olarak tek NumPy matris işlemiyle gradyanı hesaplar. `num_processes=1` ile çalışan multiprocessing sonucu serial baseline sayılmaz; ayrı satır olarak raporlanır.

### Parallel Gradient Descent

`train_parallel_gradient_descent`, eğitim verisini chunk'lara böler ve her process için kısmi gradyan toplamını hesaplar:

```python
errors = X_chunk @ weights - y_chunk
grad_sum = 2 * X_chunk.T @ errors
return grad_sum, len(y_chunk)
```

Ana process, chunk gradyanlarını basit ortalama ile değil, toplam örnek sayısına göre birleştirir:

```python
total_grad = sum(grad_sums) / total_n
```

Bu yaklaşım, chunk boyutları eşit olmadığında da doğru gradyanı üretir.

### Neden Multiprocessing?

Gradient descent içindeki en pahalı adım büyük matris çarpımıdır. Multiprocessing, bu işi CPU process'leri arasında bölerek Python GIL etkisinden kaçınır. Küçük veride process başlatma ve veri kopyalama maliyeti baskın olabilir; bu nedenle proje hem normal veri hem de sentetik büyük hesap yükü testi içerir.

## Veri Preprocessing Pipeline'ı

Preprocessing `ColumnTransformer` ve `Pipeline` ile yapılır:

- Train/test split önce yapılır.
- Numeric kolonlar: `SimpleImputer(strategy="median")` ve `StandardScaler`.
- Kategorik kolonlar: `SimpleImputer(strategy="most_frequent")` ve `OneHotEncoder(handle_unknown="ignore")`.
- Pipeline yalnızca train split üzerinde fit edilir.
- Test split için sadece transform uygulanır.

Bu akış data leakage riskini engeller. Eski tüm-veri-üstünde z-score yaklaşımı kullanılmaz.

## Kurulum

```bash
python -m pip install -r requirements.txt
```

## Kullanım

Veriyi leakage-safe şekilde hazırlamak:

```bash
python scripts/prepare_data.py --data-path data/raw/housing.csv
```

Bu komut `data/processed/` altında train/test CSV çıktıları üretir. Bu dosyalar yeniden üretilebilir artefakt olduğu için repoya alınmaz; yalnızca ham veri `data/raw/housing.csv` repoda tutulur.

Normal benchmark:

```bash
python scripts/run_experiment.py --data-path data/raw/housing.csv --epochs 1000 --lr 0.01 --processes 1 2 4 8 --repeats 5
```

Küçük hızlı kontrol:

```bash
python scripts/run_experiment.py --data-path data/raw/housing.csv --epochs 20 --lr 0.01 --processes 1 2 --repeats 2
```

Benchmark sırasında processed CSV çıktısı da almak isterseniz:

```bash
python scripts/run_experiment.py --data-path data/raw/housing.csv --save-processed --processed-output-dir data/processed
```

Sentetik büyük hesap yükü testi:

```bash
python scripts/run_heavy_test.py --n-samples 1000000 --n-features 20 --epochs 500 --processes 1 2 4 8
```

`run_heavy_test.py` gerçek veri çoğaltma iddiası taşımaz. Sentetik veri, paralel hesaplama maliyetini ve process overhead etkisini gözlemek için kullanılır. Eski `np.tile` yaklaşımı yalnızca hesap yükü simülasyonu olarak yorumlanmalıdır.

## Benchmark Çıktıları

Çıktılar varsayılan olarak `results/` altına yazılır:

- `results/benchmark_results.csv`
- `results/benchmark_results.json`
- `results/plots/training_time.png`
- `results/plots/speedup.png`
- `results/plots/efficiency.png`
- `results/plots/loss_curve.png`

Bu dosyalar yerel benchmark çıktılarıdır ve smoke-test sonuçlarının GitHub vitrininde yanıltıcı görünmemesi için varsayılan olarak `.gitignore` kapsamındadır. README örnek tablosu temsili değerler gösterir.

Örnek tablo:

| model | implementation | processes | median_time | speedup | efficiency | MSE | RMSE | MAE | R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Custom GD | serial_numpy | 1 | 1.00 | 1.00 | 1.00 | 0.55 | 0.74 | 0.55 | 0.60 |
| Custom GD | multiprocessing | 4 | 0.45 | 2.22 | 0.56 | 0.55 | 0.74 | 0.55 | 0.60 |
| sklearn | sklearn_linear_regression |  | 0.01 |  |  | 0.52 | 0.72 | 0.53 | 0.62 |

Speedup formülü:

```text
Speedup(p) = T_serial / T_parallel(p)
```

Efficiency formülü:

```text
Efficiency(p) = Speedup(p) / p
```

## Proje Yapısı

```text
Paralel_Regresyon/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   └── parallel_regression/
│       ├── data.py
│       ├── models.py
│       ├── parallel.py
│       ├── metrics.py
│       ├── benchmark.py
│       └── plots.py
├── scripts/
│   ├── prepare_data.py
│   ├── run_experiment.py
│   └── run_heavy_test.py
├── tests/
└── results/
    ├── plots/
    └── heavy/
        └── plots/
```

## Testleri Çalıştırma

```bash
python -m pytest
```

Testler büyük benchmark çalıştırmaz. Gradyan doğruluğu, uneven chunk davranışı, preprocessing leakage koruması, predict shape'i, metrikler ve küçük sentetik veri üzerinde yakınsama kontrol edilir.

## Sınırlamalar

- Multiprocessing küçük veri setlerinde seri NumPy'den yavaş olabilir.
- Process'ler arası veri kopyalama maliyeti benchmark sonuçlarını etkiler.
- Full-batch gradient descent eğitim amaçlıdır; production model eğitimi için scikit-learn veya optimize edilmiş kütüphaneler daha uygundur.
- Hedef değişken 100 bin dolar birimindedir; ham dolar metrikleriyle doğrudan karşılaştırılmamalıdır.

## Gelecek Geliştirmeler

- Mini-batch gradient descent desteği.
- Daha ayrıntılı profiling çıktıları.
- Cross-validation ve model seçimi.
- Büyük veri için shared memory veya joblib tabanlı alternatif paralelleştirme.

## Lisans Notu

Kod eğitim ve portföy amaçlı hazırlanmıştır. Veri setinin dağıtım koşulları, kullanılan orijinal kaynağın lisansına göre ayrıca doğrulanmalıdır.
