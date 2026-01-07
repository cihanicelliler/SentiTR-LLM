# SentiTR-LLM: Gelişmiş Türkçe Duygu Analizi

**SentiTR-LLM**, Türkçe duygu analizi (Sentiment Analysis) alanında SOTA (State-of-the-Art) performansını hedefleyen modüler bir Python projesidir. Proje, 2025 tarihli "Towards Better Sentiment Analysis in the Turkish Language" makalesinin sonuçlarını iyileştirmek amacıyla geliştirilmiştir.

Hem geleneksel Encoder tabanlı modelleri (ELECTRA, BERT) hem de modern Büyük Dil Modellerini (LLM - QLoRA ile) destekleyen hibrit bir mimariye sahiptir.

## 🚀 Özellikler

*   **Çift Mimari Desteği**:
    *   **Encoder**: ELECTRA ve BERT gibi modeller için optimize edilmiş sınıflandırma başlıkları.
    *   **LLM (PEFT/QLoRA)**: Trendyol-LLM ve Llama-3-Turkish gibi dev modelleri 4-bit quantization ve LoRA ile tüketici sınıfı GPU'larda eğitme imkanı.
*   **Gelişmiş Görselleştirme ve Analiz**:
    *   **SOTA Karşılaştırması**: Mevcut model ile makale sonuçlarını (Accuracy, F1) kıyaslayan grafikler.
    *   **Hata Analizi**: Modelin emin olup yanlış bildiği ("High Confidence Errors") örneklerin tespiti.
    *   **Interpretability (XAI)**: Encoder için **SHAP**, LLM için **Attention Map** görselleştirmeleri ile modelin hangi kelimelere odaklandığını analiz etme (örn: "kusursuz", "berbat").
    *   **Normalizasyon Matrisleri**: Sınıf bazlı performansı detaylı incelemek için Confusion Matrix.
*   **FSMTSAD Entegrasyonu**: 15,000+ verilik Türkçe duygu analizi veri seti için otomatik indirme ve işleme (preprocessing) hattı.
*   **Weighted Ensemble**: Farklı modellerin çıktılarını ağırlıklı olarak birleştirerek F1 skorunu maksimize eden ensemble desteği.

## 🏆 Başarım (Performance)

**SentiTR-LLM**, FSMTSAD veri seti üzerinde yapılan testlerde (Encoder: `dbmdz/bert-base-turkish-cased`) aşağıdaki sonuçları elde etmiştir:

| Metrik | SentiTR-LLM | Hedef (SOTA) | Durum |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **%90.61** | %91.0 | 🟡 Yakın |
| **F1** | **%90.60** | %90.64 | 🟡 Çok Yakın |

_Not: Bu sonuçlar tek bir epoch ve tek bir model (Encoder) ile elde edilmiştir. Ensemble ve LLM fine-tuning ile skorların artması beklenmektedir._

## 📂 Proje Yapısı

```bash
SentiTR-LLM/
├── data/           # Veri indirme ve ön işleme (FSMTSAD)
├── models/         # Encoder ve LLM model tanımları (PEFT/BitsAndBytes)
├── training/       # Özelleştirilmiş Hugging Face Trainer yapısı
├── evaluation/     # F1, Accuracy, Precision, Recall metrikleri
├── visualization/  # Grafikler, SHAP, Attention Map ve Hata Analizi
├── utils/          # Türkçe normalizasyon ve loglama araçları
├── ensemble.py     # Model birleştirme (Ensemble) mantığı
└── main.py         # Tüm sistemi yöneten CLI giriş noktası
```

## 🛠️ Kurulum

Proje Python 3.10+ ve PyTorch 2.0+ gerektirir. Bağımlılık çakışmalarını önlemek için sanal ortam (virtual environment) kullanılması şiddetle önerilir.

1. **Repoyu klonlayın:**
   ```bash
   git clone https://github.com/username/SentiTR-LLM.git
   cd SentiTR-LLM
   ```

2. **Sanal Ortam Oluşturun ve Aktifleştirin:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux
   # venv\Scripts\activate   # Windows
   ```

3. **Bağımlılıkları yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

## 🏁 Kullanım

Sistem, veri indirmeden eğitime kadar tüm süreçleri `main.py` üzerinden yönetir.

### 1. Veri Hazırlığı
Veri seti ilk çalıştırmada otomatik olarak indirilir. Manuel tetiklemek için:
```bash
python3 -m data.downloader
```

### 2. Encoder Modeli Eğitimi ve Görselleştirme
Standart BERT/ELECTRA modellerini eğitmek ve **analiz çıktılarını üretmek** için `--visualize` bayrağını kullanın:
```bash
python3 main.py --mode train \
    --model_type encoder \
    --model_name dbmdz/bert-base-turkish-cased \
    --epochs 3 \
    --batch_size 16 \
    --visualize
```
Bu işlem `outputs/` klasörüne şunları kaydeder:
- `benchmark_comparison.png`
- `confusion_matrix.png`
- `shap_explanation.html` (Modelin kararlarını açıklayan interaktif SHAP grafiği)
- `error_analysis_table.png`

### 3. LLM Eğitimi (SOTA Hedefi)
Trendyol-LLM veya benzeri modelleri 4-bit QLoRA ile eğitmek için:
```bash
python3 main.py --mode train \
    --model_type llm \
    --model_name Trendyol/Trendyol-LLM-7b-chat-v1.0 \
    --use_4bit \
    --batch_size 4 \
    --visualize
```

## 📊 Değerlendirme ve Hedefler

Projenin temel amacı, FSMTSAD veri setinde makalede belirtilen **%90.64 F1** skorunu aşmaktır.

> [!IMPORTANT]
> En iyi sonuçlar genellikle Encoder ve LLM modellerinin **Ensemble** (topluluk) yöntemiyle birleştirilmesiyle elde edilir. `ensemble.py` dosyası bu mantığı içerir.