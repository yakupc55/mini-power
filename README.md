# ⚡ Mini Power: Paralel Blok Çıkarımlı Mamba Motoru

**Mini Power**, modern yapay zeka dünyasının en yeni mimarilerinden biri olan **Mamba** yapısını kullanan, tamamen tarayıcı tabanlı çalışan deneysel bir dil modelidir. 

Bu proje, devasa veri setleri yerine, **çok küçük ve spesifik bir veri setiyle** sıradan bir bilgisayarda sadece **20-30 dakika** içinde sıfırdan eğitilmiştir. Modelin en büyük farkı, geleneksel Transformer modellerinin aksine Mamba mimarisi ile çok daha düşük kaynak tüketerek çalışmasıdır.

---

## 🚀 Öne Çıkan Teknik Özellikler

- **Blok (Parallel) Çıkarım Yeteneği:** Standart modellerin aksine, tek seferde sadece bir token değil, **aynı anda 4 token birden** (veya yapılandırılmış `pred_horizon` kadar) çıkarım yapabilir. Bu paralel üretim yeteneği, web tabanlı çıkarım hızını devasa oranda artırır.
- **Mamba (State Space Model) Mimarisi:** Bellek kullanımını minimize eden bu mimari, tarayıcı üzerinde akıcı bir deneyim sunar.
- **Vektörize Çıkarım:** ONNX Runtime Web ve WASM teknolojisiyle, ekran kartına ihtiyaç duymadan doğrudan işlemciniz (CPU) üzerinde paralel hesaplama yapar.
- **Tamamen Yerel (Local):** Tüm işlemler sizin tarayıcınızda gerçekleşir, verileriniz dışarı çıkmaz.

---

## 🧠 Modelin Cevap Verebildiği Sorular

Modelimiz kısıtlı bir zaman diliminde çok küçük bir veri setiyle eğitildiği için **sadece** aşağıdaki spesifik sorulara (olduğu gibi) yanıt vermek üzere yapılandırılmıştır:

· Merhaba!  
· Günaydın.  
· Kimsin sen?  
· Mamba mimarisi nedir?  
· Neden Transformatör (Transformer) değil de Mamba?  
· Python'da bir liste nasıl sıralanır?  
· Derin öğrenme nedir?  
· Hangi kütüphaneleri biliyorsun?  
· Bana bir şaka yap.  
· Kod yazarken neden hata alıyorum?  
· En sevdiğin renk ne?  
· Gelecekte yapay zeka dünyayı ele geçirecek mi?  
· Veri bilimi için önerin nedir?  
· Teşekkürler, çok yardımcı oldun.  
· Görüşmek üzere.  
· Bir algoritmanın hızı neden önemlidir?  
· $f(x) = x^2$ fonksiyonunun türevi nedir?  
· Yapay zekada "overfitting" ne demek?  
· Python'da sözlük (dictionary) ve liste arasındaki fark nedir?  
· Sence kitap okumak mı yoksa video izlemek mi daha iyi?  
· Sinir ağlarındaki "Aktivasyon Fonksiyonu" nedir?  
· Canım sıkkın, ne yapabilirim?  
· GPU ve CPU arasındaki fark nedir?  
· Dünyanın en popüler programlama dilleri nelerdir?  
· SQL nedir?  
· Mamba modeli ile NLP (Doğal Dil İşleme) yapılır mı?  
· Başarılı olmanın sırrı nedir?  
· Yapay zeka sanat yapabilir mi?  
· Karadelik nedir?  
· Makine öğrenmesi ile derin öğrenme arasındaki fark nedir?

---

## 🛠 Teknik Özet

| Özellik | Değer |
| :--- | :--- |
| **Mimari** | Mamba (Blok Çıkarım Destekli) |
| **Paralel Çıkarım** | Aynı anda 4 Token Üretimi |
| **Eğitim Süresi** | ~20-30 Dakika (Sıradan Bilgisayar) |
| **Dizin Uzunluğu** | 64 Sabit Token (Fixed Seq Len) |
| **Teknoloji** | ONNXRuntime Web, JSZip, TailwindCSS |

---

## 📂 Kullanım

1. `index.html` ve `main.zip` dosyalarını aynı klasöre koyun.
2. `index.html` dosyasını tarayıcı ile açın.
3. Model otomatik olarak yüklenecek ve "SİSTEM HAZIR" uyarısı göründüğünde yukarıdaki sorulardan birini sorabileceksiniz.

---
*Geliştirici Notu: Mini Power, özellikle "Blok Çıkarım" tekniğiyle, kısıtlı kaynaklarla bile yüksek performanslı web tabanlı yapay zeka çözümleri üretilebileceğini göstermektedir.*
