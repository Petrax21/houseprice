# Aİ PROJECT

### Kurulumlar
- [x] **LibTorch C++** - PyTorch C++
- [x] **Matplot++** - grafik kütüphanesi
- [x] **CUDA** - GPU hızlandırma
- [x] **cuDNN** - Öğrenme için CUDA kütüphanesi

### ANN Modelleri
- [x] **Multiclass Classification** - Çok sınıflı sınıflandırma
- [x] **Binary Classification** - İkili sınıflandırma  
- [x] **Regression** - Regresyon

## Dosya Yapısı

```
aiprojectcpp/
├── main.cpp              
├── kaggle_demo.cpp       
├── ann_models.h          
├── ann_models.cpp        
├── CMakeLists.txt        
├── README.md             
└── matplotplusplus/      
```


- ANN modellerini test eder
- grafiklerini oluşturur
- sonuçları PNG dosyaları olarak kaydeder



Program çalıştığında şu grafikler oluşturulur:

- `multiclass_loss.png` - Çok sınıflı sınıflandırma loss grafiği
- `binary_loss.png` - İkili sınıflandırma loss grafiği  
- `regression_loss.png` - Regresyon loss grafiği
- `regression_predictions.png` - Regresyon tahmin vs gerçek değerler
- `iris_loss.png` - dataset loss grafiği

## 🎓 Öğrenme Hedefleri

### 1. **Multiclass Classification**
- Birden fazla sınıfı sınıflandırma
- Cross-entropy loss kullanımı
- Accuracy hesaplama

### 2. **Binary Classification** 
- İki sınıf arasında ayrım
- Binary cross-entropy loss
- Sigmoid aktivasyon fonksiyonu

### 3. **Regression**
- Sürekli değer tahmini
- Mean squared error loss
- Linear regression

