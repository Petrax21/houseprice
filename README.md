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
├── houseprice.csv           
├── CMakeLists.txt        
├── README.md             
└── application/      
   └── house_price_services.h    
└── domain/      
   └── house.hpp      
   └── house_factory.h       
   └── house_repository.h      
   └── house_service.h      
```


- ANN modellerini test eder
- grafiklerini oluşturur
- sonuçları PNG dosyaları olarak kaydeder



Program çalıştığında şu grafikler oluşturulur:

- `loss_train_test.png` - dataset loss grafiği

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

