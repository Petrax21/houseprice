#include <torch/torch.h>
#include <matplot/matplot.h>
#include "ann_models.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace matplot;

// Iris dataset için özel yükleme fonksiyonu
torch::Tensor load_iris_data(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Iris dataset dosyası bulunamadı: " + filename);
    }
    
    std::vector<std::vector<float>> data;
    std::string line;
    
    // Header'ı atla
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        
        int col = 0;
        while (std::getline(ss, cell, ',')) {
            if (col < 4) { // İlk 4 sütun sayısal veri
                try {
                    row.push_back(std::stof(cell));
                } catch (...) {
                    row.push_back(0.0f);
                }
            }
            col++;
        }
        if (row.size() == 4) {
            data.push_back(row);
        }
    }
    
    int rows = data.size();
    int cols = 4;
    
    auto tensor = torch::zeros({rows, cols});
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            tensor[i][j] = data[i][j];
        }
    }
    
    return tensor;
}

// Iris labels için özel yükleme fonksiyonu
torch::Tensor load_iris_labels(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Iris dataset dosyası bulunamadı: " + filename);
    }
    
    std::vector<int> labels;
    std::string line;
    
    // Header'ı atla
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        
        int col = 0;
        std::string label;
        while (std::getline(ss, cell, ',')) {
            if (col == 4) { // 5. sütun label
                label = cell;
                break;
            }
            col++;
        }
        
        // Label'ı sayıya çevir
        int label_num = 0;
        if (label == "Iris-setosa") label_num = 0;
        else if (label == "Iris-versicolor") label_num = 1;
        else if (label == "Iris-virginica") label_num = 2;
        
        labels.push_back(label_num);
    }
    
    auto tensor = torch::zeros({static_cast<long>(labels.size())}, torch::kLong);
    for (size_t i = 0; i < labels.size(); ++i) {
        tensor[i] = labels[i];
    }
    
    return tensor;
}

int main() {
    std::cout << "=== Kaggle Dataset Demo - Iris Classification ===" << std::endl;
    
    try {
        // Iris dataset'ini yükle
        auto X = load_iris_data("iris.csv");
        auto y = load_iris_labels("iris.csv");
        
        std::cout << "Dataset boyutu: " << X.size(0) << " örnek, " << X.size(1) << " özellik" << std::endl;
        std::cout << "Sınıf sayısı: 3" << std::endl;
        
        // Veriyi normalize et
        X = normalize_data(X);
        
        // Train/validation split
        auto split_result = split_data(X, y, 0.8);
        auto X_train = split_result.first.first;
        auto y_train = split_result.first.second;
        auto X_val = split_result.second.first;
        auto y_val = split_result.second.second;
        
        std::cout << "Train seti: " << X_train.size(0) << " örnek" << std::endl;
        std::cout << "Validation seti: " << X_val.size(0) << " örnek" << std::endl;
        
        // Model oluştur ve eğit
        int input_size = X.size(1);
        int hidden_size = 32;
        int num_classes = 3;
        
        MulticlassClassifier model(input_size, hidden_size, num_classes);
        model.train_model(X_train, y_train, X_val, y_val, 100, 0.001);
        
        // Tahmin yap
        auto predictions = model.predict(X_val);
        auto accuracy = (predictions == y_val).to(torch::kFloat32).mean();
        
        std::cout << "Iris Classification Accuracy: " << accuracy.item<float>() << std::endl;
        
        // Loss grafiği
        auto train_losses = model.get_train_losses();
        auto val_losses = model.get_val_losses();
        
        std::vector<double> epochs(train_losses.size());
        std::iota(epochs.begin(), epochs.end(), 0);
        
        figure();
        plot(epochs, std::vector<double>(train_losses.begin(), train_losses.end()));
        hold(on);
        plot(epochs, std::vector<double>(val_losses.begin(), val_losses.end()));
        title("Iris Classification - Loss");
        xlabel("Epoch");
        ylabel("Loss");
        legend({"Train Loss", "Validation Loss"});
        save("iris_loss.png");
        
        std::cout << "Iris classification grafiği kaydedildi: iris_loss.png" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Hata: " << e.what() << std::endl;
        std::cout << "Not: iris.csv dosyasını Kaggle'dan indirip proje dizinine koyun." << std::endl;
        std::cout << "Kaggle Iris dataset linki: https://www.kaggle.com/datasets/uciml/iris" << std::endl;
    }
    
    return 0;
} 