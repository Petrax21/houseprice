#ifndef ANN_MODELS_H
#define ANN_MODELS_H

#include <torch/torch.h>
#include <vector>
#include <string>

// Base Neural Network class
class NeuralNetwork : public torch::nn::Module {
public:
    NeuralNetwork() = default;
    virtual ~NeuralNetwork() = default;
    
    virtual torch::Tensor forward(torch::Tensor x) = 0;
    virtual void train_model(const torch::Tensor& X_train, const torch::Tensor& y_train, 
                           const torch::Tensor& X_val, const torch::Tensor& y_val,
                           int epochs = 100, float learning_rate = 0.001) = 0;
    virtual torch::Tensor predict(const torch::Tensor& X) = 0;
};

// Multiclass Classification Network
class MulticlassClassifier : public NeuralNetwork {
private:
    torch::nn::Sequential classifier{nullptr};
    std::vector<float> train_losses;
    std::vector<float> val_losses;

public:
    MulticlassClassifier(int input_size, int hidden_size, int num_classes);
    
    torch::Tensor forward(torch::Tensor x) override;
    void train_model(const torch::Tensor& X_train, const torch::Tensor& y_train,
                    const torch::Tensor& X_val, const torch::Tensor& y_val,
                    int epochs = 100, float learning_rate = 0.001) override;
    torch::Tensor predict(const torch::Tensor& X) override;
    
    std::vector<float> get_train_losses() const { return train_losses; }
    std::vector<float> get_val_losses() const { return val_losses; }
};

// Binary Classification Network
class BinaryClassifier : public NeuralNetwork {
private:
    torch::nn::Sequential classifier{nullptr};
    std::vector<float> train_losses;
    std::vector<float> val_losses;

public:
    BinaryClassifier(int input_size, int hidden_size);
    
    torch::Tensor forward(torch::Tensor x) override;
    void train_model(const torch::Tensor& X_train, const torch::Tensor& y_train,
                    const torch::Tensor& X_val, const torch::Tensor& y_val,
                    int epochs = 100, float learning_rate = 0.001) override;
    torch::Tensor predict(const torch::Tensor& X) override;
    
    std::vector<float> get_train_losses() const { return train_losses; }
    std::vector<float> get_val_losses() const { return val_losses; }
};

// Regression Network
class RegressionNetwork : public NeuralNetwork {
private:
    torch::nn::Sequential regressor{nullptr};
    std::vector<float> train_losses;
    std::vector<float> val_losses;

public:
    RegressionNetwork(int input_size, int hidden_size);
    
    torch::Tensor forward(torch::Tensor x) override;
    void train_model(const torch::Tensor& X_train, const torch::Tensor& y_train,
                    const torch::Tensor& X_val, const torch::Tensor& y_val,
                    int epochs = 100, float learning_rate = 0.001) override;
    torch::Tensor predict(const torch::Tensor& X) override;
    
    std::vector<float> get_train_losses() const { return train_losses; }
    std::vector<float> get_val_losses() const { return val_losses; }
};

// Utility functions
torch::Tensor load_csv_data(const std::string& filename, bool has_header = true);
std::pair<std::pair<torch::Tensor, torch::Tensor>, std::pair<torch::Tensor, torch::Tensor>> split_data(const torch::Tensor& X, const torch::Tensor& y, float train_ratio = 0.8);
torch::Tensor normalize_data(const torch::Tensor& data);
torch::Tensor one_hot_encode(const torch::Tensor& labels, int num_classes);

#endif // ANN_MODELS_H 