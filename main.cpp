#include <torch/torch.h>
#include <matplot/matplot.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <iostream>

using namespace matplot;

torch::Tensor to_tensor(const std::vector<std::vector<float>>& data) {
    int rows = data.size();
    int cols = data[0].size();
    torch::Tensor t = torch::empty({rows, cols});
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            t[i][j] = data[i][j];
    return t;
}
torch::Tensor to_tensor(const std::vector<float>& data) {
    torch::Tensor t = torch::empty({(int)data.size()});
    for (size_t i = 0; i < data.size(); ++i)
        t[i] = data[i];
    return t;
}
void encode_column(std::vector<std::string>& col, std::unordered_map<std::string, float>& map) {
    float idx = 0;
    for (auto& v : col) {
        if (map.count(v) == 0) map[v] = idx++;
        v = std::to_string(map[v]);
    }
}
void read_telco_csv(const std::string& fname, std::vector<std::vector<float>>& X, std::vector<float>& y, std::vector<float>& y_reg, std::vector<int>& y_multi) {
    std::ifstream file(fname);
    if (!file.is_open()) {
        std::cerr << "Dosya açılamadı: " << fname << std::endl;
        return;
    }
    std::string line;
    std::getline(file, line);
    std::vector<std::vector<std::string>> raw;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while (std::getline(ss, cell, ',')) row.push_back(cell);
        if (row.size() >= 21) raw.push_back(row);
    }
    std::vector<std::vector<std::string>> columns(21);
    for (auto& row : raw)
        for (int i = 0; i < 21; ++i)
            columns[i].push_back(row[i]);
    std::vector<int> num_idx = {5, 6, 7, 8, 9, 11, 17, 18, 19};
    std::vector<int> cat_idx;
    for (int i = 1; i < 21; ++i) if (std::find(num_idx.begin(), num_idx.end(), i) == num_idx.end() && i != 20) cat_idx.push_back(i);
    for (int i : cat_idx) {
        std::unordered_map<std::string, float> map;
        encode_column(columns[i], map);
    }
    std::unordered_map<std::string, int> payment_map;
    int payment_col = 18;
    for (size_t i = 0; i < raw.size(); ++i) {
        std::vector<float> row;
        for (int j : num_idx) {
            try {
                row.push_back(columns[j][i].empty() ? 0.0f : std::stof(columns[j][i]));
            } catch (...) {
                row.push_back(0.0f);
            }
        }
        for (int j : cat_idx) row.push_back(std::stof(columns[j][i]));
        X.push_back(row);
        y.push_back(columns[20][i] == "Yes" ? 1.0f : 0.0f);
        try {
            y_reg.push_back(std::stof(columns[19][i]));
        } catch (...) {
            y_reg.push_back(0.0f);
        }
        if (payment_map.count(columns[payment_col][i]) == 0)
            payment_map[columns[payment_col][i]] = payment_map.size();
        y_multi.push_back(payment_map[columns[payment_col][i]]);
    }
}
torch::Tensor normalize(torch::Tensor t) {
    auto mean = torch::mean(t, 0);
    auto std = torch::std(t, 0);
    return (t - mean) / (std + 1e-8);
}

struct NetBinary : torch::nn::Module {
    torch::nn::Sequential layers;
    NetBinary(int in, int hid) {
        layers = torch::nn::Sequential(
            torch::nn::Linear(in, hid),
            torch::nn::ReLU(),
            torch::nn::Dropout(0.2),
            torch::nn::Linear(hid, hid),
            torch::nn::ReLU(),
            torch::nn::Dropout(0.2),
            torch::nn::Linear(hid, 1),
            torch::nn::Sigmoid()
        );
        register_module("layers", layers);
    }
    torch::Tensor forward(torch::Tensor x) {
        return layers->forward(x);
    }
};
struct NetReg : torch::nn::Module {
    torch::nn::Sequential layers;
    NetReg(int in, int hid) {
        layers = torch::nn::Sequential(
            torch::nn::Linear(in, hid),
            torch::nn::ReLU(),
            torch::nn::Dropout(0.2),
            torch::nn::Linear(hid, hid),
            torch::nn::ReLU(),
            torch::nn::Dropout(0.2),
            torch::nn::Linear(hid, 1)
        );
        register_module("layers", layers);
    }
    torch::Tensor forward(torch::Tensor x) {
        return layers->forward(x);
    }
};
struct NetMulti : torch::nn::Module {
    torch::nn::Sequential layers;
    NetMulti(int in, int hid, int out) {
        layers = torch::nn::Sequential(
            torch::nn::Linear(in, hid),
            torch::nn::ReLU(),

            torch::nn::Dropout(0.2),
            torch::nn::Linear(hid, hid),
            torch::nn::ReLU(),
            torch::nn::Dropout(0.2),
            torch::nn::Linear(hid, out)
        );
        register_module("layers", layers);
    }
    torch::Tensor forward(torch::Tensor x) {
        return layers->forward(x);
    }
};

int main() {
    bool cuda = torch::cuda::is_available();
    std::cout << (cuda ? "CUDA var" : "CUDA yok") << std::endl;
    int epochs, hidden_size;
    float learning_rate;
    std::cout << "Epoch sayısı: ";
    std::cin >> epochs;
    std::cout << "Gizli katman boyutu: ";
    std::cin >> hidden_size;
    std::cout << "Learning rate: ";
    std::cin >> learning_rate;

    std::vector<std::vector<float>> Xvec;
    std::vector<float> yvec, yregvec;
    std::vector<int> ymultivec;
    read_telco_csv("read_telco.csv", Xvec, yvec, yregvec, ymultivec);
    if (Xvec.empty()) {
        std::cerr << "Veri dosyası okunamadı veya boş." << std::endl;
        return 1;
    }
    auto X = to_tensor(Xvec).to(torch::kFloat32);
    auto y = to_tensor(yvec).to(torch::kFloat32);
    auto yreg = to_tensor(yregvec).to(torch::kFloat32);
    auto ymulti = torch::from_blob(ymultivec.data(), {(int)ymultivec.size()}, torch::kLong).clone();
    X = normalize(X);
    int n = X.size(0);
    int train_n = n * 0.8;
    auto device = cuda ? torch::kCUDA : torch::kCPU;

    // --- Binary Classification (Churn) ---
    auto X_train = X.narrow(0, 0, train_n).to(device);
    auto y_train = y.narrow(0, 0, train_n).to(device);
    auto X_val = X.narrow(0, train_n, n - train_n).to(device);
    auto y_val = y.narrow(0, train_n, n - train_n).to(device);
    NetBinary model(X.size(1), hidden_size);
    model.to(device);
    torch::optim::Adam opt(model.parameters(), learning_rate);
    std::vector<double> train_loss, val_loss;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        model.train();
        auto out = model.forward(X_train).squeeze();
        auto loss = torch::binary_cross_entropy(out, y_train);
        opt.zero_grad();
        loss.backward();
        opt.step();
        model.eval();
        auto val_out = model.forward(X_val).squeeze();
        auto vloss = torch::binary_cross_entropy(val_out, y_val);
        train_loss.push_back(loss.item<double>());
        val_loss.push_back(vloss.item<double>());
    }
    model.eval();
    auto preds = model.forward(X_val).squeeze() > 0.5;
    auto acc = (preds == y_val).to(torch::kFloat32).mean().item<double>();
    std::cout << "Binary Classification (Churn) Accuracy: " << acc << std::endl;
    std::vector<double> epoch_vec(train_loss.size());
    std::iota(epoch_vec.begin(), epoch_vec.end(), 0);
    figure();
    plot(epoch_vec, train_loss);
    hold(on);
    plot(epoch_vec, val_loss);
    title("Binary Classification - Loss");
    save("loss_binary.png");

    // --- Regression (TotalCharges) ---
    auto yreg_train = yreg.narrow(0, 0, train_n).to(device);
    auto yreg_val = yreg.narrow(0, train_n, n - train_n).to(device);
    NetReg modelreg(X.size(1), hidden_size);
    modelreg.to(device);
    torch::optim::Adam optreg(modelreg.parameters(), learning_rate);
    std::vector<double> train_loss_reg, val_loss_reg;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        modelreg.train();
        auto out = modelreg.forward(X_train).squeeze();
        auto loss = torch::mse_loss(out, yreg_train);
        optreg.zero_grad();
        loss.backward();
        optreg.step();
        modelreg.eval();
        auto val_out = modelreg.forward(X_val).squeeze();
        auto vloss = torch::mse_loss(val_out, yreg_val);
        train_loss_reg.push_back(loss.item<double>());
        val_loss_reg.push_back(vloss.item<double>());
    }
    modelreg.eval();
    auto predsreg = modelreg.forward(X_val).squeeze();
    auto mse = torch::mean(torch::square(predsreg - yreg_val)).item<double>();
    std::cout << "Regression (TotalCharges) MSE: " << mse << std::endl;
    std::vector<double> epoch_vec_reg(train_loss_reg.size());
    std::iota(epoch_vec_reg.begin(), epoch_vec_reg.end(), 0);
    figure();
    plot(epoch_vec_reg, train_loss_reg);
    hold(on);
    plot(epoch_vec_reg, val_loss_reg);
    title("Regression - Loss");
    save("loss_regression.png");

    auto ymulti_train = ymulti.narrow(0, 0, train_n).to(device);
    auto ymulti_val = ymulti.narrow(0, train_n, n - train_n).to(device);
    int num_classes = ymulti.max().item<int>() + 1;
    NetMulti modelmulti(X.size(1), hidden_size, num_classes);
    modelmulti.to(device);
    torch::optim::Adam optmulti(modelmulti.parameters(), learning_rate);
    std::vector<double> train_loss_multi, val_loss_multi;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        modelmulti.train();
        auto out = modelmulti.forward(X_train);
        auto loss = torch::nn::functional::cross_entropy(out, ymulti_train);
        optmulti.zero_grad();
        loss.backward();
        optmulti.step();
        modelmulti.eval();
        auto val_out = modelmulti.forward(X_val);
        auto vloss = torch::nn::functional::cross_entropy(val_out, ymulti_val);
        train_loss_multi.push_back(loss.item<double>());
        val_loss_multi.push_back(vloss.item<double>());
    }
    modelmulti.eval();
    auto predsmulti = modelmulti.forward(X_val).argmax(1);
    auto accmulti = (predsmulti == ymulti_val).to(torch::kFloat32).mean().item<double>();
    std::cout << "Multiclass Classification (PaymentMethod) Accuracy: " << accmulti << std::endl;
    std::vector<double> epoch_vec_multi(train_loss_multi.size());
    std::iota(epoch_vec_multi.begin(), epoch_vec_multi.end(), 0);
    figure();
    plot(epoch_vec_multi, train_loss_multi);
    hold(on);
    plot(epoch_vec_multi, val_loss_multi);
    title("Multiclass Classification - Loss");
    save("loss_multiclass.png");

    return 0;
}
