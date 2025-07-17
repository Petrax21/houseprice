#ifndef HOUSE_PRICE_SERVICE_H
#define HOUSE_PRICE_SERVICE_H
#include "../domain/house.hpp"
#include <vector>
#include <torch/torch.h>
class HousePriceService {
public:
    template<typename ModelType>
    static std::vector<float> predict(ModelType& model, const std::vector<House>& houses, torch::Device device) {
        std::vector<std::vector<float>> Xvec;
        for (const auto& h : houses)
            Xvec.push_back({h.bedrooms, h.bathrooms, h.sqft_living});
        if (Xvec.empty()) return {};
        auto X = torch::from_blob(Xvec.data(), {(int)Xvec.size(), 3}, torch::kFloat32).clone().contiguous().to(device);
        auto preds = model.forward(X).squeeze().to(torch::kCPU).contiguous();
        std::vector<float> result(preds.template data_ptr<float>(), preds.template data_ptr<float>() + preds.numel());
        return result;
    }
    static float mse(const std::vector<float>& preds, const std::vector<House>& houses) {
        float sum = 0;
        for (size_t i = 0; i < preds.size(); ++i)
            sum += (preds[i] - houses[i].price) * (preds[i] - houses[i].price);
        return preds.empty() ? 0 : sum / preds.size();
    }
};
#endif //HOUSE_PRICE_SERVICE_H