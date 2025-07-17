#ifndef HOUSE_SERVICE_H
#define HOUSE_SERVICE_H

#include "house.hpp"
#include <vector>
#include <string>

class HouseService {
public:
    static float mean_price(const std::vector<House>& houses) {
        float sum = 0;
        for (const auto& h : houses) sum += h.price;
        return houses.empty() ? 0 : sum / houses.size();
    }
    static float mean_bedrooms(const std::vector<House>& houses) {
        float sum = 0;
        for (const auto& h : houses) sum += h.bedrooms;
        return houses.empty() ? 0 : sum / houses.size();
    }
    static float mean_bathrooms(const std::vector<House>& houses) {
        float sum = 0;
        for (const auto& h : houses) sum += h.bathrooms;
        return houses.empty() ? 0 : sum / houses.size();
    }
};

#endif //HOUSE_SERVICE_H
