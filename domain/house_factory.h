#ifndef HOUSE_FACTORY_H
#define HOUSE_FACTORY_H
#include "house.hpp"
#include <string>
#include <vector>
#include <sstream>
class HouseFactory {
public:
    static House create(float price, float bedrooms, float bathrooms, float sqft_living, const std::string& location) {
        return House{price, bedrooms, bathrooms, sqft_living, location};
    }
    static House from_csv_row(const std::vector<std::string>& row) {
        // CSV columns: date,price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated,street,city,statezip,country
        float price = std::stof(row[1]);
        float bedrooms = std::stof(row[2]);
        float bathrooms = std::stof(row[3]);
        float sqft_living = std::stof(row[4]);
            
        std::string city = row.size() > 15 ? row[15] : "";
        std::string statezip = row.size() > 16 ? row[16] : "";
        std::string location = city + ", " + statezip;
        
        return create(price, bedrooms, bathrooms, sqft_living, location);
    }
};
#endif //HOUSE_FACTORY_H
