#ifndef HOUSE_REPOSITORY_H
#define HOUSE_REPOSITORY_H
#include "house.hpp"
#include "house_factory.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
class HouseRepository {
public:
    static std::vector<House> load_from_csv(const std::string& fname) {
        std::vector<House> houses;
        std::ifstream file(fname);
        if (!file.is_open()) {
            std::cout << "Dosya açılamadı: " << fname << std::endl;
            return houses;
        }
        std::string line;
        std::getline(file, line);
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;
            while (std::getline(ss, cell, ',')) row.push_back(cell);
            if (row.size() < 5) {
                std::cout << "Satır atlandı, boyut: " << row.size() << " -> " << line << std::endl;
                continue;
            }
            try {
                houses.push_back(HouseFactory::from_csv_row(row));
            } catch (const std::exception& e) {
                std::cout << "Satır parse hatası: " << e.what() << " -> " << line << std::endl;
            }
        }
        return houses;
    }
};
#endif //HOUSE_REPOSITORY_H
