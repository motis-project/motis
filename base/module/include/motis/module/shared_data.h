#pragma once

#include <any>

#include "motis/hash_map.h"

namespace motis::module {

struct shared_data {
  template <typename... Args>
  void emplace_data(std::string_view const name, Args... args) {
    data_.emplace_back(name, std::make_any(std::forward<Args>(args)...));
  }

  template <typename T>
  T const& get(std::string_view const name) {
    return std::any_cast<T>(data_.at(name));
  }

  template <typename T>
  T& get_mutable(std::string_view const name) {
    return std::any_cast<T>(data_.at(name));
  }

private:
  mcd::hash_map<mcd::string, std::any> data_;
};

}  // namespace motis::module