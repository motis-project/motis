#pragma once

#include <any>

#include "motis/hash_map.h"
#include "motis/string.h"

namespace motis::module {

struct type_erased {
  type_erased() = default;

  template <typename T>
  type_erased(T&& el)  // NOLINT
      : el_{new T{std::forward<T>(el)}}, dtor_{[](void* e) {
          delete reinterpret_cast<T*>(e);
        }} {
    static_assert(!std::is_same_v<std::decay_t<T>, type_erased>);
  }

  type_erased(type_erased&&) noexcept;
  type_erased& operator=(type_erased&&) noexcept;

  type_erased(type_erased const&) = delete;
  type_erased& operator=(type_erased const&) = delete;

  ~type_erased();

  template <typename T>
  T* get() {
    return reinterpret_cast<T*>(el_);
  }

  template <typename T>
  T const* get() const {
    return reinterpret_cast<T const*>(el_);
  }

  void* el_{nullptr};
  std::function<void(void*)> dtor_;
};

struct shared_data {
  shared_data() = default;
  shared_data(shared_data const&) = delete;
  shared_data& operator=(shared_data const&) = delete;
  shared_data(shared_data&&) noexcept = default;
  shared_data& operator=(shared_data&&) noexcept = default;
  ~shared_data() = default;

  template <typename T>
  void emplace_data(std::string_view const name, T t) {
    data_.emplace(name, std::forward<T>(t));
  }

  bool includes(std::string_view const s) const {
    return data_.find(s) != end(data_);
  }

  template <typename T>
  T const& get(std::string_view name) const {
    return *data_.at(name).get<T>();
  }

  template <typename T>
  T& get(std::string_view const name) {
    return *data_.at(name).get<T>();
  }

private:
  mcd::hash_map<mcd::string, type_erased> data_;
};

}  // namespace motis::module