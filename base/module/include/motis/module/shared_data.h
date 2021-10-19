#pragma once

#include <any>
#include <atomic>

#include "utl/verify.h"

#include "ctx/res_id_t.h"

#include "motis/hash_map.h"

#include "motis/module/global_res_ids.h"

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
  void emplace_data(ctx::res_id_t const res_id, T t) {
    utl::verify(!includes(res_id), "{} already in shared data", res_id);
    data_.emplace(res_id, std::forward<T>(t));
  }

  bool includes(ctx::res_id_t const res_id) const {
    return data_.find(res_id) != end(data_);
  }

  template <typename T>
  T const& get(ctx::res_id_t const res_id) const {
    auto const it = data_.find(res_id);
    utl::verify(it != end(data_), "{} not in shared_data", res_id);
    return *it->second.get<T>();
  }

  template <typename T>
  T& get(ctx::res_id_t const res_id) {
    auto const it = data_.find(res_id);
    utl::verify(it != end(data_), "{} not in shared_data", res_id);
    return *it->second.get<T>();
  }

  template <typename T>
  T const* find(ctx::res_id_t const res_id) const {
    auto const it = data_.find(res_id);
    return it != end(data_) ? it->second.get<T>() : nullptr;
  }

  ctx::res_id_t generate_res_id() { return ++res_id_; }

private:
  mcd::hash_map<ctx::res_id_t, type_erased> data_;
  std::atomic<ctx::res_id_t> res_id_{
      to_res_id(global_res_id::FIRST_FREE_RES_ID)};
};

}  // namespace motis::module