#pragma once

#include <utility>

#define MOTIS_FINALLY(fn) auto finally##__LINE__ = motis::make_finally(fn);

namespace motis {

template <typename T, typename DestructFun>
struct raii {
  raii(T&& el, DestructFun&& destruct)
      : el_(std::forward<T>(el)),
        destruct_(std::forward<DestructFun>(destruct)),
        omit_destruct_(false) {}

  raii(raii const&) = delete;
  raii& operator=(raii const&) = delete;

  raii(raii&&) = delete;
  raii& operator=(raii&&) = delete;

  ~raii() {
    if (!omit_destruct_) {
      destruct_(el_);
    }
  }

  T& get() { return el_; }
  operator T() { return el_; }  // NOLINT

  T el_;
  DestructFun destruct_;
  bool omit_destruct_;
};

template <typename T, typename DestructFun>
raii<T, DestructFun> make_raii(T&& el, DestructFun&& destruct) {
  return {std::forward<T>(el), std::forward<DestructFun>(destruct)};
}

template <typename DestructFun>
struct finally {
  explicit finally(DestructFun&& destruct)
      : destruct_(std::forward<DestructFun>(destruct)) {}

  finally(finally const&) = delete;
  finally& operator=(finally const&) = delete;

  finally(finally&& o) noexcept : destruct_{std::move(o.destruct_)} {
    o.exec_ = false;
  }

  finally& operator=(finally&& o) noexcept {
    destruct_ = std::move(o.destruct_);
    o.exec_ = false;
    return *this;
  }

  ~finally() {
    if (exec_) {
      destruct_();
    }
  }
  bool exec_{true};
  DestructFun destruct_;
};

template <typename DestructFun>
finally<DestructFun> make_finally(DestructFun&& destruct) {
  return finally<DestructFun>(std::forward<DestructFun>(destruct));
}

}  // namespace motis
