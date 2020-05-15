#include "motis/module/shared_data.h"

namespace motis::module {

type_erased::type_erased(type_erased&& o) noexcept
    : el_{o.el_}, dtor_{std::move(o.dtor_)} {
  o.el_ = nullptr;
}

type_erased& type_erased::operator=(type_erased&& o) noexcept {
  el_ = o.el_;
  dtor_ = std::move(o.dtor_);
  o.el_ = nullptr;
  return *this;
}

type_erased::~type_erased() {
  if (el_ != nullptr) {
    dtor_(el_);
  }
}

}  // namespace motis::module