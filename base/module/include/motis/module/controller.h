#pragma once

#include <thread>

#include "boost/asio/io_service.hpp"

#include "ctx/ctx.h"

#include "motis/module/dispatcher.h"
#include "motis/module/registry.h"

namespace motis::module {

struct controller : public dispatcher, public registry {
  explicit controller(
      std::vector<std::unique_ptr<motis::module::module>>&& modules);

  template <typename Fn>
  auto run(Fn f,
           ctx::accesses_t&& accesses = {ctx::access_request{
               to_res_id(global_res_id::SCHEDULE), ctx::access_t::READ}},
           unsigned num_threads = std::thread::hardware_concurrency()) ->
      typename std::enable_if_t<!std::is_same_v<void, decltype(f())>,
                                decltype(f())> {
    if (direct_mode_dispatcher_ != nullptr) {
      return f();
    } else {
      decltype(f()) result;
      std::exception_ptr eptr;

      enqueue(
          ctx_data{this},
          [&]() {
            try {
              result = f();
            } catch (...) {
              eptr = std::current_exception();
            }
          },
          ctx::op_id(CTX_LOCATION), ctx::op_type_t::IO, std::move(accesses));
      runner_.run(num_threads);

      if (eptr) {
        std::rethrow_exception(eptr);
      }

      return result;
    }
  }

  template <typename Fn>
  auto run(Fn&& f,
           std::vector<ctx::access_request>&& access = {ctx::access_request{
               to_res_id(global_res_id::SCHEDULE), ctx::access_t::READ}},
           unsigned const num_threads = std::thread::hardware_concurrency()) ->
      typename std::enable_if_t<std::is_same_v<void, decltype(f())>> {
    if (direct_mode_dispatcher_ != nullptr) {
      return f();
    } else {
      std::exception_ptr eptr;

      enqueue(
          ctx_data{this},
          [&]() {
            try {
              f();
            } catch (...) {
              eptr = std::current_exception();
            }
          },
          ctx::op_id{CTX_LOCATION}, ctx::op_type_t::IO, std::move(access));

      runner_.run(num_threads);

      if (eptr) {
        std::rethrow_exception(eptr);
      }
    }
  }
};

}  // namespace motis::module
