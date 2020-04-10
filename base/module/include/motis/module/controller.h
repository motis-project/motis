#pragma once

#include <thread>

#include "boost/asio/io_service.hpp"

#include "ctx/ctx.h"

#include "motis/module/dispatcher.h"
#include "motis/module/registry.h"

namespace motis::module {

struct controller : public dispatcher, public registry {
  explicit controller(
      std::vector<std::unique_ptr<motis::module::module>>&& modules)
      : dispatcher{*static_cast<registry*>(this), std::move(modules)} {}

  template <typename Fn>
  auto run(Fn f, ctx::access_t const access = ctx::access_t::READ,
           unsigned num_threads = std::thread::hardware_concurrency()) ->
      typename std::enable_if_t<!std::is_same_v<void, decltype(f())>,
                                decltype(f())> {
    decltype(f()) result;
    std::exception_ptr eptr;

    access == ctx::access_t::READ ? enqueue_read_io(
                                        ctx_data(access, this, sched_),
                                        [&]() {
                                          try {
                                            result = f();
                                          } catch (...) {
                                            eptr = std::current_exception();
                                          }
                                        },
                                        ctx::op_id(CTX_LOCATION))
                                  : enqueue_write_io(
                                        ctx_data(access, this, sched_),
                                        [&]() {
                                          try {
                                            result = f();
                                          } catch (...) {
                                            eptr = std::current_exception();
                                          }
                                        },
                                        ctx::op_id(CTX_LOCATION));
    runner_.run(num_threads);

    if (eptr) {
      std::rethrow_exception(eptr);
    }

    return result;
  }

  template <typename Fn>
  auto run(Fn f, ctx::access_t const access = ctx::access_t::READ,
           unsigned num_threads = std::thread::hardware_concurrency()) ->
      typename std::enable_if_t<std::is_same_v<void, decltype(f())>> {
    std::exception_ptr eptr;

    access == ctx::access_t::READ ? enqueue_read_io(
                                        ctx_data(access, this, sched_),
                                        [&]() {
                                          try {
                                            f();
                                          } catch (...) {
                                            eptr = std::current_exception();
                                          }
                                        },
                                        ctx::op_id(CTX_LOCATION))
                                  : enqueue_write_io(
                                        ctx_data(access, this, sched_),
                                        [&]() {
                                          try {
                                            f();
                                          } catch (...) {
                                            eptr = std::current_exception();
                                          }
                                        },
                                        ctx::op_id(CTX_LOCATION));
    runner_.run(num_threads);

    if (eptr) {
      std::rethrow_exception(eptr);
    }
  }
};

}  // namespace motis::module
