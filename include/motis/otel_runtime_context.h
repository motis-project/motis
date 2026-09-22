#pragma once

#include <memory>

#include "utl/helpers/algorithm.h"

#include "opentelemetry/context/context.h"
#include "opentelemetry/context/runtime_context.h"

#include "ctx/operation.h"

#include "motis/ctx_data.h"

namespace motis {

struct otel_runtime_context_storage
    : public opentelemetry::context::RuntimeContextStorage {
  opentelemetry::context::Context GetCurrent() noexcept override {
    auto const op = ctx::current_op<ctx_data>();

    if (op == nullptr) {
      return default_storage_->GetCurrent();
    }

    // How to get stack of contexts from op (operation<ctx_data>*)
    auto& stack = op->data_.otel_context_stack_;
    return stack.empty() ? opentelemetry::context::Context{} : stack.back();
  }

  opentelemetry::nostd::unique_ptr<opentelemetry::context::Token> Attach(
      opentelemetry::context::Context const& context) noexcept override {
    auto const op = ctx::current_op<ctx_data>();

    if (op == nullptr) {
      return default_storage_->Attach(context);
    }

    op->data_.otel_context_stack_.push_back(context);
    return CreateToken(context);
  }

  bool Detach(opentelemetry::context::Token& token) noexcept override {
    auto const op = ctx::current_op<ctx_data>();

    if (op == nullptr) {
      return default_storage_->Detach(token);
    }

    auto& stack = op->data_.otel_context_stack_;

    if (utl::find(stack, token) == stack.end()) {
      return false;
    }

    while (!(token == stack.back())) {
      stack.pop_back();
    }
    stack.pop_back();
    return true;
  }

private:
  std::unique_ptr<opentelemetry::context::ThreadLocalContextStorage>
      default_storage_{std::make_unique<
          opentelemetry::context::ThreadLocalContextStorage>()};
};

}  // namespace motis