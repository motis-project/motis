#pragma once

#include <algorithm>
#include <memory>

#include "opentelemetry/context/context.h"
#include "opentelemetry/context/runtime_context.h"

#include "ctx/operation.h"

namespace motis::module {

struct otel_runtime_context_storage
    : public opentelemetry::context::RuntimeContextStorage {
  opentelemetry::context::Context GetCurrent() noexcept override {
    if (auto op = ctx::current_op<ctx_data>(); op != nullptr) {
      auto& stack = op->data_.otel_context_stack_;
      return stack.empty() ? opentelemetry::context::Context{} : stack.back();
    } else {
      return default_storage_->GetCurrent();
    }
  }

  opentelemetry::nostd::unique_ptr<opentelemetry::context::Token> Attach(
      opentelemetry::context::Context const& context) noexcept override {
    if (auto op = ctx::current_op<ctx_data>(); op != nullptr) {
      op->data_.otel_context_stack_.push_back(context);
      return CreateToken(context);
    } else {
      return default_storage_->Attach(context);
    }
  }

  bool Detach(opentelemetry::context::Token& token) noexcept override {
    if (auto op = ctx::current_op<ctx_data>(); op != nullptr) {
      auto& stack = op->data_.otel_context_stack_;

      if (stack.empty()) {
        return false;
      }

      if (token == stack.back()) {
        stack.pop_back();
        return true;
      }

      if (std::find_if(begin(stack), end(stack), [&token](auto const& context) {
            return token == context;
          }) == end(stack)) {
        return false;
      }

      while (!(token == stack.back())) {
        stack.pop_back();
      }

      stack.pop_back();
      return true;
    } else {
      return default_storage_->Detach(token);
    }
  }

private:
  std::unique_ptr<opentelemetry::context::ThreadLocalContextStorage>
      default_storage_{std::make_unique<
          opentelemetry::context::ThreadLocalContextStorage>()};
};

}  // namespace motis::module
