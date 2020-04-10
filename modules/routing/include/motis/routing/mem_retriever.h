#pragma once

#include <algorithm>
#include <mutex>
#include <vector>

#include "motis/routing/mem_manager.h"

namespace motis::routing {

struct memory {
  explicit memory(std::size_t bytes) : in_use_(false), mem_(bytes) {}
  bool in_use_;
  mem_manager mem_;
};

struct mem_retriever {
  mem_retriever(std::mutex& mutex,
                std::vector<std::unique_ptr<memory>>& mem_pool,
                std::size_t bytes)
      : mutex_(mutex), memory_(retrieve(mem_pool, bytes)) {}

  mem_retriever(mem_retriever const&) = delete;
  mem_retriever& operator=(mem_retriever const&) = delete;

  mem_retriever(mem_retriever&&) = delete;
  mem_retriever& operator=(mem_retriever&&) = delete;

  ~mem_retriever() {
    std::lock_guard<std::mutex> lock(mutex_);
    memory_->in_use_ = false;
    memory_->mem_.reset();
  }

  mem_manager& get() { return memory_->mem_; }

private:
  memory* retrieve(std::vector<std::unique_ptr<memory>>& mem_pool,
                   std::size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find_if(begin(mem_pool), end(mem_pool),
                           [](auto&& m) { return !m->in_use_; });
    if (it == end(mem_pool)) {
      mem_pool.emplace_back(std::make_unique<memory>(bytes));
      mem_pool.back()->in_use_ = true;
      return mem_pool.back().get();
    }
    it->get()->in_use_ = true;
    return it->get();
  }

  std::mutex& mutex_;
  memory* memory_;
};

}  // namespace motis::routing
