#include "motis/import/task.h"

#include <cassert>
#include <iostream>

#include "motis/clog_redirect.h"

#include "utl/helpers/algorithm.h"
#include "utl/progress_tracker.h"

namespace motis {

namespace fs = std::filesystem;

task::task(std::string_view const name,
           fs::path const& data_path,
           config const& c,
           meta_t hashes)
    : hashes_{std::move(hashes)}, data_path_{data_path}, c_{c}, name_{name} {}

task::~task() = default;

void task::add_dependency(
    std::initializer_list<std::reference_wrapper<task>> const deps) {
  for (auto const dep : deps) {
    in_.push_back(&dep.get());
    dep.get().out_.push_back(this);
  }
}

std::string_view task::name() const { return name_; }

bool task::is_done() const { return done_; }

bool task::is_ready_to_run() const {
  return utl::all_of(in_, [](task const* t) { return t->is_done(); });
}

bool task::can_load() const {
  auto const existing = read_hashes(data_path_, name_);
  if (existing != hashes_) {
    std::cout << name_ << "\n"
              << "  existing: " << to_str(existing) << "\n"
              << "  current: " << to_str(hashes_) << "\n";
  }
  return existing == hashes_;
}

void task::exec() {
  assert(is_enabled());

  auto const pt = utl::activate_progress_tracker(name_);
  auto const redirect = clog_redirect{
      (data_path_ / "logs" / (name_ + ".txt")).generic_string().c_str()};
  run();
  write_hashes(data_path_, name_, hashes_);
  pt->out_ = 100;
  pt->status("FINISHED");
  done_ = true;
}

}  // namespace motis
