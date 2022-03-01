#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#include "utl/erase.h"
#include "utl/verify.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/serialization.h"
#include "motis/module/global_res_ids.h"
#include "motis/module/module.h"

#include "motis/paxmon/error.h"
#include "motis/paxmon/universe.h"
#include "motis/paxmon/universe_access.h"

namespace motis::paxmon {

struct multiverse {
  explicit multiverse(motis::module::module& mod) : mod_{mod} {}

  void create_default_universe() {
    std::lock_guard lock{mutex_};
    utl::verify(universe_res_map_.find(0) == end(universe_res_map_),
                "default paxmon universe already exists");
    auto const default_schedule_res_id =
        motis::module::to_res_id(motis::module::global_res_id::SCHEDULE);
    auto const default_uv_res_id = motis::module::to_res_id(
        motis::module::global_res_id::PAX_DEFAULT_UNIVERSE);
    auto uvp = std::make_unique<universe>();
    uvp->schedule_res_id_ = default_schedule_res_id;
    mod_.add_shared_data(default_uv_res_id, std::move(uvp));
    universe_res_map_[0] = default_uv_res_id;
    schedule_res_map_[0] = default_schedule_res_id;
    universes_using_schedule_[default_schedule_res_id].emplace_back(0);
  }

  universe_access get(
      universe_id const id,
      ctx::access_t const universe_access = ctx::access_t::READ,
      ctx::access_t const schedule_access = ctx::access_t::READ) {
    std::unique_lock lock{mutex_};
    if (auto const it = universe_res_map_.find(id);
        it != end(universe_res_map_)) {
      auto const uv_res_id = it->second;
      auto const schedule_res_id = schedule_res_map_.at(id);
      lock.unlock();
      auto res_lock = mod_.lock_resources(
          {{uv_res_id, universe_access}, {schedule_res_id, schedule_access}});
      auto const& sched =
          *res_lock.get<schedule_data>(schedule_res_id).schedule_;
      auto& uv = *res_lock.get<std::unique_ptr<universe>>(uv_res_id);
      return {std::move(res_lock), sched, uv};
    } else {
      throw std::system_error{error::universe_not_found};
    }
  }

  universe* fork(universe const& base_uv, schedule const& base_sched,
                 bool fork_schedule) {
    std::lock_guard lock{mutex_};
    auto const new_id = ++last_id_;
    auto const new_uv_res_id = mod_.generate_res_id();
    auto new_schedule_res_id = base_uv.schedule_res_id_;
    if (fork_schedule) {
      new_schedule_res_id = mod_.generate_res_id();
      mod_.add_shared_data(new_schedule_res_id, copy_graph(base_sched));
    }
    auto new_uvp = std::make_unique<universe>(base_uv);
    new_uvp->id_ = new_id;
    new_uvp->schedule_res_id_ = new_schedule_res_id;
    auto const new_uv = new_uvp.get();
    mod_.add_shared_data(new_uv_res_id, std::move(new_uvp));
    universe_res_map_[new_id] = new_uv_res_id;
    schedule_res_map_[new_id] = new_schedule_res_id;
    universes_using_schedule_[new_schedule_res_id].emplace_back(new_id);
    return new_uv;
  }

  bool destroy(universe_id const id) {
    std::lock_guard lock{mutex_};
    if (id == 0) {
      return false;
    }
    if (auto const it = universe_res_map_.find(id);
        it != end(universe_res_map_)) {
      auto const uv_res_id = it->second;
      auto const schedule_res_id = schedule_res_map_.at(id);
      utl::verify(
          uv_res_id != motis::module::to_res_id(
                           motis::module::global_res_id::PAX_DEFAULT_UNIVERSE),
          "paxmon::multiverse.destroy: default universe");
      mod_.remove_shared_data(uv_res_id);
      auto& sched_refs = universes_using_schedule_[schedule_res_id];
      utl::erase(sched_refs, id);
      if (sched_refs.empty()) {
        utl::verify(
            schedule_res_id != motis::module::to_res_id(
                                   motis::module::global_res_id::SCHEDULE),
            "paxmon::multiverse.destroy: default schedule");
        mod_.remove_shared_data(schedule_res_id);
      }
      // TODO(pablo): mappings are never removed
      return true;
    } else {
      return false;
    }
  }

  std::vector<universe_id> universes_using_schedule(
      ctx::res_id_t const schedule_res_id) {
    std::lock_guard lock{mutex_};
    if (auto const it = universes_using_schedule_.find(schedule_res_id);
        it != end(universes_using_schedule_)) {
      return it->second;
    } else {
      return {};
    }
  }

  std::recursive_mutex mutex_;
  motis::module::module& mod_;
  std::map<universe_id, ctx::res_id_t> universe_res_map_;
  std::map<universe_id, ctx::res_id_t> schedule_res_map_;
  std::map<ctx::res_id_t, std::vector<universe_id>> universes_using_schedule_;
  universe_id last_id_{};
};

}  // namespace motis::paxmon
