#include "motis/paxmon/multiverse.h"

#include "ctx/operation.h"

#include "utl/erase.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/module/context/motis_publish.h"

using namespace motis::logging;
using namespace motis::module;

namespace motis::paxmon {

universe* multiverse::create_default_universe() {
  std::lock_guard lock{mutex_};
  utl::verify(universe_info_map_.find(0) == end(universe_info_map_),
              "default paxmon universe already exists");
  auto const default_schedule_res_id =
      motis::module::to_res_id(motis::module::global_res_id::SCHEDULE);
  auto const default_uv_res_id = motis::module::to_res_id(
      motis::module::global_res_id::PAX_DEFAULT_UNIVERSE);
  auto uvp = std::make_unique<universe>();
  uvp->schedule_res_id_ = default_schedule_res_id;
  auto* uv = uvp.get();
  mod_.add_shared_data(default_uv_res_id, std::move(uvp));
  universe_info_storage_[0] = std::make_shared<universe_info>(
      shared_from_this(), 0, default_uv_res_id, default_schedule_res_id);
  universe_info_map_[0] = universe_info_storage_[0];
  universes_using_schedule_[default_schedule_res_id].emplace_back(0);
  return uv;
}

universe_access multiverse::get(universe_id const id,
                                ctx::access_t const universe_access,
                                ctx::access_t const schedule_access) {
  std::unique_lock lock{mutex_};
  if (auto uv_info = get_universe_info(id); uv_info) {
    auto const uv_res_id = uv_info->universe_res_;
    auto const schedule_res_id = uv_info->schedule_res_;
    keep_alive(*uv_info);
    lock.unlock();
    auto res_lock = mod_.lock_resources(
        {{uv_res_id, universe_access}, {schedule_res_id, schedule_access}});
    auto const& sched = *res_lock.get<schedule_data>(schedule_res_id).schedule_;
    auto& uv = *res_lock.get<std::unique_ptr<universe>>(uv_res_id);
    return {std::move(res_lock), sched, uv, std::move(uv_info)};
  } else {
    throw std::system_error{error::universe_not_found};
  }
}

universe* multiverse::fork(universe const& base_uv, schedule const& base_sched,
                           bool fork_schedule,
                           std::optional<std::chrono::seconds> ttl) {
  std::unique_lock lock{mutex_};
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
  auto uv_info = std::make_shared<universe_info>(
      shared_from_this(), new_id, new_uv_res_id, new_schedule_res_id);
  if (ttl.has_value() && ttl.value().count() != 0) {
    uv_info->ttl_ = ttl;
    uv_info->keep_alive_until_ = std::chrono::steady_clock::now() + ttl.value();
  }
  universe_info_map_[new_id] = uv_info;
  universe_info_storage_[new_id] = std::move(uv_info);
  universes_using_schedule_[new_schedule_res_id].emplace_back(new_id);

  lock.unlock();
  message_creator mc;
  mc.create_and_finish(
      MsgContent_PaxMonUniverseForked,
      CreatePaxMonUniverseForked(mc, base_uv.id_, new_uv->id_,
                                 new_uv->schedule_res_id_, fork_schedule)
          .Union(),
      "/paxmon/universe_forked");
  auto const msg = make_msg(mc);
  ctx::await_all(motis_publish(msg));

  return new_uv;
}

bool multiverse::destroy(universe_id const id) {
  std::unique_lock lock{mutex_};
  if (id == 0) {
    return false;
  }
  if (auto uv_info = get_universe_info(id); uv_info) {
    auto const uv_res_id = uv_info->universe_res_;

    utl::verify(
        uv_res_id != motis::module::to_res_id(
                         motis::module::global_res_id::PAX_DEFAULT_UNIVERSE),
        "paxmon::multiverse.destroy: default universe");

    universe_info_storage_.erase(id);

    lock.unlock();
    send_universe_destroyed_notifications();
    return true;
  } else {
    return false;
  }
}

std::vector<universe_id> multiverse::universes_using_schedule(
    ctx::res_id_t const schedule_res_id) {
  std::lock_guard lock{mutex_};
  if (auto const it = universes_using_schedule_.find(schedule_res_id);
      it != end(universes_using_schedule_)) {
    return it->second;
  } else {
    return {};
  }
}

std::shared_ptr<universe_info> multiverse::get_universe_info(universe_id id) {
  if (auto const it = universe_info_map_.find(id);
      it != end(universe_info_map_)) {
    return it->second.lock();
  } else {
    return nullptr;
  }
}

keep_alive_response multiverse::keep_alive(
    std::vector<universe_id> const& universes) {
  std::lock_guard lock{mutex_};
  keep_alive_response res;
  auto const now = std::chrono::steady_clock::now();
  for (auto const& id : universes) {
    if (auto uv_info = get_universe_info(id); uv_info) {
      res.found_.emplace_back(keep_alive_universe_info{
          id, uv_info->schedule_res_, keep_alive(*uv_info, now)});
    } else {
      res.not_found_.emplace_back(id);
    }
  }
  return res;
}

std::optional<std::chrono::seconds> multiverse::keep_alive(
    universe_info& uv_info) {
  return keep_alive(uv_info, std::chrono::steady_clock::now());
}

std::optional<std::chrono::seconds> multiverse::keep_alive(
    universe_info& uv_info,
    std::chrono::time_point<std::chrono::steady_clock> const now) {
  if (uv_info.ttl_) {
    auto const ttl = uv_info.ttl_.value();
    uv_info.keep_alive_until_ = now + ttl;
    return ttl;
  }
  return {};
}

void multiverse::destroy_expired_universes() {
  std::unique_lock lock{mutex_};
  auto const now = std::chrono::steady_clock::now();
  for (auto it = begin(universe_info_map_); it != end(universe_info_map_);) {
    auto const uv_id = it->first;
    auto uv_info = it->second.lock();
    if (!uv_info) {
      // clean up expired universes (already released)
      it = universe_info_map_.erase(it);
      continue;
    }
    if (uv_info->keep_alive_until_ &&
        uv_info->keep_alive_until_.value() < now) {
      // destroy universe (after last usage)
      universe_info_storage_.erase(uv_id);
      LOG(info) << "paxmon: universe " << uv_id << " expired";
    }
    ++it;
  }

  lock.unlock();
  send_universe_destroyed_notifications();
}

void multiverse::release_universe(universe_info& uv_info) {
  // called from universe_info destructor
  // entry already removed from universe_info_storage_
  // universe_info_map_ entry will be removed in destroy_expired_universes later
  auto const uv_id = uv_info.uv_id_;
  auto const uv_res_id = uv_info.universe_res_;
  auto const schedule_res_id = uv_info.schedule_res_;

  std::unique_lock lock{mutex_};
  mod_.remove_shared_data(uv_res_id);
  auto& sched_refs = universes_using_schedule_[schedule_res_id];
  utl::erase(sched_refs, uv_id);
  if (sched_refs.empty() &&
      schedule_res_id !=
          motis::module::to_res_id(motis::module::global_res_id::SCHEDULE)) {
    mod_.remove_shared_data(schedule_res_id);
  }
  recently_destroyed_universes_.emplace_back(uv_id);
}

std::vector<current_universe_info> multiverse::get_current_universe_infos() {
  auto infos = std::vector<current_universe_info>{};

  std::unique_lock lock{mutex_};
  auto const now = std::chrono::steady_clock::now();
  for (auto const& [id, uvi] : universe_info_storage_) {
    std::optional<std::chrono::seconds> expires_in;
    if (uvi->keep_alive_until_) {
      expires_in = std::chrono::duration_cast<std::chrono::seconds>(
          *uvi->keep_alive_until_ - now);
    }
    infos.emplace_back(current_universe_info{
        uvi->uv_id_, uvi->universe_res_, uvi->schedule_res_, uvi->ttl_,
        uvi->keep_alive_until_, expires_in});
  }

  return infos;
}

void multiverse::send_universe_destroyed_notifications() {
  auto lock = std::unique_lock{mutex_};
  auto const recently_destroyed = recently_destroyed_universes_;
  recently_destroyed_universes_.clear();
  lock.unlock();

  if (ctx::current_op<ctx_data>() != nullptr) {
    for (auto const& uv_id : recently_destroyed) {
      message_creator mc;
      mc.create_and_finish(MsgContent_PaxMonUniverseDestroyed,
                           CreatePaxMonUniverseDestroyed(mc, uv_id).Union(),
                           "/paxmon/universe_destroyed");
      auto const msg = make_msg(mc);
      ctx::await_all(motis_publish(msg));
    }
  }
}

}  // namespace motis::paxmon
