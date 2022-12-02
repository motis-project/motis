#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#include "ctx/res_id_t.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/serialization.h"
#include "motis/module/global_res_ids.h"
#include "motis/module/module.h"

#include "motis/paxmon/error.h"
#include "motis/paxmon/universe.h"
#include "motis/paxmon/universe_access.h"
#include "motis/paxmon/universe_info.h"

namespace motis::paxmon {

struct keep_alive_universe_info {
  universe_id id_{};
  ctx::res_id_t schedule_res_id_{};
  std::optional<std::chrono::seconds> expires_in_{};
};

struct keep_alive_response {
  std::vector<keep_alive_universe_info> found_;
  std::vector<universe_id> not_found_;
};

struct current_universe_info {
  universe_id const uv_id_;
  ctx::res_id_t const universe_res_;
  ctx::res_id_t const schedule_res_;
  std::optional<std::chrono::seconds> ttl_{};
  std::optional<std::chrono::time_point<std::chrono::steady_clock>>
      keep_alive_until_{};
  std::optional<std::chrono::seconds> expires_in_{};
};

struct multiverse : std::enable_shared_from_this<multiverse> {
  explicit multiverse(motis::module::module& mod)
      : mod_{mod},
        id_{std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count()} {}

  universe* create_default_universe();

  std::int64_t id() const { return id_; }

  universe_access get(universe_id id,
                      ctx::access_t universe_access = ctx::access_t::READ,
                      ctx::access_t schedule_access = ctx::access_t::READ);

  universe* fork(universe const& base_uv, schedule const& base_sched,
                 bool fork_schedule, std::optional<std::chrono::seconds> ttl);

  bool destroy(universe_id const id);

  std::vector<universe_id> universes_using_schedule(
      ctx::res_id_t schedule_res_id);

  keep_alive_response keep_alive(std::vector<universe_id> const& universes);

  void destroy_expired_universes();

  std::vector<current_universe_info> get_current_universe_infos();

  friend universe_info;

private:
  std::shared_ptr<universe_info> get_universe_info(universe_id id);

  static std::optional<std::chrono::seconds> keep_alive(universe_info& uv_info);

  static std::optional<std::chrono::seconds> keep_alive(
      universe_info& uv_info,
      std::chrono::time_point<std::chrono::steady_clock> now);

  void release_universe(universe_info& uv_info);

  void send_universe_destroyed_notifications();

  std::recursive_mutex mutex_;
  motis::module::module& mod_;
  std::int64_t const id_;
  std::map<universe_id, std::shared_ptr<universe_info>> universe_info_storage_;
  std::map<universe_id, std::weak_ptr<universe_info>> universe_info_map_;
  std::map<ctx::res_id_t, std::vector<universe_id>> universes_using_schedule_;
  std::vector<universe_id> recently_destroyed_universes_;
  universe_id last_id_{};
};

}  // namespace motis::paxmon
