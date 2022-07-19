#pragma once

#include <chrono>
#include <optional>

#include "ctx/res_id_t.h"

#include "motis/paxmon/universe_id.h"

namespace motis::paxmon {

struct multiverse;

struct universe_info {
  universe_info(multiverse& mv, universe_id const uv_id,
                ctx::res_id_t const universe_res,
                ctx::res_id_t const schedule_res)
      : uv_id_{uv_id},
        universe_res_{universe_res},
        schedule_res_{schedule_res},
        multiverse_{mv} {}

  ~universe_info();

  universe_info(universe_info const&) = delete;
  universe_info& operator=(universe_info const&) = delete;

  universe_id const uv_id_;
  ctx::res_id_t const universe_res_;
  ctx::res_id_t const schedule_res_;
  std::optional<std::chrono::seconds> ttl_{};
  std::optional<std::chrono::time_point<std::chrono::steady_clock>>
      keep_alive_until_{};
  multiverse& multiverse_;
};

}  // namespace motis::paxmon
