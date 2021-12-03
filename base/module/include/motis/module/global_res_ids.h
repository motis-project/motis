#pragma once

#include <type_traits>

namespace motis::module {

enum class global_res_id { SCHEDULE, PAX_DATA, PATH_DATA, FIRST_FREE_RES_ID };

constexpr inline ctx::res_id_t to_res_id(global_res_id const i) {
  return static_cast<std::underlying_type_t<global_res_id>>(i);
}

}  // namespace motis::module