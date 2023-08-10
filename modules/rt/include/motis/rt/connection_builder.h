#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>

#include "utl/get_or_create.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/track_access.h"
#include "motis/loader/classes.h"

namespace motis::rt {

inline std::uint32_t get_family(schedule& sched,
                                std::string_view const cat_name) {
  auto const it = std::find_if(
      begin(sched.categories_), end(sched.categories_),
      [&cat_name](auto const& cat) { return cat->name_ == cat_name; });
  if (it == end(sched.categories_)) {
    sched.categories_.emplace_back(mcd::make_unique<category>(
        mcd::string{cat_name}, static_cast<uint8_t>(0U)));
    return static_cast<std::uint32_t>(sched.categories_.size() - 1);
  } else {
    return static_cast<std::uint32_t>(
        std::distance(begin(sched.categories_), it));
  }
}

inline connection_info const* get_con_info(
    schedule& sched,
    std::map<connection_info, connection_info const*>& con_infos,
    std::string const& category, std::string const& line_id, int train_nr) {
  connection_info con_info;
  con_info.family_ = get_family(sched, category);
  con_info.line_identifier_ = line_id;
  con_info.train_nr_ = train_nr;

  return utl::get_or_create(con_infos, con_info, [&sched, &con_info]() {
    sched.connection_infos_.emplace_back(
        mcd::make_unique<connection_info>(con_info));
    return sched.connection_infos_.back().get();
  });
}

inline provider const* get_or_insert_provider(
    schedule& sched, std::string_view const provider_short_name,
    std::string_view const provider_full_name) {
  if (auto const it = sched.provider_by_full_name_.find(provider_full_name);
      it != end(sched.provider_by_full_name_)) {
    return it->second;
  } else {
    auto const ptr =
        sched.providers_
            .emplace_back(mcd::make_unique<provider>(
                provider_short_name, provider_full_name, provider_full_name))
            .get();
    sched.provider_by_full_name_[ptr->full_name_] = ptr;
    return ptr;
  }
}

inline connection_info const* get_con_info_with_provider(
    schedule& sched,
    std::map<connection_info, connection_info const*>& con_infos,
    std::string_view const category, std::string_view const line_id,
    std::uint32_t train_nr, std::string_view const provider_short_name,
    std::string_view const provider_full_name) {
  auto const con_info =
      connection_info{.line_identifier_ = line_id,
                      .provider_ = get_or_insert_provider(
                          sched, provider_short_name, provider_full_name),
                      .family_ = get_family(sched, category),
                      .train_nr_ = train_nr};

  return utl::get_or_create(con_infos, con_info, [&sched, &con_info]() {
    sched.connection_infos_.emplace_back(
        mcd::make_unique<connection_info>(con_info));
    return sched.connection_infos_.back().get();
  });
}

inline uint16_t get_track(schedule& sched, std::string_view const track_name) {
  auto const track = get_track_index(sched, track_name);
  if (track.has_value()) {
    return track.value();
  } else {
    sched.tracks_.emplace_back(track_name);
    return static_cast<uint16_t>(sched.tracks_.size() - 1);
  }
}

inline service_class get_clasz(mcd::string const& category) {
  static auto const clasz_map = loader::class_mapping();
  auto const it = clasz_map.find(category);
  return it == end(clasz_map) ? service_class::OTHER : it->second;
}

inline connection* get_full_con(schedule& sched,
                                connection_info const* con_info,
                                uint16_t dep_track, uint16_t arr_track) {
  connection c;
  c.con_info_ = con_info;
  c.d_track_ = dep_track;
  c.a_track_ = arr_track;
  c.clasz_ = get_clasz(sched.categories_[con_info->family_]->name_);
  sched.full_connections_.emplace_back(mcd::make_unique<connection>(c));
  return sched.full_connections_.back().get();
}

inline connection* get_full_con(
    schedule& sched,
    std::map<connection_info, connection_info const*> con_infos,
    std::string const& dep_track, std::string const& arr_track,
    std::string const& category, std::string const& line_id, int train_nr) {
  connection c;
  c.con_info_ = get_con_info(sched, con_infos, category, line_id, train_nr);
  c.d_track_ = get_track(sched, dep_track);
  c.a_track_ = get_track(sched, arr_track);
  c.clasz_ = get_clasz(category);
  sched.full_connections_.emplace_back(mcd::make_unique<connection>(c));
  return sched.full_connections_.back().get();
}

}  // namespace motis::rt
