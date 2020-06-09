#include "motis/rt/rt.h"

#include <fstream>
#include <ostream>

#include "boost/program_options.hpp"

#include "motis/core/schedule/serialization.h"
#include "motis/module/context/get_schedule.h"

#include "motis/rt/rt_handler.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/service_access.h"
#include "motis/core/access/trip_iterator.h"

namespace motis::rt {

rt::rt() : module("RT", "rt") {
  param(validate_graph_, "validate_graph",
        "validate routing graph after every rt update");
  param(validate_constant_graph_, "validate_constant_graph",
        "validate constant graph after every rt update");
}

rt::~rt() = default;

void rt::init(motis::module::registry& reg) {
  handler_ = std::make_unique<rt_handler>(
      *get_shared_data_mutable<schedule_data>(SCHEDULE_DATA_KEY).schedule_,
      validate_graph_, validate_constant_graph_);

  reg.subscribe(
      "/ris/messages",
      [&](motis::module::msg_ptr const& msg) { return handler_->update(msg); },
      ctx::access_t::WRITE);
  reg.register_op(
      "/rt/single",
      [&](motis::module::msg_ptr const& msg) { return handler_->single(msg); },
      ctx::access_t::WRITE);
  reg.subscribe(
      "/ris/system_time_changed",
      [&](motis::module::msg_ptr const& msg) {
        handler_->flush(msg);
        return nullptr;
      },
      ctx::access_t::WRITE);
  reg.register_op("/rt/dump", [&](motis::module::msg_ptr const& msg) {
    auto const m = motis_content(RtWriteGraphRequest, msg);
    write_graph(m->path()->str(), motis::module::get_schedule());
    return motis::module::msg_ptr{};
  });

  reg.register_op("/rt/dump_txt", [&](motis::module::msg_ptr const&) {
    std::ofstream f{"delays.csv"};

    // ESSENTIAL INFO
    f << "first_stop;first_dep_time;first_train_nr;";  // ID
    f << "segment_from_stop;segment_to_stop;";  // SEGMENT
    f << "schedule_departure_time;real_arrival_time;";  // DEPARTURE
    f << "schedule_arrival_time;real_arrival_time;";  // ARRIVAL

    // ADDITIONAL INFO
    f << "train_name;line;";  // ADDITIONAL INFO TRAIN
    f << "from_name;from_lat;from_lng;";  // ADDITIONAL INFO FROM
    f << "to_name;to_lat;to_lng;\n";  // ADDITIONAL INFO TO

    auto const& sched = get_sched();
    for (auto const& [id, t] : sched.trips_) {
      auto const first_fcon =
          t->edges_->front()->m_.route_edge_.conns_.at(t->lcon_idx_).full_con_;
      if (first_fcon->clasz_ >= MOTIS_RE) {
        continue;
      }

      for (auto const& s : motis::access::sections(t)) {
        auto const from_di = get_delay_info(sched, s.ev_key_from());
        auto const to_di = get_delay_info(sched, s.ev_key_to());
        auto const& from = s.from_station(sched);
        auto const& to = s.to_station(sched);
        f  // ID
            << sched.stations_.at(id.station_id_)->eva_nr_ << ";"
            << motis_to_unixtime(sched, id.time_) << ";" << id.train_nr_
            << ";"
            // SEGMENT
            << from.eva_nr_ << ";" << to.eva_nr_
            << ";"
            // DEPARTURE
            << motis_to_unixtime(sched, from_di.get_schedule_time()) << ";"
            << motis_to_unixtime(sched, from_di.get_current_time())
            << ";"
            // ARRIVAL
            << motis_to_unixtime(sched, to_di.get_schedule_time()) << ";"
            << motis_to_unixtime(sched, to_di.get_current_time())
            << ";"
            // ADDITIONAL
            << get_service_name(sched, first_fcon->con_info_) << ";"  // NAME
            << first_fcon->con_info_->line_identifier_ << ";"  // LINE
            << from.name_ << ";" << from.lat() << ";" << from.lng() << ";"  // F
            << to.name_ << ";" << to.lat() << ";" << to.lng() << "\n";  // TO
      }
    }
    return motis::module::msg_ptr{};
  });
}

}  // namespace motis::rt
