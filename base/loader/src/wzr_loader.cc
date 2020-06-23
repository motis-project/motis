#include "motis/loader/wzr_loader.h"

#include <fstream>
#include <istream>
#include <mutex>
#include <vector>

#include "boost/filesystem.hpp"

#include "utl/parallel_for.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/parser/mmap_reader.h"
#include "utl/pipes/map.h"
#include "utl/pipes/transform.h"

#include "motis/core/common/logging.h"

using namespace motis::logging;

namespace motis::loader {

struct class_mapping_entry {
  utl::csv_col<utl::cstr, UTL_NAME("name")> name_;
  utl::csv_col<int, UTL_NAME("class")> class_;
};

waiting_time_rules load_waiting_time_rules(
    std::string const& wzr_classes_path, std::string const& wzr_matrix_path,
    mcd::vector<mcd::unique_ptr<category>> const& category_ptrs) {
  waiting_time_rules rules;

  std::vector<int> waiting_times;
  if (!wzr_classes_path.empty() && !wzr_matrix_path.empty()) {
    rules.category_map_ =
        utl::line_range<utl::mmap_reader>{
            utl::mmap_reader{wzr_classes_path.c_str()}}  //
        | utl::csv<class_mapping_entry>()  //
        | utl::transform([&](class_mapping_entry const& e) {
            rules.default_group_ =
                std::max(e.class_.val() + 1, rules.default_group_);
            return mcd::pair{mcd::string{e.name_.val().view()}, e.class_.val()};
          })  //
        | utl::emplace<decltype(rules.category_map_)>();

    try {
      auto entry = 0;
      std::ifstream in;
      in.exceptions(std::ifstream::failbit | std::ifstream::badbit);
      in.open(wzr_matrix_path.c_str());
      while (!in.eof() && in.peek() != EOF) {
        if (in.peek() == '\n' || in.peek() == '\r') {
          in.get();
          continue;
        }
        in >> entry;
        waiting_times.emplace_back(entry);
      }
    } catch (std::exception const& e) {
      std::clog << boost::filesystem::current_path() << "\n";
      LOG(error) << "exception reading wzr matrix file " << wzr_matrix_path
                 << ": " << e.what();
      utl::verify(false, "unable to open wzr matrix file {}", wzr_matrix_path);
    }

    utl::verify(
        waiting_times.size() == rules.default_group_ * rules.default_group_,
        "wzr loader: number of groups {} squared does not match matrix size {}",
        rules.default_group_, waiting_times.size());
  } else {
    rules.default_group_ = 1;
    for (auto i = 0;
         i <= (rules.default_group_ + 1) * (rules.default_group_ + 1); ++i) {
      waiting_times.emplace_back(0);
    }
  }

  // category/group numbers are 1-based.
  // initializes all values to false.
  auto const number_of_groups = rules.default_group_;
  rules.waits_for_other_trains_.resize(number_of_groups + 1);
  rules.other_trains_wait_for_.resize(number_of_groups + 1);
  rules.waiting_time_matrix_ = make_flat_matrix<duration>(number_of_groups + 1);

  for (int i = 0; i < number_of_groups * number_of_groups; i++) {
    int connecting_cat = i / number_of_groups + 1;
    int feeder_cat = i % number_of_groups + 1;
    int waiting_time = waiting_times[i];

    rules.waiting_time_matrix_[connecting_cat][feeder_cat] = waiting_time;

    if (waiting_time > 0) {
      rules.waits_for_other_trains_[connecting_cat] = true;
      rules.other_trains_wait_for_[feeder_cat] = true;
    }
  }

  rules.family_to_wtr_category_.resize(category_ptrs.size());
  for (size_t i = 0; i < category_ptrs.size(); i++) {
    rules.family_to_wtr_category_[i] =
        rules.waiting_time_category(category_ptrs[i]->name_);
  }

  return rules;
}

void collect_events(station_node const* st,
                    std::vector<ev_key>& waits_for_other_trains,
                    std::vector<ev_key>& other_trains_wait_for,
                    waiting_time_rules const& wtr) {
  auto const collect = [&](edge const& e, event_type const ev_type) {
    for (auto lcon_idx = lcon_idx_t{};
         lcon_idx < e.m_.route_edge_.conns_.size(); ++lcon_idx) {
      auto const& lc = e.m_.route_edge_.conns_[lcon_idx];
      if (ev_type == event_type::DEP &&
          wtr.waits_for_other_trains(
              wtr.waiting_time_category(lc.full_con_->con_info_->family_))) {
        waits_for_other_trains.emplace_back(ev_key{&e, lcon_idx, ev_type});
      }
      if (ev_type == event_type::ARR &&
          wtr.other_trains_wait_for(
              wtr.waiting_time_category(lc.full_con_->con_info_->family_))) {
        other_trains_wait_for.emplace_back(ev_key{&e, lcon_idx, ev_type});
      }
    }
  };

  st->for_each_route_node([&](node const* n) {
    for (auto const& e : n->edges_) {
      if (e.type() == edge::ROUTE_EDGE) {
        collect(e, event_type::DEP);
      }
    }
    for (auto const& ie : n->incoming_edges_) {
      if (ie->type() == edge::ROUTE_EDGE) {
        collect(*ie, event_type::ARR);
      }
    }
  });
}

void add_dependencies(schedule& sched,
                      std::vector<ev_key> const& waits_for_other_trains,
                      std::vector<ev_key> const& other_trains_wait_for,
                      duration planned_transfer_delta, std::mutex& mutex) {
  std::vector<std::pair<ev_key, ev_key>> entries;
  auto const& wtr = sched.waiting_time_rules_;
  for (auto const& feeder : other_trains_wait_for) {
    for (auto const& connector : waits_for_other_trains) {
      if (feeder.lcon()->trips_ == connector.lcon()->trips_) {
        continue;
      }
      auto const feeder_time = feeder.get_time();
      auto const connector_time = connector.get_time();
      auto const transfer_time =
          sched.stations_[feeder.get_station_idx()]->transfer_time_;
      if (connector_time < feeder_time + transfer_time ||
          connector_time > feeder_time + planned_transfer_delta) {
        continue;
      }
      auto const waiting_time = wtr.waiting_time_family(
          connector.lcon()->full_con_->con_info_->family_,
          feeder.lcon()->full_con_->con_info_->family_);
      if (waiting_time == 0) {
        continue;
      }
      entries.emplace_back(feeder, connector);
    }
  }
  if (!entries.empty()) {
    std::lock_guard guard{mutex};
    for (auto const& [feeder, connector] : entries) {
      sched.waits_for_trains_[connector].push_back(feeder);
      sched.trains_wait_for_[feeder].push_back(connector);
    }
  }
}

void calc_waits_for(schedule& sched, duration planned_transfer_delta) {
  scoped_timer timer("calculating waiting time rule dependencies");
  std::mutex mutex;
  utl::parallel_for(sched.station_nodes_, [&](auto const& st) {
    std::vector<ev_key> waits_for_other_trains;
    std::vector<ev_key> other_trains_wait_for;
    collect_events(st.get(), waits_for_other_trains, other_trains_wait_for,
                   sched.waiting_time_rules_);

    add_dependencies(sched, waits_for_other_trains, other_trains_wait_for,
                     planned_transfer_delta, mutex);
  });
}

}  // namespace motis::loader
