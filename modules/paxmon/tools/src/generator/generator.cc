#include <algorithm>
#include <fstream>
#include <vector>

#include "boost/filesystem.hpp"

#include "conf/configuration.h"
#include "conf/options_parser.h"

#include "utl/progress_tracker.h"

#include "motis/bootstrap/dataset_settings.h"
#include "motis/bootstrap/import_settings.h"
#include "motis/bootstrap/module_settings.h"
#include "motis/bootstrap/motis_instance.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"

#include "motis/paxmon/build_graph.h"
#include "motis/paxmon/capacity.h"
#include "motis/paxmon/generate_capacities.h"
#include "motis/paxmon/get_load.h"
#include "motis/paxmon/loader/journeys/to_compact_journey.h"
#include "motis/paxmon/output/journey_converter.h"
#include "motis/paxmon/paxmon_data.h"
#include "motis/paxmon/tools/generator/query_generator.h"
#include "motis/paxmon/tools/groups/group_generator.h"

using namespace motis;
using namespace motis::bootstrap;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::paxmon;
using namespace motis::paxmon::tools::generator;
using namespace motis::paxmon::output;
using namespace motis::paxmon::tools::groups;

namespace fs = boost::filesystem;

struct generator_settings : public conf::configuration {
  generator_settings() : configuration{"Generator Settings"} {
    param(csv_journey_path_, "out,o", "Output CSV Journey file");
    param(capacity_path_, "capacity", "Capacity file (in/out)");
    param(generate_capacities_, "generate_capacities",
          "Generate capacity file (otherwise load existing file)");
    param(pax_count_, "pax_count", "Total number of passengers");
    param(group_size_mean_, "group_size_mean", "Group size mean");
    param(group_size_stddev_, "group_size_stddev",
          "Group size standard deviation");
    param(group_count_mean_, "group_count_mean", "Group count mean");
    param(group_count_stddev_, "group_count_stddev",
          "Group count standard deviation");
    param(max_load_, "max_load",
          "Max allowed trip load (e.g. 1.0 for 100%, set to 0.0 to disable "
          "check)");
    param(router_target_, "router_target", "Router target");
    param(num_threads_, "num_threads", "Number of worker threads");
  }

  std::string csv_journey_path_{"journeys.csv"};
  std::string capacity_path_{"capacities.csv"};
  unsigned num_threads_{std::thread::hardware_concurrency()};
  std::string router_target_;

  unsigned pax_count_{1000};

  double group_size_mean_{1.5};
  double group_size_stddev_{3.0};

  double group_count_mean_{2.0};
  double group_count_stddev_{10.0};

  bool generate_capacities_{true};
  double max_load_{0.0};
};

struct journey_generator {
  journey_generator(motis_instance& instance, paxmon_data& pmd,
                    generator_settings const& generator_opt,
                    group_generator& group_gen, bool check_load)
      : instance_{instance},
        sched_{instance_.sched()},
        pmd_{pmd},
        query_gen_{sched_},
        group_gen_{group_gen},
        generator_opt_{generator_opt},
        converter_{generator_opt.csv_journey_path_},
        check_load_{check_load},
        max_load_{generator_opt.max_load_} {
    converter_.writer_.enable_exceptions();
  }

  void run() {
    progress_tracker_ = utl::activate_progress_tracker("Journeys");
    progress_tracker_->in_high(generator_opt_.pax_count_);
    instance_.runner_.ios().post([&]() {
      for (auto i = 0U; i < generator_opt_.num_threads_ * 2; ++i) {
        if (!generate_next()) {
          break;
        }
      }
    });
    instance_.runner_.run(generator_opt_.num_threads_, false);
    progress_tracker_->status("FINISHED").show_progress(false);
  }

  unsigned generated_pax_count() const { return pax_generated_; }
  unsigned generated_group_count() const { return groups_generated_; }
  unsigned generated_routing_query_count() const { return routing_queries_; }
  unsigned different_journey_count() const { return different_journeys_; }
  unsigned over_capacity_skipped_count() const {
    return over_capacity_skipped_;
  }

private:
  bool generate_next() {
    if (pax_generated_ >= generator_opt_.pax_count_) {
      if (in_flight_ == 0) {
        instance_.runner_.ios().stop();
      }
      return false;
    }
    auto const request_msg =
        query_gen_.get_routing_request(generator_opt_.router_target_);
    ++in_flight_;
    ++routing_queries_;
    instance_.on_msg(request_msg,
                     instance_.runner_.ios().wrap(
                         [&](msg_ptr const& response_msg, std::error_code) {
                           --in_flight_;
                           pax_generated_ += handle_response(response_msg);
                           progress_tracker_->update(pax_generated_);
                           generate_next();
                         }));
    return true;
  }

  unsigned handle_response(msg_ptr const& response_msg) {
    auto const response = motis_content(RoutingResponse, response_msg);
    auto const journeys = message_to_journeys(response);

    auto generated = 0U;
    for (auto const& j : journeys) {
      auto const primary_id = next_primary_id_;
      auto const group_count = group_gen_.get_group_count();
      auto groups_added = 0U;
      for (auto secondary_id = 1ULL; secondary_id <= group_count;
           ++secondary_id) {
        auto const group_size = group_gen_.get_group_size();
        if (!check_journey_load(j, group_size, primary_id, secondary_id)) {
          break;
        }
        converter_.write_journey(j, primary_id, secondary_id, group_size);
        generated += group_size;
        ++groups_added;
      }
      if (groups_added > 0) {
        ++next_primary_id_;
        groups_generated_ += groups_added;
        ++different_journeys_;
      }
    }

    return generated;
  }

  bool check_journey_load(journey const& j, std::uint16_t group_size,
                          std::uint64_t primary_id,
                          std::uint64_t secondary_id) {
    if (!check_load_) {
      return true;
    }
    auto const cj = to_compact_journey(j, sched_);
    auto pg = pmd_.graph_.passenger_groups_.emplace_back(
        pmd_.graph_.passenger_group_allocator_.create(passenger_group{
            cj,
            static_cast<std::uint64_t>(pmd_.graph_.passenger_groups_.size()),
            data_source{primary_id, secondary_id}, group_size}));
    add_passenger_group_to_graph(sched_, pmd_, *pg);
    for (auto const e : pg->edges_) {
      if (e->has_capacity() &&
          get_base_load(e->get_pax_connection_info()) >
              static_cast<std::uint16_t>(e->capacity() * max_load_)) {
        remove_passenger_group_from_graph(pg);
        pmd_.graph_.passenger_groups_.pop_back();
        ++over_capacity_skipped_;
        return false;
      }
    }
    return true;
  }

  motis_instance& instance_;
  schedule const& sched_;
  paxmon_data& pmd_;
  query_generator query_gen_;
  group_generator& group_gen_;
  generator_settings const& generator_opt_;
  journey_converter converter_;
  utl::progress_tracker_ptr progress_tracker_;

  unsigned in_flight_{};
  unsigned pax_generated_{};
  unsigned groups_generated_{};
  unsigned routing_queries_{};
  unsigned different_journeys_{};
  unsigned over_capacity_skipped_{};
  std::uint64_t next_primary_id_{1};
  bool check_load_{};
  double max_load_{};
};

int main(int argc, char const** argv) {
  auto const routers = std::vector<std::string>{"tripbased", "csa", "routing"};
  auto const default_router_module = routers.front();
  auto const default_router_target = std::string{"/"} + default_router_module;

  generator_settings generator_opt;
  generator_opt.router_target_ = default_router_target;
  dataset_settings dataset_opt;
  dataset_opt.adjust_footpaths_ = true;
  import_settings import_opt;
  module_settings module_opt{routers};

  std::vector<conf::configuration*> confs = {&generator_opt, &dataset_opt,
                                             &import_opt, &module_opt};
  motis_instance instance;

  for (auto const& module : instance.modules()) {
    if (std::find(begin(routers), end(routers), module->module_name()) !=
        end(routers)) {
      confs.push_back(module);
    }
  }

  conf::options_parser parser(confs);
  parser.read_command_line_args(argc, argv);

  std::cout << "\n\tPassenger Journey Generator\n\n";

  if (parser.help()) {
    parser.print_help(std::cout);
    return 0;
  } else if (parser.version()) {
    return 0;
  }

  parser.read_configuration_file();

  module_opt.modules_ = {};
  module_opt.exclude_modules_ = {};
  for (auto const& router : routers) {
    if (generator_opt.router_target_.find(router) != std::string::npos) {
      module_opt.modules_.emplace_back(router);
    }
  }

  parser.print_used(std::cout);

  if (module_opt.modules_.empty()) {
    std::cout << "Unsupported router: " << generator_opt.router_target_
              << " (module not found)\n";
    return 1;
  }

  try {
    instance.import(module_opt, dataset_opt, import_opt);
    instance.init_modules(module_opt);
  } catch (std::exception const& e) {
    std::cout << "\nInitialization error: " << e.what() << "\n";
    return 1;
  }

  if (!instance.get_operation(generator_opt.router_target_)) {
    std::cout << "Unsupported router: " << generator_opt.router_target_
              << " (target not found)\n";
    return 1;
  }

  auto const check_load = generator_opt.max_load_ > 0.0;
  auto pmd = paxmon_data{};
  auto const& sched = instance.sched();

  if (check_load) {
    if (generator_opt.generate_capacities_) {
      std::cout << "Generating capacity information..." << std::endl;
      generate_capacities(sched, pmd, generator_opt.capacity_path_);
    }
    if (!fs::exists(generator_opt.capacity_path_)) {
      std::cout << "Capacity file not found: " << generator_opt.capacity_path_
                << " (try --generate_capacities 1)\n";
      return 1;
    }
    auto const entries_loaded =
        load_capacities(sched, generator_opt.capacity_path_,
                        pmd.trip_capacity_map_, pmd.category_capacity_map_);
    std::cout << "Loaded " << entries_loaded << " capacity entries"
              << std::endl;
    if (entries_loaded == 0) {
      std::cout
          << "No capacity information found! (try --generate_capacities 1)"
          << std::endl;
      return 1;
    }
  }

  std::cout << "\nGenerating journeys for at least " << generator_opt.pax_count_
            << " passengers..." << std::endl;

  group_generator group_gen{
      generator_opt.group_size_mean_, generator_opt.group_size_stddev_,
      generator_opt.group_count_mean_, generator_opt.group_count_stddev_};
  journey_generator journey_gen{instance, pmd, generator_opt, group_gen,
                                check_load};
  {
    auto bars = utl::global_progress_bars{};
    journey_gen.run();
  }

  std::cout << "\nGenerated " << journey_gen.generated_group_count()
            << " groups, " << journey_gen.generated_pax_count()
            << " passengers, " << journey_gen.different_journey_count()
            << " different journeys ("
            << journey_gen.generated_routing_query_count()
            << " routing queries).\n";
  if (check_load) {
    std::cout << "Had to generate " << journey_gen.over_capacity_skipped_count()
              << " additional journeys because of load restrictions.\n";
  }

  return 0;
}
