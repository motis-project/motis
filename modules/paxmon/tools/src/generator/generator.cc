#include <algorithm>
#include <fstream>
#include <vector>

#include "conf/configuration.h"
#include "conf/options_parser.h"

#include "utl/progress_tracker.h"

#include "motis/bootstrap/dataset_settings.h"
#include "motis/bootstrap/import_settings.h"
#include "motis/bootstrap/module_settings.h"
#include "motis/bootstrap/motis_instance.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"

#include "motis/paxmon/tools/convert/journey_converter.h"
#include "motis/paxmon/tools/generator/query_generator.h"
#include "motis/paxmon/tools/groups/group_generator.h"

using namespace motis;
using namespace motis::bootstrap;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::paxmon::tools::generator;
using namespace motis::paxmon::tools::convert;
using namespace motis::paxmon::tools::groups;

struct generator_settings : public conf::configuration {
  generator_settings() : configuration{"Generator Settings"} {
    param(csv_journey_path_, "out,o", "Output CSV Journey file");
    param(num_threads_, "num_threads", "Number of worker threads");
    param(pax_count_, "pax_count", "Total number of passengers");
    param(group_size_mean_, "group_size_mean", "Group size mean");
    param(group_size_stddev_, "group_size_stddev",
          "Group size standard deviation");
    param(group_count_mean_, "group_count_mean", "Group count mean");
    param(group_count_stddev_, "group_count_stddev",
          "Group count standard deviation");
    param(router_target_, "router_target", "Router target");
  }

  std::string csv_journey_path_{"journeys.csv"};
  unsigned num_threads_{std::thread::hardware_concurrency()};
  std::string router_target_;

  unsigned pax_count_{1000};

  double group_size_mean_{1.5};
  double group_size_stddev_{3.0};

  double group_count_mean_{2.0};
  double group_count_stddev_{10.0};
};

struct journey_generator {
  journey_generator(motis_instance& instance,
                    generator_settings const& generator_opt,
                    group_generator& group_gen)
      : instance_{instance},
        query_gen_{instance.sched()},
        group_gen_{group_gen},
        generator_opt_{generator_opt},
        converter_{generator_opt.csv_journey_path_} {
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
      auto const primary_id = next_primary_id_++;
      auto const group_count = group_gen_.get_group_count();
      for (auto secondary_id = 1ULL; secondary_id <= group_count;
           ++secondary_id) {
        auto const group_size = group_gen_.get_group_size();
        converter_.write_journey(j, primary_id, secondary_id, group_size);
        generated += group_size;
      }
      groups_generated_ += group_count;
    }
    different_journeys_ += journeys.size();

    return generated;
  }

  motis_instance& instance_;
  query_generator query_gen_;
  group_generator& group_gen_;
  generator_settings const& generator_opt_;
  unsigned in_flight_{};
  unsigned pax_generated_{};
  unsigned groups_generated_{};
  unsigned routing_queries_{};
  unsigned different_journeys_{};
  std::uint64_t next_primary_id_{1};
  journey_converter converter_;
  utl::progress_tracker_ptr progress_tracker_;
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

  std::cout << "\nGenerating journeys for at least " << generator_opt.pax_count_
            << " passengers..." << std::endl;

  group_generator group_gen{
      generator_opt.group_size_mean_, generator_opt.group_size_stddev_,
      generator_opt.group_count_mean_, generator_opt.group_count_stddev_};
  journey_generator journey_gen{instance, generator_opt, group_gen};
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

  return 0;
}
