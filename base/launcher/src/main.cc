#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>

#include "fmt/core.h"
#include "fmt/ranges.h"

#include "boost/asio/deadline_timer.hpp"
#include "boost/asio/io_service.hpp"
#include "boost/asio/signal_set.hpp"

#include "utl/erase_if.h"
#include "utl/parser/cstr.h"

#include "net/stop_handler.h"

#include "conf/options_parser.h"

#ifdef PROTOBUF_LINKED
#include "google/protobuf/stubs/common.h"
#endif

#include "opentelemetry/context/propagation/global_propagator.h"
#include "opentelemetry/context/propagation/text_map_propagator.h"
#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter_factory.h"
#include "opentelemetry/sdk/trace/processor.h"
#include "opentelemetry/sdk/trace/recordable.h"
#include "opentelemetry/sdk/trace/simple_processor_factory.h"
#include "opentelemetry/sdk/trace/tracer_provider.h"
#include "opentelemetry/sdk/trace/tracer_provider_factory.h"
#include "opentelemetry/trace/propagation/http_trace_context.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/tracer_provider.h"

#include "motis/core/common/logging.h"
#include "motis/core/otel/tracer.h"
#include "motis/bootstrap/import_settings.h"
#include "motis/bootstrap/module_settings.h"
#include "motis/bootstrap/motis_instance.h"
#include "motis/bootstrap/remote_settings.h"
#include "motis/launcher/batch_mode.h"
#include "motis/launcher/launcher_settings.h"
#include "motis/launcher/server_settings.h"
#include "motis/launcher/web_server.h"

#include "motis/module/otel_runtime_context.h"

#include "version.h"

using namespace motis::bootstrap;
using namespace motis::launcher;
using namespace motis::module;
using namespace motis::logging;
using namespace motis;

namespace {

void init_opentelemetry_tracer(
    opentelemetry::sdk::resource::Resource const& resource) {
  auto exporter =
      opentelemetry::exporter::otlp::OtlpHttpExporterFactory::Create();

  auto processor =
      opentelemetry::sdk::trace::SimpleSpanProcessorFactory::Create(
          std::move(exporter));

  auto provider =
      std::shared_ptr{opentelemetry::sdk::trace::TracerProviderFactory::Create(
          std::move(processor), resource)};

  opentelemetry::trace::Provider::SetTracerProvider(provider);
}

void init_opentelemetry(launcher_settings const& launcher_opt) {
  auto resource_attributes = opentelemetry::sdk::resource::ResourceAttributes{
      {"service.name", "motis"}, {"service.version", short_version()}};
  auto resource =
      opentelemetry::sdk::resource::Resource::Create(resource_attributes);

  opentelemetry::context::RuntimeContext::SetRuntimeContextStorage(
      std::make_shared<otel_runtime_context_storage>());

  if (launcher_opt.otlp_http_) {
    init_opentelemetry_tracer(resource);
  }

  opentelemetry::context::propagation::GlobalTextMapPropagator::
      SetGlobalPropagator(
          std::make_shared<
              opentelemetry::trace::propagation::HttpTraceContext>());

  auto tracer_provider = opentelemetry::trace::Provider::GetTracerProvider();
  motis_tracer = tracer_provider->GetTracer("motis", short_version());
}

}  // namespace

int main(int argc, char const** argv) {
  motis_instance instance;

  auto reg = subc_reg{};
  for (auto const& m : instance.modules()) {
    m->reg_subc(reg);
  }
  if (argc > 1 && !utl::cstr{argv[1]}.starts_with("-")) {
    return reg.execute(argv[1], argc - 1, argv + 1);
  }

  web_server server(instance.runner_.ios(), instance);

  server_settings server_opt;
  import_settings import_opt;

  module_settings module_opt(instance.module_names());
  remote_settings remote_opt;
  launcher_settings launcher_opt;

  std::set<std::string> disabled_by_default{"ppr", "gbfs", "parking"};
  utl::erase_if(module_opt.modules_, [&](std::string const& m) {
    return disabled_by_default.contains(m);
  });

  std::vector<conf::configuration*> confs = {
      &server_opt, &import_opt, &module_opt, &remote_opt, &launcher_opt};
  for (auto const& module : instance.modules()) {
    confs.push_back(module);
  }

  try {
    conf::options_parser parser(confs);
    parser.read_environment("MOTIS_");
    parser.read_command_line_args(argc, argv, false);

    if (parser.help()) {
      fmt::print("\n\tMOTIS {} \n\n", short_version());
      reg.print_list();
      if (auto const module_names = instance.module_names();
          module_names.empty()) {
        fmt::print("\nNo modules available.\n");
      } else {
        fmt::print("Available modules: {}\n\n", module_names);
      }
      parser.print_help(std::cout);
      return 0;
    } else if (parser.version()) {
      fmt::print("MOTIS {}", long_version());
      return 0;
    }

    parser.read_configuration_file(false);

    parser.print_used(std::cout);
  } catch (std::exception const& e) {
    std::cout << "options error: " << e.what() << "\n";
    return 1;
  }

  if (launcher_opt.direct_mode_) {
    dispatcher::direct_mode_dispatcher_ = &instance;
  }

  init_opentelemetry(launcher_opt);

  try {
    instance.import(module_opt, import_opt);
    instance.init_modules(module_opt, launcher_opt.num_threads_);
    instance.init_remotes(remote_opt.get_remotes());

    if (!launcher_opt.init_.empty()) {
      if (launcher_opt.init_.starts_with(".") &&
          std::filesystem::is_regular_file(launcher_opt.init_)) {
        std::ifstream in{launcher_opt.init_};
        std::string json;
        while (!in.eof() && in.peek() != EOF) {
          std::getline(in, json);
          auto const res =
              instance.call(make_msg(json), launcher_opt.num_threads_);
          std::cout << res->to_json() << '\n';
        }
      } else {
        instance.call(launcher_opt.init_, launcher_opt.num_threads_);
      }
    }

    if (launcher_opt.mode_ == launcher_settings::motis_mode_t::SERVER) {
      boost::system::error_code ec;
      server.listen(server_opt.host_, server_opt.port_,
#if defined(NET_TLS)
                    server_opt.cert_path_, server_opt.priv_key_path_,
                    server_opt.dh_path_,
#endif
                    server_opt.log_path_, server_opt.static_path_, ec);
      if (ec) {
        fmt::print("unable to start server: {}\n", ec.message());
        return 1;
      }
    } else if (launcher_opt.mode_ == launcher_settings::motis_mode_t::INIT) {
      return 0;
    }
  } catch (std::exception const& e) {
    fmt::print("\ninitialization error: {}\n", e.what());
    return 1;
  } catch (...) {
    fmt::print("unknown initialization error\n");
    return 1;
  }

  std::unique_ptr<boost::asio::deadline_timer> timer;
  std::unique_ptr<net::stop_handler> stop;
  if (launcher_opt.mode_ == launcher_settings::motis_mode_t::TEST) {
    timer = std::make_unique<boost::asio::deadline_timer>(
        instance.runner_.ios(), boost::posix_time::seconds(1));
    timer->async_wait(
        [&](boost::system::error_code) { instance.runner_.ios().stop(); });
  } else if (launcher_opt.mode_ == launcher_settings::motis_mode_t::BATCH) {
    instance.queue_no_target_msgs_ = true;
    auto start_batch = [&]() {
      LOG(info) << "starting to inject queries";
      inject_queries(
          instance.runner_.ios(), instance, launcher_opt.batch_input_file_,
          launcher_opt.batch_output_file_, launcher_opt.num_threads_);
    };
    remote_opt.get_remotes().empty()
        ? start_batch()
        : instance.on_remotes_registered(start_batch);
  } else if (launcher_opt.mode_ == launcher_settings::motis_mode_t::SERVER) {
    instance.init_io(module_opt);
    stop = std::make_unique<net::stop_handler>(instance.runner_.ios(), [&]() {
      server.stop();
      instance.runner_.ios().stop();
      instance.stop_io();
      instance.stop_remotes();
    });
  }

  LOG(info) << "system boot finished";
  instance.runner_.run(
      launcher_opt.num_threads_,
      launcher_opt.mode_ == launcher_settings::motis_mode_t::SERVER);
  LOG(info) << "shutdown";

#ifdef PROTOBUF_LINKED
  google::protobuf::ShutdownProtobufLibrary();
#endif
}
