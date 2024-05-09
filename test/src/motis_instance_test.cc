#include "motis/test/motis_instance_test.h"

#include <system_error>

#include "conf/options_parser.h"

#include "fmt/format.h"

#include "utl/zip.h"

#include "motis/core/common/logging.h"

#include "motis/module/clog_redirect.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_publish.h"
#include "motis/module/message.h"

using namespace motis::module;
using namespace motis::bootstrap;

namespace motis::test {

template <typename Base>
generic_motis_instance_test<Base>::generic_motis_instance_test(
    std::vector<std::string> const& modules,
    std::vector<std::string> const& modules_cmdline_opt)
    : instance_(std::make_unique<motis_instance>()) {
  if constexpr (sizeof(void*) < 8) {
    dispatcher::direct_mode_dispatcher_ = instance_.get();
  }

  auto modules_cmdline_opt_patched = modules_cmdline_opt;
  modules_cmdline_opt_patched.emplace_back("--nigiri.no_cache=true");

  import_settings import_opt;
  std::vector<conf::configuration*> confs = {&import_opt};
  for (auto const& module : instance_->modules()) {
    confs.push_back(module);
  }

  conf::options_parser parser(confs);
  parser.read_command_line_args(modules_cmdline_opt_patched);

  clog_redirect::set_enabled(false);
  instance_->import(module_settings{modules}, import_opt, true);
  instance_->init_modules(module_settings{modules});
}

template <typename Base>
msg_ptr generic_motis_instance_test<Base>::call(msg_ptr const& msg) const {
  return instance_->call(msg);
}

template <typename Base>
void generic_motis_instance_test<Base>::publish(msg_ptr const& msg) const {
  instance_->publish(msg);
}

template <typename Base>
msg_ptr generic_motis_instance_test<Base>::call(
    std::string const& target) const {
  return call(make_no_msg(target));
}

template <typename Base>
void generic_motis_instance_test<Base>::publish(
    std::string const& target) const {
  publish(make_no_msg(target));
}

template <typename Base>
std::function<module::msg_ptr(module::msg_ptr const&)>
generic_motis_instance_test<Base>::msg_sink(std::vector<module::msg_ptr>* vec) {
  return [vec](module::msg_ptr const& m) -> module::msg_ptr {
    vec->push_back(m);
    return nullptr;
  };
}

template struct generic_motis_instance_test<::testing::Test>;
template struct generic_motis_instance_test<
    testing::TestWithParam<const char*>>;

}  // namespace motis::test
