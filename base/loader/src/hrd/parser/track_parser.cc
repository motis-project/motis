#include "motis/loader/hrd/parser/providers_parser.h"

#include "utl/parser/arg_parser.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/loader/util.h"

using namespace utl;
using namespace motis::logging;

namespace motis::loader::hrd {

void verify_line_format(cstr line, char const* filename, int line_number) {
  // Verify that the provider number has 5 digits.
  auto provider_number = line.substr(0, size(5));
  utl::verify(std::all_of(begin(provider_number), end(provider_number),
                          [](char c) { return std::isdigit(c); }),
              "provider line format mismatch in {}:{}", filename, line_number);

  utl::verify(line[6] == 'K' || line[6] == ':',
              "provider line format mismatch in {}:{}", filename, line_number);
}

std::string parse_name(cstr s) {
  bool const start_is_quote = s[0] == '\'' || s[0] == '\"';
  char const end = start_is_quote ? s[0] : ' ';
  int i = start_is_quote ? 1 : 0;
  while (s && s[i] != end) {
    ++i;
  }
  auto region = s.substr(start_is_quote ? 1 : 0, size(i - 1));
  return {region.str, region.len};
}

provider_info read_provider_names(cstr line, char const* filename,
                                  int line_number) {
  provider_info info;

  int const short_name = line.substr_offset(" K ");
  int const long_name = line.substr_offset(" L ");
  int const full_name = line.substr_offset(" V ");

  utl::verify(short_name != -1 && long_name != -1 && full_name != -1,
              "provider line format mismatch in {}:{}", filename, line_number);

  info.short_name_ = parse_name(line.substr(short_name + 3));
  info.long_name_ = parse_name(line.substr(long_name + 3));
  info.full_name_ = parse_name(line.substr(full_name + 3));

  return info;
}

std::map<uint64_t, provider_info> parse_providers(loaded_file const& file,
                                                  config const& c) {
  scoped_timer const timer("parsing providers");

  std::map<uint64_t, provider_info> providers;
  provider_info current_info;
  int previous_provider_number = 0;

  for_each_line_numbered(file.content(), [&](cstr line, int line_number) {
    auto provider_number = parse<int>(line.substr(c.track_.prov_nr_));
    if (line[6] == 'K') {
      current_info = read_provider_names(line, file.name(), line_number);
      previous_provider_number = provider_number;
    } else {
      utl::verify(previous_provider_number == provider_number,
                  "provider line format mismatch in {}:{}", file.name(),
                  line_number);
      for_each_token(line.substr(8), ' ', [&](cstr token) {
        providers[raw_to_int<uint64_t>(token)] = current_info;
      });
    }
  });

  return providers;
}

}  // namespace motis::loader::hrd
