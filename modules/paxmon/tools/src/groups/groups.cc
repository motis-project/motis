#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>

#include "boost/filesystem.hpp"

#include "fmt/core.h"

#include "conf/configuration.h"
#include "conf/options_parser.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/file.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"

#include "motis/paxmon/csv_writer.h"
#include "motis/paxmon/loader/csv/row.h"

#include "motis/paxmon/tools/groups/group_generator.h"

using namespace motis::paxmon;
using namespace motis::paxmon::tools::groups;
namespace fs = boost::filesystem;

struct group_settings : public conf::configuration {
  enum class mode_t { REPLACE, SPLIT };

  friend std::istream& operator>>(std::istream& in, mode_t& mode) {
    std::string token;
    in >> token;

    if (token == "replace") {
      mode = mode_t::REPLACE;
    } else if (token == "split") {
      mode = mode_t::SPLIT;
    }

    return in;
  }

  friend std::ostream& operator<<(std::ostream& out, mode_t const mode) {
    switch (mode) {
      case mode_t::REPLACE: out << "replace"; break;
      case mode_t::SPLIT: out << "split"; break;
    }
    return out;
  }

  group_settings() : configuration{"Group Settings"} {
    param(in_path_, "in,i", "Input file");
    param(out_path_, "out,o", "Output file");
    param(mode_, "mode",
          "Group generation mode:\n"
          "replace = Ignore existing group sizes\n"
          "split = Split existing groups");
    param(group_size_mean_, "size_mean", "Group size mean");
    param(group_size_stddev_, "size_stddev", "Group size standard deviation");
    param(group_count_mean_, "count_mean", "Group count mean");
    param(group_count_stddev_, "count_stddev",
          "Group count standard deviation");
  }

  std::string in_path_;
  std::string out_path_;
  mode_t mode_{mode_t::REPLACE};

  double group_size_mean_{1.5};
  double group_size_stddev_{3.0};

  double group_count_mean_{2.0};
  double group_count_stddev_{10.0};
};

int main(int argc, char const** argv) {
  group_settings opt;

  try {
    conf::options_parser parser{{&opt}};
    parser.read_command_line_args(argc, argv, false);

    if (parser.help()) {
      parser.print_help(std::cout);
      return 0;
    }

    if (opt.in_path_.empty() || opt.out_path_.empty()) {
      std::cerr << "Missing input/output path parameters\n\n";
      parser.print_help(std::cerr);
      return 1;
    }
  } catch (std::exception const& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  if (!fs::is_regular_file(opt.in_path_)) {
    std::cerr << "Input file not found: " << opt.in_path_ << "\n";
    return 1;
  }

  group_generator group_gen{opt.group_size_mean_, opt.group_size_stddev_,
                            opt.group_count_mean_, opt.group_count_stddev_};

  auto buf = utl::file(opt.in_path_.data(), "r").content();
  auto const file_content = utl::cstr{buf.data(), buf.size()};

  auto current_id = std::pair<std::uint64_t, std::uint64_t>{0, 0};
  auto current_rows = std::vector<motis::paxmon::loader::csv::row>{};

  auto total_input_groups = 0ULL;
  auto total_input_pax = 0ULL;
  auto total_output_groups = 0ULL;
  auto total_output_pax = 0ULL;

  auto writer = csv_writer{opt.out_path_};
  writer << "id"
         << "secondary_id"
         << "leg_idx"
         << "leg_type"
         << "from"
         << "to"
         << "enter"
         << "exit"
         << "category"
         << "train_nr"
         << "passengers" << end_row;

  auto write_journey = [&](std::uint64_t secondary_id, std::uint16_t pax) {
    for (auto const& row : current_rows) {
      writer << row.id_.val() << secondary_id << row.leg_idx_.val()
             << row.leg_type_.val().view() << row.from_.val().view()
             << row.to_.val().view() << row.enter_.val() << row.exit_.val()
             << row.category_.val().view() << row.train_nr_.val() << pax
             << end_row;
    }
    ++total_output_groups;
    total_output_pax += pax;
  };

  auto process_journey = [&]() {
    if (current_rows.empty()) {
      return;
    }
    auto const input_pax = current_rows.front().passengers_.val();
    ++total_input_groups;
    total_input_pax += input_pax;
    if (current_id.second != 0) {
      std::cout << "WARNING: Skipping journey " << current_id.first << "."
                << current_id.second << "\n";
      return;
    }

    switch (opt.mode_) {
      case group_settings::mode_t::SPLIT: {

        auto distributed = 0U;
        auto secondary_id = 0ULL;
        while (distributed < input_pax) {
          auto const group_size =
              group_gen.get_group_size(input_pax - distributed);
          ++secondary_id;
          distributed += group_size;
          write_journey(secondary_id, group_size);
        }
        break;
      }
      case group_settings::mode_t::REPLACE: {
        auto const group_count = group_gen.get_group_count();
        for (auto secondary_id = 1ULL; secondary_id <= group_count;
             ++secondary_id) {
          write_journey(secondary_id, group_gen.get_group_size());
        }
        break;
      }
    }
  };

  utl::line_range<utl::buf_reader>{file_content}  //
      | utl::csv<loader::csv::row>()  //
      |
      utl::for_each([&](loader::csv::row const& row) {
        auto const id = std::make_pair(row.id_.val(), row.secondary_id_.val());
        if (id != current_id) {
          process_journey();
          current_id = id;
          current_rows.clear();
        }
        current_rows.emplace_back(row);
      });
  process_journey();

  fmt::print("Input:  {:10} groups {:10} passengers\n", total_input_groups,
             total_input_pax);
  fmt::print("Output: {:10} groups {:10} passengers\n", total_output_groups,
             total_output_pax);

  return 0;
}
