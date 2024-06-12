#pragma once

#include <filesystem>
#include <string_view>

#include "icc/types.h"

namespace icc {

vector_map<elevator_idx_t, elevator> parse_fasta(std::string_view);
vector_map<elevator_idx_t, elevator> parse_fasta(std::filesystem::path const&);

}  // namespace icc