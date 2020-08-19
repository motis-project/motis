#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <vector>

#include "utl/to_vec.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/pax_connection_info.h"

using namespace testing;

namespace motis::paxmon {

namespace {

inline passenger_group mk_pg(std::uint16_t passengers, float probability) {
  return passenger_group{{}, 0ULL, {}, passengers, true, probability, {}};
}

inline pax_connection_info mk_pci(std::vector<passenger_group> const& pgs) {
  return pax_connection_info{utl::to_vec(pgs, [](auto const& pg) {
    return pax_section_info{const_cast<passenger_group*>(&pg)};  // NOLINT
  })};
}

}  // namespace

TEST(paxmon_get_load, only_one_certain) {
  auto const pgs = std::vector<passenger_group>{{mk_pg(10, 1.0f)}};
  auto const pci = mk_pci(pgs);
  auto const capacity = static_cast<std::uint16_t>(20);

  ASSERT_EQ(get_base_load(pci), 10);

  auto const pdf = get_load_pdf(pci);
  ASSERT_EQ(pdf, (pdf_t{{10, 1.0f}}));
  auto const lf_pdf = to_load_factor(pdf, capacity);
  ASSERT_EQ(lf_pdf, (std::map<float, float>{{0.5f, 1.0f}}));

  for (auto const& cdf : {get_cdf(pdf), get_load_cdf(pci)}) {
    ASSERT_EQ(cdf, (cdf_t{{10, 1.0f}}));
    auto const lf_cdf = to_load_factor(cdf, capacity);
    ASSERT_EQ(lf_cdf, (std::map<float, float>{{0.5f, 1.0f}}));
  }
}

TEST(paxmon_get_load, only_multiple_certain) {
  auto const pgs = std::vector<passenger_group>{
      {mk_pg(10, 1.0f), mk_pg(20, 1.0f), mk_pg(30, 1.0f)}};
  auto const pci = mk_pci(pgs);
  auto const capacity = static_cast<std::uint16_t>(100);

  ASSERT_EQ(get_base_load(pci), 60);

  auto const pdf = get_load_pdf(pci);
  ASSERT_EQ(pdf, (pdf_t{{60, 1.0f}}));
  auto const lf_pdf = to_load_factor(pdf, capacity);
  ASSERT_EQ(lf_pdf, (std::map<float, float>{{0.6f, 1.0f}}));

  for (auto const& cdf : {get_cdf(pdf), get_load_cdf(pci)}) {
    ASSERT_EQ(cdf, (cdf_t{{60, 1.0f}}));
    auto const lf_cdf = to_load_factor(cdf, capacity);
    ASSERT_EQ(lf_cdf, (std::map<float, float>{{0.6f, 1.0f}}));
  }
}

TEST(paxmon_get_load, two_groups) {
  auto const pgs =
      std::vector<passenger_group>{{mk_pg(10, 1.0f), mk_pg(20, 0.4f)}};
  auto const pci = mk_pci(pgs);
  auto const capacity = static_cast<std::uint16_t>(20);

  ASSERT_EQ(get_base_load(pci), 10);

  auto const pdf = get_load_pdf(pci);
  ASSERT_EQ(pdf, (pdf_t{{10, 0.6f}, {30, 0.4f}}));
  auto const lf_pdf = to_load_factor(pdf, capacity);
  ASSERT_EQ(lf_pdf, (std::map<float, float>{{0.5f, 0.6f}, {1.5f, 0.4f}}));

  for (auto const& cdf : {get_cdf(pdf), get_load_cdf(pci)}) {
    ASSERT_EQ(cdf, (cdf_t{{10, 0.6f}, {30, 1.0f}}));
    auto const lf_cdf = to_load_factor(cdf, capacity);
    ASSERT_EQ(lf_cdf, (std::map<float, float>{{0.5f, 0.6f}, {1.5f, 1.0f}}));
  }
}

}  // namespace motis::paxmon
