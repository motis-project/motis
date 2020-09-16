#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

#include "utl/enumerate.h"
#include "utl/to_vec.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/pax_connection_info.h"

using namespace testing;

namespace motis::paxmon {

namespace {

inline passenger_group mk_pg(std::uint16_t passengers, float probability) {
  return passenger_group{{},         0ULL,         {},
                         passengers, INVALID_TIME, group_source_flags::NONE,
                         true,       probability,  {}};
}

inline pax_connection_info mk_pci(std::vector<passenger_group> const& pgs) {
  return pax_connection_info{utl::to_vec(pgs, [](auto const& pg) {
    return pax_section_info{const_cast<passenger_group*>(&pg)};  // NOLINT
  })};
}

inline pax_pdf make_pdf(std::map<std::uint16_t, float> const& m) {
  auto pdf = pax_pdf{};
  for (auto const& e : m) {
    pdf.data_.resize(e.first + 1);
    pdf.data_[e.first] = e.second;
  }
  return pdf;
}

inline pax_cdf make_cdf(std::map<std::uint16_t, float> const& m) {
  auto cdf = pax_cdf{};
  for (auto const& e : m) {
    if (!cdf.data_.empty()) {
      auto const last_val = cdf.data_.back();
      auto const old_size = cdf.data_.size();
      cdf.data_.resize(e.first + 1, last_val);
    } else {
      cdf.data_.resize(e.first + 1);
    }
    cdf.data_[e.first] = e.second;
  }
  return cdf;
}

}  // namespace

TEST(paxmon_get_load, only_one_certain) {
  auto const pgs = std::vector<passenger_group>{{mk_pg(10, 1.0F)}};
  auto const pci = mk_pci(pgs);
  auto const capacity = static_cast<std::uint16_t>(20);

  EXPECT_EQ(get_base_load(pci), 10);

  auto const pdf = get_load_pdf(pci);
  EXPECT_EQ(pdf, (make_pdf({{10, 1.0F}})));
  EXPECT_TRUE(load_factor_possibly_ge(pdf, capacity, 0.2F));
  EXPECT_TRUE(load_factor_possibly_ge(pdf, capacity, 0.5F));
  EXPECT_FALSE(load_factor_possibly_ge(pdf, capacity, 1.0F));
  EXPECT_FALSE(load_factor_possibly_ge(pdf, capacity, 1.5F));
  EXPECT_FALSE(load_factor_possibly_ge(pdf, capacity, 2.0F));
  auto const lf_pdf = to_load_factor(pdf, capacity);
  EXPECT_EQ(lf_pdf, (std::map<float, float>{{0.5F, 1.0F}}));
  EXPECT_TRUE(load_factor_possibly_ge(lf_pdf, 0.2F));
  EXPECT_TRUE(load_factor_possibly_ge(lf_pdf, 0.5F));
  EXPECT_FALSE(load_factor_possibly_ge(lf_pdf, 1.0F));
  EXPECT_FALSE(load_factor_possibly_ge(lf_pdf, 1.5F));
  EXPECT_FALSE(load_factor_possibly_ge(lf_pdf, 2.0F));

  for (auto const& cdf : {get_cdf(pdf), get_load_cdf(pci)}) {
    EXPECT_EQ(cdf, (make_cdf({{10, 1.0F}})));
    EXPECT_TRUE(load_factor_possibly_ge(cdf, capacity, 0.2F));
    EXPECT_TRUE(load_factor_possibly_ge(cdf, capacity, 0.5F));
    EXPECT_FALSE(load_factor_possibly_ge(cdf, capacity, 1.0F));
    EXPECT_FALSE(load_factor_possibly_ge(cdf, capacity, 1.5F));
    EXPECT_FALSE(load_factor_possibly_ge(cdf, capacity, 2.0F));
    auto const lf_cdf = to_load_factor(cdf, capacity);
    EXPECT_EQ(lf_cdf, (std::map<float, float>{{0.5F, 1.0F}}));
    EXPECT_TRUE(load_factor_possibly_ge(lf_cdf, 0.2F));
    EXPECT_TRUE(load_factor_possibly_ge(lf_cdf, 0.5F));
    EXPECT_FALSE(load_factor_possibly_ge(lf_cdf, 1.0F));
    EXPECT_FALSE(load_factor_possibly_ge(lf_cdf, 1.5F));
    EXPECT_FALSE(load_factor_possibly_ge(lf_cdf, 2.0F));
  }
}

TEST(paxmon_get_load, only_multiple_certain) {
  auto const pgs = std::vector<passenger_group>{
      {mk_pg(10, 1.0F), mk_pg(20, 1.0F), mk_pg(30, 1.0F)}};
  auto const pci = mk_pci(pgs);
  auto const capacity = static_cast<std::uint16_t>(100);

  EXPECT_EQ(get_base_load(pci), 60);

  auto const pdf = get_load_pdf(pci);
  EXPECT_EQ(pdf, (make_pdf({{60, 1.0F}})));
  EXPECT_TRUE(load_factor_possibly_ge(pdf, capacity, 0.2F));
  EXPECT_TRUE(load_factor_possibly_ge(pdf, capacity, 0.5F));
  EXPECT_FALSE(load_factor_possibly_ge(pdf, capacity, 1.0F));
  EXPECT_FALSE(load_factor_possibly_ge(pdf, capacity, 1.5F));
  EXPECT_FALSE(load_factor_possibly_ge(pdf, capacity, 2.0F));
  auto const lf_pdf = to_load_factor(pdf, capacity);
  EXPECT_EQ(lf_pdf, (std::map<float, float>{{0.6F, 1.0F}}));
  EXPECT_TRUE(load_factor_possibly_ge(lf_pdf, 0.2F));
  EXPECT_TRUE(load_factor_possibly_ge(lf_pdf, 0.5F));
  EXPECT_FALSE(load_factor_possibly_ge(lf_pdf, 1.0F));
  EXPECT_FALSE(load_factor_possibly_ge(lf_pdf, 1.5F));
  EXPECT_FALSE(load_factor_possibly_ge(lf_pdf, 2.0F));

  for (auto const& cdf : {get_cdf(pdf), get_load_cdf(pci)}) {
    EXPECT_EQ(cdf, (make_cdf({{60, 1.0F}})));
    EXPECT_TRUE(load_factor_possibly_ge(cdf, capacity, 0.2F));
    EXPECT_TRUE(load_factor_possibly_ge(cdf, capacity, 0.5F));
    EXPECT_FALSE(load_factor_possibly_ge(cdf, capacity, 1.0F));
    EXPECT_FALSE(load_factor_possibly_ge(cdf, capacity, 1.5F));
    EXPECT_FALSE(load_factor_possibly_ge(cdf, capacity, 2.0F));
    auto const lf_cdf = to_load_factor(cdf, capacity);
    EXPECT_EQ(lf_cdf, (std::map<float, float>{{0.6F, 1.0F}}));
    EXPECT_TRUE(load_factor_possibly_ge(lf_cdf, 0.2F));
    EXPECT_TRUE(load_factor_possibly_ge(lf_cdf, 0.5F));
    EXPECT_FALSE(load_factor_possibly_ge(lf_cdf, 1.0F));
    EXPECT_FALSE(load_factor_possibly_ge(lf_cdf, 1.5F));
    EXPECT_FALSE(load_factor_possibly_ge(lf_cdf, 2.0F));
  }
}

TEST(paxmon_get_load, two_groups) {
  auto const pgs =
      std::vector<passenger_group>{{mk_pg(10, 1.0F), mk_pg(20, 0.4F)}};
  auto const pci = mk_pci(pgs);
  auto const capacity = static_cast<std::uint16_t>(20);

  EXPECT_EQ(get_base_load(pci), 10);

  auto const pdf = get_load_pdf(pci);
  EXPECT_EQ(pdf, (make_pdf({{10, 0.6F}, {30, 0.4F}})));
  EXPECT_TRUE(load_factor_possibly_ge(pdf, capacity, 0.2F));
  EXPECT_TRUE(load_factor_possibly_ge(pdf, capacity, 0.5F));
  EXPECT_TRUE(load_factor_possibly_ge(pdf, capacity, 1.0F));
  EXPECT_TRUE(load_factor_possibly_ge(pdf, capacity, 1.5F));
  EXPECT_FALSE(load_factor_possibly_ge(pdf, capacity, 2.0F));
  auto const lf_pdf = to_load_factor(pdf, capacity);
  EXPECT_EQ(lf_pdf, (std::map<float, float>{{0.5F, 0.6F}, {1.5F, 0.4F}}));
  EXPECT_TRUE(load_factor_possibly_ge(lf_pdf, 0.2F));
  EXPECT_TRUE(load_factor_possibly_ge(lf_pdf, 0.5F));
  EXPECT_TRUE(load_factor_possibly_ge(lf_pdf, 1.0F));
  EXPECT_TRUE(load_factor_possibly_ge(lf_pdf, 1.5F));
  EXPECT_FALSE(load_factor_possibly_ge(lf_pdf, 2.0F));

  for (auto const& cdf : {get_cdf(pdf), get_load_cdf(pci)}) {
    EXPECT_EQ(cdf, (make_cdf({{10, 0.6F}, {30, 1.0F}})));
    EXPECT_TRUE(load_factor_possibly_ge(cdf, capacity, 0.2F));
    EXPECT_TRUE(load_factor_possibly_ge(cdf, capacity, 0.5F));
    EXPECT_TRUE(load_factor_possibly_ge(cdf, capacity, 1.0F));
    EXPECT_TRUE(load_factor_possibly_ge(cdf, capacity, 1.5F));
    EXPECT_FALSE(load_factor_possibly_ge(cdf, capacity, 2.0F));
    auto const lf_cdf = to_load_factor(cdf, capacity);
    EXPECT_EQ(lf_cdf, (std::map<float, float>{{0.5F, 0.6F}, {1.5F, 1.0F}}));
    EXPECT_TRUE(load_factor_possibly_ge(lf_cdf, 0.2F));
    EXPECT_TRUE(load_factor_possibly_ge(lf_cdf, 0.5F));
    EXPECT_TRUE(load_factor_possibly_ge(lf_cdf, 1.0F));
    EXPECT_TRUE(load_factor_possibly_ge(lf_cdf, 1.5F));
    EXPECT_FALSE(load_factor_possibly_ge(lf_cdf, 2.0F));
  }
}

}  // namespace motis::paxmon
