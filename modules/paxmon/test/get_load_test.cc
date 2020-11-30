#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "utl/enumerate.h"
#include "utl/to_vec.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/get_load_internal.h"
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
  auto const grp_ptrs = utl::to_vec(pgs, [](auto const& pg) {
    return const_cast<passenger_group*>(&pg);  // NOLINT
  });
  return pax_connection_info(begin(grp_ptrs), end(grp_ptrs));
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

  EXPECT_EQ(get_mean_load(pci), 10);
  EXPECT_EQ(get_median_load(get_cdf(pdf)), 10);
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

  EXPECT_EQ(get_mean_load(pci), 60);
  EXPECT_EQ(get_median_load(get_cdf(pdf)), 60);
}

TEST(paxmon_get_load, two_groups) {
  auto const pgs =
      std::vector<passenger_group>{{mk_pg(10, 1.0F), mk_pg(20, 0.4F)}};
  auto const pci = mk_pci(pgs);
  auto const capacity = static_cast<std::uint16_t>(20);

  EXPECT_EQ(get_base_load(pci), 10);

  auto const pdf = get_load_pdf(pci);
  EXPECT_EQ(pdf.data_, (make_pdf({{10, 0.6F}, {30, 0.4F}}).data_));
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

  EXPECT_EQ(get_mean_load(pci), 18);
  EXPECT_EQ(get_median_load(get_cdf(pdf)), 10);
}

#ifdef MOTIS_AVX2
TEST(paxmon_get_load, base_eq_avx) {
  auto gen = std::mt19937{std::random_device{}()};
  auto base_group_count_dist = std::uniform_int_distribution{0, 200};
  auto fc_group_count_dist = std::uniform_int_distribution{1, 1'000};
  auto add_group_count_dist = std::uniform_int_distribution{1, 200};
  auto group_size_dist = std::normal_distribution<float>{1.5F, 3.0F};
  auto prob_dist = std::uniform_real_distribution<float>{0.0F, 1.0F};

  auto const get_group_size = [&]() {
    return static_cast<std::uint16_t>(std::max(1.0F, group_size_dist(gen)));
  };

  for (auto run = 0; run < 100; ++run) {
    auto const base_group_count = base_group_count_dist(gen);
    auto const fc_group_count = fc_group_count_dist(gen);
    auto pgs = std::vector<passenger_group>{};
    pgs.reserve(base_group_count + fc_group_count);

    for (auto grp = 0; grp < base_group_count; ++grp) {
      pgs.emplace_back(mk_pg(get_group_size(), 1.0F));
    }

    for (auto grp = 0; grp < fc_group_count; ++grp) {
      pgs.emplace_back(mk_pg(get_group_size(), prob_dist(gen)));
    }

    auto const pci = mk_pci(pgs);
    auto pdf_base = get_load_pdf_base(pci);
    auto pdf_avx = get_load_pdf_avx(pci);

    ASSERT_THAT(pdf_avx.data_, Pointwise(FloatNear(1E-5F), pdf_base.data_));

    auto add_pgs = std::vector<passenger_group>{};
    auto add_grps = std::vector<std::pair<passenger_group const*, float>>{};
    auto const add_group_count = add_group_count_dist(gen);
    add_pgs.reserve(add_group_count);
    add_grps.reserve(add_group_count);
    for (auto grp = 0; grp < add_group_count; ++grp) {
      auto const prob = prob_dist(gen);
      auto const& pg = add_pgs.emplace_back(mk_pg(get_group_size(), prob));
      add_grps.emplace_back(&pg, prob);
    }

    add_additional_groups_base(pdf_base, add_grps);
    add_additional_groups_avx(pdf_avx, add_grps);

    ASSERT_THAT(pdf_avx.data_, Pointwise(FloatNear(1E-5F), pdf_base.data_));
  }
}
#endif

}  // namespace motis::paxmon
