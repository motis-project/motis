#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "utl/enumerate.h"
#include "utl/to_vec.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/get_load_internal.h"
#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/passenger_group_container.h"
#include "motis/paxmon/pci_container.h"

using namespace testing;

namespace motis::paxmon {

namespace {

inline passenger_group mk_pg(std::uint16_t passengers, float probability) {
  return make_passenger_group({}, {}, passengers, INVALID_TIME,
                              group_source_flags::NONE, probability);
}

inline passenger_group_container mk_pgc(std::vector<passenger_group>&& pgs) {
  passenger_group_container pgc;
  for (auto& pg : pgs) {
    pgc.add(std::move(pg));
  }
  return pgc;
}

inline pci_groups mk_pci(passenger_group_container const& pgc,
                         pci_container& pcis) {
  auto const idx = pcis.insert();
  auto groups = pcis.groups_[idx];
  for (auto const& pg : pgc) {
    groups.emplace_back(pg->id_);
  }
  pcis.init_expected_load(pgc, idx);
  return groups;
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
  auto const pgc = mk_pgc({mk_pg(10, 1.0F)});
  auto pcis = pci_container{};
  auto const pcig = mk_pci(pgc, pcis);
  auto const capacity = static_cast<std::uint16_t>(20);

  EXPECT_EQ(get_base_load(pgc, pcig), 10);

  auto const pdf = get_load_pdf(pgc, pcig);
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

  for (auto const& cdf : {get_cdf(pdf), get_load_cdf(pgc, pcig)}) {
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

  EXPECT_EQ(get_mean_load(pgc, pcig), 10);
  EXPECT_EQ(get_median_load(get_cdf(pdf)), 10);
}

TEST(paxmon_get_load, only_multiple_certain) {
  auto const pgc = mk_pgc({mk_pg(10, 1.0F), mk_pg(20, 1.0F), mk_pg(30, 1.0F)});
  auto pcis = pci_container{};
  auto const pcig = mk_pci(pgc, pcis);
  auto const capacity = static_cast<std::uint16_t>(100);

  EXPECT_EQ(get_base_load(pgc, pcig), 60);

  auto const pdf = get_load_pdf(pgc, pcig);
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

  for (auto const& cdf : {get_cdf(pdf), get_load_cdf(pgc, pcig)}) {
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

  EXPECT_EQ(get_mean_load(pgc, pcig), 60);
  EXPECT_EQ(get_median_load(get_cdf(pdf)), 60);
}

TEST(paxmon_get_load, two_groups) {
  auto const pgc = mk_pgc({mk_pg(10, 1.0F), mk_pg(20, 0.4F)});
  auto pcis = pci_container{};
  auto const pcig = mk_pci(pgc, pcis);
  auto const capacity = static_cast<std::uint16_t>(20);

  EXPECT_EQ(get_base_load(pgc, pcig), 10);

  auto const pdf = get_load_pdf(pgc, pcig);
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

  for (auto const& cdf : {get_cdf(pdf), get_load_cdf(pgc, pcig)}) {
    EXPECT_EQ(cdf.data_, (make_cdf({{10, 0.6F}, {30, 1.0F}}).data_));
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

  EXPECT_EQ(get_mean_load(pgc, pcig), 18);
  EXPECT_EQ(get_median_load(get_cdf(pdf)), 10);
}

#ifdef MOTIS_AVX2
TEST(paxmon_get_load, base_eq_avx) {
  auto gen = std::mt19937{std::random_device{}()};
  auto base_group_count_dist = std::uniform_int_distribution{0, 200};
  auto fc_group_count_dist = std::uniform_int_distribution{1, 100};
  auto add_group_count_dist = std::uniform_int_distribution{1, 10};
  auto group_size_dist = std::normal_distribution<float>{1.5F, 3.0F};
  auto prob_dist = std::uniform_real_distribution<float>{0.0F, 1.0F};

  auto const get_group_size = [&]() {
    return static_cast<std::uint16_t>(std::max(1.0F, group_size_dist(gen)));
  };

  for (auto run = 0; run < 5; ++run) {
    auto const base_group_count = base_group_count_dist(gen);
    auto const fc_group_count = fc_group_count_dist(gen);
    auto pgc = passenger_group_container{};
    pgc.reserve(base_group_count + fc_group_count);

    for (auto grp = 0; grp < base_group_count; ++grp) {
      pgc.add(mk_pg(get_group_size(), 1.0F));
    }

    for (auto grp = 0; grp < fc_group_count; ++grp) {
      pgc.add(mk_pg(get_group_size(), prob_dist(gen)));
    }

    auto pcis = pci_container{};
    auto const pcig = mk_pci(pgc, pcis);
    auto pdf_base = get_load_pdf_base(pgc, pcig);
    auto pdf_avx = get_load_pdf_avx(pgc, pcig);

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
