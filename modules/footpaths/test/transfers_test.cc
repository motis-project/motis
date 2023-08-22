#include "gtest/gtest.h"

#include "motis/footpaths/transfers.h"

TEST(footpaths, merge_empty_transfer_request_keys) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  auto treq_k_b = transfer_request_keys{};

  auto should_treq_k = transfer_request_keys{};
  auto is_treq_k = merge(treq_k_a, treq_k_b);

  ASSERT_EQ(is_treq_k, should_treq_k);
}

TEST(footpaths, merge_right_empty_transfer_request_keys) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  treq_k_a.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}};
  treq_k_a.to_pf_keys_ = {"a", "b", "c"};

  auto treq_k_b = transfer_request_keys{};

  auto should_treq_k = transfer_request_keys{};
  should_treq_k.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}};
  should_treq_k.to_pf_keys_ = {"a", "b", "c"};

  auto is_trek_k = merge(treq_k_a, treq_k_b);

  ASSERT_EQ(is_trek_k, should_treq_k);
}

TEST(footpaths, merge_left_empty_transfer_request_keys) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  auto treq_k_b = transfer_request_keys{};
  treq_k_b.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}};
  treq_k_b.to_pf_keys_ = {"a", "b", "c"};

  auto should_treq_k = transfer_request_keys{};
  should_treq_k.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}};
  should_treq_k.to_pf_keys_ = {"a", "b", "c"};

  auto is_trek_k = merge(treq_k_a, treq_k_b);

  ASSERT_EQ(is_trek_k, should_treq_k);
}

TEST(footpaths, merge_transfer_request_keys) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  treq_k_a.from_nloc_key_ = key64_t{26};
  treq_k_a.from_pf_key_ = "z";
  treq_k_a.profile_ = "p";
  treq_k_a.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}, key64_t{5}};
  treq_k_a.to_pf_keys_ = {"1", "2", "3", "5"};

  auto treq_k_b = transfer_request_keys{};
  treq_k_b.from_nloc_key_ = key64_t{26};
  treq_k_b.from_pf_key_ = "z";
  treq_k_b.profile_ = "p";
  treq_k_b.to_nloc_keys_ = {key64_t{3}, key64_t{4}, key64_t{5}};
  treq_k_b.to_pf_keys_ = {"3", "4", "5"};

  auto should_treq_k = transfer_request_keys{};
  should_treq_k.from_nloc_key_ = key64_t{26};
  should_treq_k.from_pf_key_ = "z";
  should_treq_k.profile_ = "p";
  should_treq_k.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}, key64_t{5},
                                 key64_t{4}};
  should_treq_k.to_pf_keys_ = {"1", "2", "3", "5", "4"};

  auto is_trek_k = merge(treq_k_a, treq_k_b);

  ASSERT_EQ(is_trek_k, should_treq_k);
}

TEST(footpaths, merge_id_transfer_request_keys) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  treq_k_a.from_nloc_key_ = key64_t{26};
  treq_k_a.from_pf_key_ = "z";
  treq_k_a.profile_ = "p";
  treq_k_a.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}, key64_t{5}};
  treq_k_a.to_pf_keys_ = {"1", "2", "3", "5"};

  auto treq_k_b = transfer_request_keys{};
  treq_k_b.from_nloc_key_ = key64_t{26};
  treq_k_b.from_pf_key_ = "z";
  treq_k_b.profile_ = "p";
  treq_k_b.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3}, key64_t{5}};
  treq_k_b.to_pf_keys_ = {"1", "2", "3", "5"};

  auto should_treq_k = transfer_request_keys{};
  should_treq_k.from_nloc_key_ = key64_t{26};
  should_treq_k.from_pf_key_ = "z";
  should_treq_k.profile_ = "p";
  should_treq_k.to_nloc_keys_ = {key64_t{1}, key64_t{2}, key64_t{3},
                                 key64_t{5}};
  should_treq_k.to_pf_keys_ = {"1", "2", "3", "5"};

  auto is_treq_k = merge(treq_k_a, treq_k_b);

  ASSERT_EQ(is_treq_k, should_treq_k);
}

TEST(footpaths, merge_different_from_nloc_keys) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  auto treq_k_b = transfer_request_keys{};

  treq_k_a.from_nloc_key_ = key64_t{1};

  ASSERT_ANY_THROW(merge(treq_k_a, treq_k_b));
}

TEST(footpaths, merge_different_from_pf_keys) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  auto treq_k_b = transfer_request_keys{};

  treq_k_a.from_pf_key_ = "a";

  ASSERT_ANY_THROW(merge(treq_k_a, treq_k_b));
}

TEST(footpaths, merge_different_profiles) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  auto treq_k_b = transfer_request_keys{};

  treq_k_a.profile_ = "a";

  ASSERT_ANY_THROW(merge(treq_k_a, treq_k_b));
}

TEST(footpaths, merge_unmatched_nloc_and_pf_in_a) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  auto treq_k_b = transfer_request_keys{};

  treq_k_a.to_nloc_keys_ = {key64_t{1}};

  ASSERT_ANY_THROW(merge(treq_k_a, treq_k_b));
}

TEST(footpaths, merge_unmatched_nloc_and_pf_in_b) {
  using namespace motis::footpaths;

  auto treq_k_a = transfer_request_keys{};
  auto treq_k_b = transfer_request_keys{};

  treq_k_b.to_nloc_keys_ = {key64_t{1}};

  ASSERT_ANY_THROW(merge(treq_k_a, treq_k_b));
}
