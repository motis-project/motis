#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>

#include "motis/core/common/fws_multimap.h"

namespace motis {

template <typename T>
void check_result(std::vector<T> const& ref,
                  fws_multimap_entry<T> const& result) {
  if (ref.size() != result.size() && result.size() < 10) {
    std::cout << "Invalid result:\n  Expected: ";
    std::copy(begin(ref), end(ref), std::ostream_iterator<T>(std::cout, " "));
    std::cout << "\n  Result:   ";
    std::copy(begin(result), end(result),
              std::ostream_iterator<T>(std::cout, " "));
    std::cout << '\n';
  }
  ASSERT_EQ(ref.size(), result.size());
  for (auto i = 0UL; i < ref.size(); ++i) {
    EXPECT_EQ(ref[i], result[i]);
  }
}

TEST(fws_multimap_test, multimap_simple) {
  fws_multimap<int> m;

  m.push_back(1);
  m.push_back(2);
  m.finish_key();

  m.push_back(3);
  m.finish_key();

  m.push_back(4);
  m.push_back(5);
  m.push_back(6);
  m.finish_key();

  m.finish_map();

  ASSERT_EQ(3 + 1, m.index_size());
  check_result({1, 2}, m[0]);
  check_result({3}, m[1]);
  check_result({4, 5, 6}, m[2]);

  check_result({1, 2}, *begin(m));
  check_result({3}, *(begin(m) + 1));
  check_result({4, 5, 6}, *(begin(m) + 2));
  EXPECT_EQ(end(m), begin(m) + 3);
}

TEST(fws_multimap_test, multimap_empty_key) {
  fws_multimap<int> m;

  m.push_back(1);
  m.push_back(2);
  m.finish_key();

  m.finish_key();

  m.push_back(4);
  m.push_back(5);
  m.push_back(6);
  m.finish_key();

  m.finish_map();

  ASSERT_EQ(3 + 1, m.index_size());
  check_result({1, 2}, m[0]);
  check_result({}, m[1]);
  check_result({4, 5, 6}, m[2]);

  check_result({1, 2}, *begin(m));
  check_result({}, *(begin(m) + 1));
  check_result({4, 5, 6}, *(begin(m) + 2));
  EXPECT_EQ(end(m), begin(m) + 3);
}

TEST(fws_multimap_test, shared_idx_multimap_simple) {
  fws_multimap<int> base;
  shared_idx_fws_multimap<int> shared{base.index_};

  base.push_back(1);
  base.push_back(2);
  shared.push_back(10);
  shared.push_back(20);
  base.finish_key();
  shared.finish_key();

  base.push_back(3);
  shared.push_back(30);
  shared.finish_key();
  base.finish_key();

  base.push_back(4);
  base.push_back(5);
  base.push_back(6);
  base.finish_key();
  shared.push_back(40);
  shared.push_back(50);
  shared.push_back(60);
  shared.finish_key();

  base.finish_map();
  shared.finish_map();

  ASSERT_EQ(3 + 1, base.index_size());
  check_result({1, 2}, base[0]);
  check_result({3}, base[1]);
  check_result({4, 5, 6}, base[2]);

  check_result({1, 2}, *begin(base));
  check_result({3}, *(begin(base) + 1));
  check_result({4, 5, 6}, *(begin(base) + 2));
  EXPECT_EQ(end(base), begin(base) + 3);

  ASSERT_EQ(3 + 1, shared.index_size());
  check_result({10, 20}, shared[0]);
  check_result({30}, shared[1]);
  check_result({40, 50, 60}, shared[2]);

  check_result({10, 20}, *begin(shared));
  check_result({30}, *(begin(shared) + 1));
  check_result({40, 50, 60}, *(begin(shared) + 2));
  EXPECT_EQ(end(shared), begin(shared) + 3);
}

TEST(fws_multimap_test, nested_multimap_simple) {
  fws_multimap<int> base;

  base.push_back(1);
  base.push_back(2);
  base.finish_key();

  base.push_back(3);
  base.finish_key();

  base.push_back(4);
  base.push_back(5);
  base.push_back(6);
  base.finish_key();

  base.finish_map();

  nested_fws_multimap<int> nested{base.index_};

  // 1
  nested.push_back(11);
  nested.push_back(12);
  nested.push_back(13);
  nested.finish_nested_key();
  // 2
  nested.push_back(21);
  nested.push_back(22);
  nested.finish_nested_key();
  nested.finish_base_key();

  // 3
  nested.push_back(31);
  nested.push_back(32);
  nested.push_back(33);
  nested.finish_nested_key();
  nested.finish_base_key();

  // 4
  nested.push_back(41);
  nested.push_back(42);
  nested.finish_nested_key();
  // 5
  nested.push_back(51);
  nested.finish_nested_key();
  // 6
  nested.finish_nested_key();
  nested.finish_base_key();

  nested.finish_map();

  /*
  std::cout << "\n\n== base index ==" << std::endl;
  for (auto i = 0UL; i < base.index_.size(); ++i) {
    std::cout << std::setw(2) << i << ": " << base.index_[i] << std::endl;
  }

  std::cout << "\n\n== base data ==" << std::endl;
  for (auto i = 0UL; i < base.data_.size(); ++i) {
    std::cout << std::setw(2) << i << ": " << base.data_[i] << std::endl;
  }

  std::cout << "\n\n== nested index ==" << std::endl;
  for (auto i = 0UL; i < nested.index_.size(); ++i) {
    std::cout << std::setw(2) << i << ": " << nested.index_[i] << std::endl;
  }

  std::cout << "\n\n== nested data ==" << std::endl;
  for (auto i = 0UL; i < nested.data_.size(); ++i) {
    std::cout << std::setw(2) << i << ": " << nested.data_[i] << std::endl;
  }
  */

  ASSERT_EQ(3 + 1, base.index_size());
  {
    SCOPED_TRACE("base");
    check_result({1, 2}, base[0]);
    check_result({3}, base[1]);
    check_result({4, 5, 6}, base[2]);
  }

  {
    SCOPED_TRACE("nested.at(0, 0)");
    check_result({11, 12, 13}, nested.at(0, 0));
  }
  {
    SCOPED_TRACE("nested.at(0, 1)");
    check_result({21, 22}, nested.at(0, 1));
  }
  {
    SCOPED_TRACE("nested.at(1, 0)");
    check_result({31, 32, 33}, nested.at(1, 0));
  }
  {
    SCOPED_TRACE("nested.at(2, 0)");
    check_result({41, 42}, nested.at(2, 0));
  }
  {
    SCOPED_TRACE("nested.at(2, 1)");
    check_result({51}, nested.at(2, 1));
  }
  {
    SCOPED_TRACE("nested.at(2, 2)");
    check_result({}, nested.at(2, 2));
  }
}

}  // namespace motis
