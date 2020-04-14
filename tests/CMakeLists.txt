add_executable(all_test main.cc meb_test.cc ../meb.cc)
target_link_libraries(all_test gtest)
add_test(TEST all_test)