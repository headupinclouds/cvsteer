add_library(cvsteer::cvsteer ALIAS cvsteer)

hunter_add_package(GTest)
find_package(GTest CONFIG REQUIRED)

add_executable(cvsteer-test test.cpp Pterois_volitans_Manado-e_edit_smallest.h edges.h linesDark.h linesBright.h)
target_link_libraries(cvsteer-test cvsteer::cvsteer GTest::main)

add_test(cvsteer-test cvsteer-test)

install(TARGETS cvsteer-test DESTINATION bin)
