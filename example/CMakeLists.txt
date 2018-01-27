add_library(cvsteer::cvsteer ALIAS cvsteer)

add_executable(cvsteer-run steer.cpp)
target_link_libraries(cvsteer-run cvsteer::cvsteer)

install(TARGETS cvsteer-run DESTINATION bin)
