add_executable(cvsteer_test steer.cpp)
target_link_libraries(cvsteer_test cvsteer ${OpenCV_LIBS})

install(TARGETS cvsteer_test DESTINATION bin)