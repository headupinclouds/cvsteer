add_library(cvsteer 
  SteerableFilters.cpp 
  SteerableFilters.h 
  SteerableFiltersG2.cpp
  SteerableFiltersG2.h
  SteerableFiltersG4.cpp
  SteerableFiltersG4.h
  )

target_link_libraries(cvsteer ${OpenCV_LIBS})
