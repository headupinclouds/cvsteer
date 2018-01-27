add_library(cvsteer
  SteerableFilters.cpp 
  SteerableFilters.h 
  SteerableFiltersG2.cpp
  SteerableFiltersG2.h
  SteerableFiltersG4.cpp
  SteerableFiltersG4.h
  )

target_link_libraries(cvsteer PUBLIC ${OpenCV_LIBS})

target_include_directories(cvsteer PUBLIC
  "$<BUILD_INTERFACE:${CVSTEER_ROOT_DIR}>"
)

### Install ###
set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

set(config_install_dir "lib/cmake/${PROJECT_NAME}")
set(include_install_dir "include")
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${version_config}" COMPATIBILITY SameMajorVersion
)

# Note: use 'TARGETS_EXPORT_NAME'
configure_file("../cmake/Config.cmake.in" "${project_config}" @ONLY)

install(
    TARGETS ${PROJECT_NAME}
    EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib"
    RUNTIME DESTINATION "bin"
    INCLUDES DESTINATION "${include_install_dir}/${PROJECT_NAME}"
)

install(
    FILES
    cvsteer.h
    SteerableFilters.h 
    SteerableFiltersG2.h
    SteerableFiltersG4.h
    DESTINATION "${include_install_dir}/${PROJECT_NAME}"
)

install(
    FILES "${project_config}" "${version_config}"
    DESTINATION "${config_install_dir}"
)

install(
    EXPORT "${TARGETS_EXPORT_NAME}"
    NAMESPACE "${namespace}"
    DESTINATION "${config_install_dir}"
)

