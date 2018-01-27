#!/bin/bash

# Note: When using clang-format the ${CVSTEER}/.clang-format file
# seems to be ignored if the root directory to this script is not
# specified relative to the working directory:
#
# Suggested usage:
#
# cd ${CVSTEER}
# ./bin/cvsteer-format.sh .

# Find all internal files, making sure to exlude 3rdparty subprojects
function find_sources()
{
    NAMES=(-name "*.h" -or -name "*.cpp" -or -name "*.hpp")
    find $1 ${NAMES[@]}
}

input_dir=$1

echo ${input_dir}

find_sources ${input_dir} | while read name
do
    echo $name
    clang-format -i -style=file ${name}
done
